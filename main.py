import os
import time

import numpy
import ray
import torch
import torch.optim as optim
import yaml
from ray import tune
from ray.tune.progress_reporter import CLIReporter
from ray.tune.stopper import TrialPlateauStopper
from torch.utils.data import DataLoader

from utils import criterion_utils, data_utils, eval_utils, model_utils

numpy.random.seed(16)


def train(config, checkpoint_dir=None):
    training_config = config["training"]

    text_datasets, vocabulary = data_utils.load_data(config["train_data"])

    text_loaders = {}
    for split in ["train", "val", "test"]:
        _dataset = text_datasets[split]
        _loader = DataLoader(dataset=_dataset, batch_size=training_config["algorithm"]["batch_size"],
                             shuffle=True, collate_fn=data_utils.collate_fn)
        text_loaders[split] = _loader

    model_config = config[training_config["model"]]
    model = model_utils.get_model(model_config, vocabulary)
    print(model)

    alg_config = training_config["algorithm"]

    criterion_config = config[alg_config["criterion"]]
    criterion = getattr(criterion_utils, criterion_config["name"], None)(**criterion_config["args"])

    optimizer_config = config[alg_config["optimizer"]]
    optimizer = getattr(optim, optimizer_config["name"], None)(
        model.parameters(), **optimizer_config["args"]
    )

    lr_scheduler = getattr(optim.lr_scheduler, "ReduceLROnPlateau")(optimizer, **optimizer_config["scheduler_args"])

    if checkpoint_dir is not None:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    for epoch in range(alg_config["epochs"] + 1):
        if epoch > 0:
            model_utils.train(model, optimizer, criterion, text_loaders["train"])

        epoch_results = {}
        for split in ["train", "val", "test"]:
            epoch_results["{0}_loss".format(split)] = model_utils.eval(model, criterion, text_loaders[split])

        # Reduce learning rate based on validation loss
        lr_scheduler.step(epoch_results[config["ray_conf"]["stopper_args"]["metric"]])

        # Save the model to the trial directory: local_dir/exp_name/trial_name/checkpoint_<step>
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        # Send the current statistics back to the Ray cluster
        tune.report(**epoch_results)


# Main
if __name__ == "__main__":
    # Load configurations
    with open("conf.yaml", "rb") as stream:
        conf = yaml.full_load(stream)

    ray_conf = conf["ray_conf"]

    # Initialize a Ray cluster
    ray.init(**ray_conf["init_args"])

    # Initialize a trial stopper
    stopper = getattr(tune.stopper, ray_conf["trial_stopper"], TrialPlateauStopper)(
        **ray_conf["stopper_args"]
    )

    # Initialize a progress reporter
    reporter = getattr(tune.progress_reporter, ray_conf["reporter"], CLIReporter)()
    for _split in ["train", "val", "test"]:
        reporter.add_metric_column(metric="{0}_loss".format(_split))


    def trial_name_creator(trial):
        trial_name = "{0}_{1}".format(
            conf["training"]["model"], trial.trial_id
        )
        return trial_name


    def trial_dirname_creator(trial):
        trial_dirname = "{0}_{1}_{2}".format(
            conf["training"]["model"], trial.trial_id, time.strftime("%Y-%m-%d_%H-%M-%S")
        )
        return trial_dirname


    # Run a Ray cluster - local_dir/exp_name/trial_name
    analysis = tune.run(
        run_or_experiment=train,
        metric=ray_conf["stopper_args"]["metric"],
        mode=ray_conf["stopper_args"]["mode"],
        name=conf["experiment"],
        stop=stopper,
        config=conf,
        resources_per_trial={
            "cpu": 1,
            "gpu": ray_conf["init_args"]["num_gpus"] / ray_conf["init_args"]["num_cpus"]
        },
        num_samples=1,
        local_dir=conf["output_path"],
        # search_alg=search_alg,
        # scheduler=scheduler,
        keep_checkpoints_num=None,
        checkpoint_score_attr=None,
        progress_reporter=reporter,
        log_to_file=True,
        trial_name_creator=trial_name_creator,
        trial_dirname_creator=trial_dirname_creator,
        # max_failures=1,
        fail_fast=False,
        # restore="",  # Only makes sense to set if running 1 trial.
        # resume="ERRORED_ONLY",
        queue_trials=False,
        reuse_actors=True,
        raise_on_failed_trial=True
    )

    # Check the best trial and its best checkpoint
    best_trial = analysis.get_best_trial(
        metric=ray_conf["stopper_args"]["metric"],
        mode=ray_conf["stopper_args"]["mode"],
        scope="all"
    )
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial,
        metric=ray_conf["stopper_args"]["metric"],
        mode=ray_conf["stopper_args"]["mode"]
    )
    print("Best trial:", best_trial.trial_id)
    print("Best checkpoint:", best_checkpoint)

    # Evaluate at the best checkpoint
    eval_utils.eval_checkpoint(conf, best_checkpoint)
