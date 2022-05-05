import os
import random
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

random.seed(16)
torch.manual_seed(16)
numpy.random.seed(16)


def exec_trial(conf, ckp_dir=None):
    data_conf = conf["data_conf"]
    param_conf = conf["param_conf"]

    train_ds = data_utils.load_data(data_conf["train_data"])
    train_dl = DataLoader(dataset=train_ds, batch_size=param_conf["batch_size"],
                          shuffle=True, collate_fn=data_utils.collate_fn)

    val_ds = data_utils.load_data(data_conf["val_data"])
    val_dl = DataLoader(dataset=val_ds, batch_size=param_conf["batch_size"],
                        shuffle=True, collate_fn=data_utils.collate_fn)

    eval_ds = data_utils.load_data(data_conf["eval_data"])
    eval_dl = DataLoader(dataset=eval_ds, batch_size=param_conf["batch_size"],
                         shuffle=True, collate_fn=data_utils.collate_fn)

    model_params = conf[param_conf["model"]]
    model = model_utils.get_model(model_params, train_ds.text_vocab)
    print(model)

    criterion_params = conf[param_conf["criterion"]]
    criterion = getattr(criterion_utils, criterion_params["name"], None)(**criterion_params["args"])

    optimizer_params = conf[param_conf["optimizer"]]
    optimizer = getattr(optim, optimizer_params["name"], None)(
        model.parameters(), **optimizer_params["args"])

    lr_params = conf[param_conf["lr_scheduler"]]
    lr_scheduler = getattr(optim.lr_scheduler, lr_params["name"], "ReduceLROnPlateau")(
        optimizer, **lr_params["args"])

    if ckp_dir is not None:
        model_state, optimizer_state = torch.load(os.path.join(ckp_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    for epoch in range(param_conf["epochs"] + 1):
        if epoch > 0:
            model_utils.train(model, train_dl, criterion, optimizer)

        epoch_results = {}
        epoch_results["train_loss"] = model_utils.eval(model, train_dl, criterion)
        epoch_results["val_loss"] = model_utils.eval(model, val_dl, criterion)
        epoch_results["eval_loss"] = model_utils.eval(model, eval_dl, criterion)

        # Reduce learning rate w.r.t validation loss
        lr_scheduler.step(epoch_results["val_loss"])

        # Save the model to the trial directory: local_dir/exp_name/trial_name/checkpoint_<step>
        with tune.checkpoint_dir(step=epoch) as ckp_dir:
            path = os.path.join(ckp_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        # Send the current statistics back to the Ray cluster
        tune.report(**epoch_results)


# Main
if __name__ == "__main__":
    # Load parameters
    with open("conf.yaml", "rb") as stream:
        conf = yaml.full_load(stream)

    ray_conf = conf["ray_conf"]  # parameters for ray-tune clusters

    # Initialize ray-tune clusters
    ray.init(**ray_conf["init_args"])

    # Initialize a trial stopper
    trial_stopper = getattr(tune.stopper, ray_conf["trial_stopper"], TrialPlateauStopper)(
        **ray_conf["stopper_args"])

    # Initialize a progress reporter
    trial_reporter = getattr(tune.progress_reporter, ray_conf["reporter"], CLIReporter)()

    trial_reporter.add_metric_column(metric="train_loss")
    trial_reporter.add_metric_column(metric="val_loss")
    trial_reporter.add_metric_column(metric="eval_loss")


    def trial_name_creator(trial):
        trial_name = "{0}_{1}".format(conf["param_conf"]["model"], trial.trial_id)
        return trial_name


    def trial_dirname_creator(trial):
        trial_dirname = "{0}_{1}".format(time.strftime("%Y-%m-%d"), trial.trial_id)
        return trial_dirname


    # Execute trials - local_dir/exp_name/trial_name
    analysis = tune.run(
        run_or_experiment=exec_trial,
        metric=ray_conf["stopper_args"]["metric"],
        mode=ray_conf["stopper_args"]["mode"],
        name=conf["trial_series"],
        stop=trial_stopper,
        config=conf,
        resources_per_trial={
            "cpu": 1,
            "gpu": ray_conf["init_args"]["num_gpus"] / ray_conf["init_args"]["num_cpus"]
        },
        num_samples=1,
        local_dir=conf["trial_base"],
        # search_alg=search_alg,
        # scheduler=scheduler,
        keep_checkpoints_num=None,
        checkpoint_score_attr=None,
        progress_reporter=trial_reporter,
        log_to_file=False,
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

    # Check best trials and checkpoints
    best_trial = analysis.get_best_trial(
        metric=ray_conf["stopper_args"]["metric"],
        mode=ray_conf["stopper_args"]["mode"],
        scope="all")

    best_ckp = analysis.get_best_checkpoint(
        trial=best_trial,
        metric=ray_conf["stopper_args"]["metric"],
        mode=ray_conf["stopper_args"]["mode"])

    print("Best trial:", best_trial.trial_id)
    print("Best checkpoint:", best_ckp)

    # Evaluate at the best checkpoint
    eval_utils.predict(conf, best_ckp)
