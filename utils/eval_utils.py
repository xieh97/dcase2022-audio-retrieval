import os
import pickle
import time

from utils import data_utils, metric_utils, model_utils


def eval_checkpoint(config, checkpoint_dir):
    # Load config
    training_config = config["training"]

    # Load evaluation
    caption_datasets, vocabulary = data_utils.load_data(config["eval_data"])

    # Initialize a model instance
    model_config = config[training_config["model"]]
    model = model_utils.get_model(model_config, vocabulary)
    print(model)

    # Restore model states
    model = model_utils.restore(model, checkpoint_dir)
    model.eval()

    # Compute and save retrieval & sed metrics
    for split in ["test", "val", "train"]:
        print(time.time(), "Calculating", split, "eval_metrics ...")

        retrieval_metrics = metric_utils.retrieval_metrics(model, caption_datasets[split])

        # Save retrieval metrics
        result_file = "eval_{0}.pkl".format(split)
        with open(os.path.join(checkpoint_dir, result_file), "wb") as store:
            pickle.dump({
                "retrieval_metrics": retrieval_metrics
            }, store)
        print("Saved", split, "eval_metrics to", checkpoint_dir, result_file)

        # Print audio-caption retrieval metrics
        for key in retrieval_metrics:
            print(split, key, "{:.2f}".format(retrieval_metrics[key]))
