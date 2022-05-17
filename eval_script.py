import numpy as np
import pandas as pd


def load_clotho_csv(fpath):
    caption_fname = {}

    rows = pd.read_csv(fpath)
    rows = [list(row) for row in rows.values]

    for row in rows:
        for cap in row[1:]:  # captions
            caption_fname[cap] = row[0]

    return caption_fname


def load_output_csv(fpath):
    caption_fnames = {}

    rows = pd.read_csv(fpath)
    rows = [list(row) for row in rows.values]

    for row in rows:
        caption_fnames[row[0]] = row[1:]

    return caption_fnames


def retrieval_metrics(gt_csv, pred_csv):
    R1, R5, R10, mAP10 = [], [], [], []

    gt_items = load_clotho_csv(gt_csv)
    pred_items = load_output_csv(pred_csv)

    for i, cap in enumerate(gt_items):
        gt_fname = gt_items[cap]
        pred_fnames = pred_items[cap]

        preds = np.asarray([gt_fname == pred for pred in pred_fnames])

        # Metric calculation
        # Given that only one correct audio file for each caption query
        R1.append(np.sum(preds[:1], dtype=float))
        R5.append(np.sum(preds[:5], dtype=float))
        R10.append(np.sum(preds[:10], dtype=float))

        positions = np.arange(1, 11, dtype=float)[preds[:10] > 0]
        if len(positions) > 0:
            precisions = np.divide(np.arange(1, len(positions) + 1, dtype=float), positions)
            avg_precision = np.sum(precisions, dtype=float)
            mAP10.append(avg_precision)
        else:
            mAP10.append(0.0)

    metrics = {
        "R1": np.mean(R1),  # 0.03
        "R5": np.mean(R5),  # 0.11
        "R10": np.mean(R10),  # 0.19
        "mAP10": np.mean(mAP10)  # 0.07
    }

    for key in metrics:
        print(key, "{:.2f}".format(metrics[key]))


gt_csv = "test.gt.csv"  # ground truth for Clotho evaluation data
pred_csv = "test.output.csv"  # baseline system retrieved output for Clotho evaluation data

retrieval_metrics(gt_csv, pred_csv)
