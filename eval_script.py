import numpy as np
import pandas as pd
from astropy.stats import jackknife


def load_clotho_csv(fpath):
    cap2fname = {}

    rows = pd.read_csv(fpath)
    rows = [list(row) for row in rows.values]

    for row in rows:
        for cap in row[1:]:  # captions
            cap2fname[cap] = row[0]

    return cap2fname


def load_output_csv(fpath):
    cap2fnames = {}

    rows = pd.read_csv(fpath)
    rows = [list(row) for row in rows.values]

    for row in rows:
        cap2fnames[row[0]] = row[1:]

    return cap2fnames


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

    # Jackknife estimation with 95% confidence interval on evaluation metrics
    estimate, bias, std_err, conf_interval = jackknife.jackknife_stats(np.asarray(R1), np.mean, 0.95)
    print("R1", f"{estimate:.2f}", f"[{conf_interval[0]:.2f}, {conf_interval[1]:.2f}]")

    estimate, bias, std_err, conf_interval = jackknife.jackknife_stats(np.asarray(R5), np.mean, 0.95)
    print("R5", f"{estimate:.2f}", f"[{conf_interval[0]:.2f}, {conf_interval[1]:.2f}]")

    estimate, bias, std_err, conf_interval = jackknife.jackknife_stats(np.asarray(R10), np.mean, 0.95)
    print("R10", f"{estimate:.2f}", f"[{conf_interval[0]:.2f}, {conf_interval[1]:.2f}]")

    estimate, bias, std_err, conf_interval = jackknife.jackknife_stats(np.asarray(mAP10), np.mean, 0.95)
    print("mAP10", f"{estimate:.2f}", f"[{conf_interval[0]:.2f}, {conf_interval[1]:.2f}]")


gt_csv = "test.gt.csv"  # ground truth for Clotho-evaluation data
pred_csv = "test.output.csv"  # baseline system retrieved output for Clotho-evaluation data

retrieval_metrics(gt_csv, pred_csv)
