import pandas as pd
import torch
import torchmetrics


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
    # Initialize retrieval metrics
    R1 = torchmetrics.RetrievalRecall(empty_target_action="neg", compute_on_step=False, k=1)
    R5 = torchmetrics.RetrievalRecall(empty_target_action="neg", compute_on_step=False, k=5)
    R10 = torchmetrics.RetrievalRecall(empty_target_action="neg", compute_on_step=False, k=10)
    mAP10 = torchmetrics.RetrievalMAP(empty_target_action="neg", compute_on_step=False)

    gt_items = load_clotho_csv(gt_csv)
    pred_items = load_output_csv(pred_csv)

    for i, cap in enumerate(gt_items):
        gt_fname = gt_items[cap]
        pred_fnames = pred_items[cap]

        preds = torch.as_tensor([1.0 / (pred_fnames.index(pred) + 1) for pred in pred_fnames],
                                dtype=torch.float)
        targets = torch.as_tensor([gt_fname == pred for pred in pred_fnames], dtype=torch.bool)
        indexes = torch.as_tensor([i for pred in pred_fnames], dtype=torch.long)

        # Update retrieval metrics
        R1(preds, targets, indexes=indexes)
        R5(preds, targets, indexes=indexes)
        R10(preds, targets, indexes=indexes)
        mAP10(preds[:10], targets[:10], indexes=indexes[:10])

    metrics = {
        "R1": R1.compute().item(),  # 0.03
        "R5": R5.compute().item(),  # 0.11
        "R10": R10.compute().item(),  # 0.19
        "mAP10": mAP10.compute().item()  # 0.07
    }

    for key in metrics:
        print(key, "{:.2f}".format(metrics[key]))


gt_csv = "test.gt.csv"  # ground truth for Clotho evaluation data
pred_csv = "test.output.csv"  # baseline system retrieved output for Clotho evaluation data

retrieval_metrics(gt_csv, pred_csv)
