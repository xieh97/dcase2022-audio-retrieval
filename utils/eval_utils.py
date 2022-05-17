import dbm
import os
import shelve
from dbm import dumb

import numpy as np
import pandas as pd
import torch

from utils import criterion_utils, data_utils, model_utils

dbm._defaultmod = dumb
dbm._modules = {"dbm.dumb": dumb}


def measure(rid_list, rid2pred_sims, rid2pred_pids, pid2rid):
    # Retrieval metrics over samples (audio or caption instances)
    rid2pid_R1, rid2pid_R5, rid2pid_R10, rid2pid_mAP10 = [], [], [], []

    for rind, rid in enumerate(rid_list):
        preds = rid2pred_sims[rind]
        target = np.array([((pid2rid.get(pid, None) == rid) or (pid2rid.get(rid, None) == pid))
                           for pid in rid2pred_pids[rind]])

        desc_indices = np.argsort(preds, axis=-1)[::-1]
        target = np.take_along_axis(arr=target, indices=desc_indices, axis=-1)

        recall_at_1 = np.sum(target[:1], dtype=float) / np.sum(target, dtype=float)
        recall_at_5 = np.sum(target[:5], dtype=float) / np.sum(target, dtype=float)
        recall_at_10 = np.sum(target[:10], dtype=float) / np.sum(target, dtype=float)

        rid2pid_R1.append(recall_at_1)
        rid2pid_R5.append(recall_at_5)
        rid2pid_R10.append(recall_at_10)

        positions = np.arange(1, 11, dtype=float)[target[:10] > 0]

        if len(positions) > 0:
            precisions = np.divide(np.arange(1, len(positions) + 1, dtype=float), positions)
            avg_precision = np.sum(precisions, dtype=float) / np.sum(target, dtype=float)
            rid2pid_mAP10.append(avg_precision)
        else:
            rid2pid_mAP10.append(0.0)

    print("R1: {:.3f}".format(np.mean(rid2pid_R1)),
          "R5: {:.3f}".format(np.mean(rid2pid_R5)),
          "R10: {:.3f}".format(np.mean(rid2pid_R10)),
          "mAP10: {:.3f}".format(np.mean(rid2pid_mAP10)), end="\n")


def predict(conf, ckp_fpath):
    data_conf = conf["data_conf"]
    param_conf = conf["param_conf"]
    model_params = conf[param_conf["model"]]

    # Load data
    eval_ds = data_utils.load_data(data_conf["eval_data"])

    # Restore model checkpoint
    model = model_utils.get_model(model_params, eval_ds.text_vocab)
    model = model_utils.restore(model, ckp_fpath)
    print(model)

    model.eval()

    # Retrieve audio files for captions
    for name, ds in zip(["eval"], [eval_ds]):

        # Process captions
        cid2fid, cid2caption, cap2vec, fid2fname = {}, {}, {}, {}

        for idx in ds.text_data.index:
            item = ds.text_data.iloc[idx]

            text_len = len(item[ds.text_col])

            text_vec = np.array([ds.text_vocab(token) for token in item[ds.text_col]])
            text_vec = torch.as_tensor(text_vec)
            text_vec = torch.unsqueeze(text_vec, dim=0)

            cid2fid[item["cid"]] = item["fid"]
            cid2caption[item["cid"]] = item["original"]
            cap2vec[item["cid"]] = (text_vec, text_len)
            fid2fname[item["fid"]] = item["fname"]

        # Compute pairwise cross-modal similarities
        sim_fpath = os.path.join(ckp_fpath, f"{name}_xmodal_Sim.db")
        with shelve.open(filename=sim_fpath, flag="n", protocol=2) as stream:
            for ref_fid in ds.text_data["fid"].unique():
                group_sims = {}

                # Encode audio data
                audio_vec = torch.as_tensor(ds.audio_data[ref_fid][()])
                audio_vec = torch.unsqueeze(audio_vec, dim=0)
                audio_embed = model.audio_branch(audio_vec)[0]

                for cand_cid in cap2vec:
                    # Encode text data
                    text_vec, text_len = cap2vec[cand_cid]
                    text_embed = model.text_branch(text_vec, [text_len])[0]

                    xmodal_S = criterion_utils.score(audio_embed, text_embed)
                    group_sims[cand_cid] = xmodal_S.item()

                stream[ref_fid] = group_sims
        print("Save", sim_fpath)

        # Compute retrieval metrics
        with shelve.open(filename=sim_fpath, flag="r", protocol=2) as stream:
            cid2predA_sims, cid2predA_fids = {}, {}  # Caption2Audio retrieval

            for fid in stream:
                group_sims = stream[fid]
                for cid in group_sims:
                    if cid2predA_sims.get(cid, None) is None:
                        cid2predA_sims[cid] = [group_sims[cid]]
                        cid2predA_fids[cid] = [fid]
                    else:
                        cid2predA_sims[cid].append(group_sims[cid])
                        cid2predA_fids[cid].append(fid)

            cid_list = [cid for cid in cid2predA_sims]
            cid2predA_sims = [cid2predA_sims[cid] for cid in cid_list]
            cid2predA_fids = [cid2predA_fids[cid] for cid in cid_list]

            print("Caption2Audio retrieval")
            measure(cid_list, cid2predA_sims, cid2predA_fids, cid2fid)

            # Output csv files
            predA_rows = []

            sorted_indexes = np.argsort(cid2predA_sims, axis=-1)

            for cind, cid in enumerate(cid_list):

                predA_fids = [cid2predA_fids[cind][find] for find in sorted_indexes[cind][::-1]]

                cid_refA, cid_predA = [cid2caption[cid]], [cid2caption[cid]]

                for fid in predA_fids:
                    cid_predA.append(fid2fname[fid])
                    if cid2fid[cid] == fid:
                        cid_refA.append(fid2fname[fid])

                predA_rows.append(cid_predA[:11])

            predA_rows = pd.DataFrame(data=predA_rows,
                                      columns=["caption", "file_name_1", "file_name_2",
                                               "file_name_3", "file_name_4", "file_name_5", "file_name_6",
                                               "file_name_7", "file_name_8", "file_name_9", "file_name_10"])
            predA_fpath = os.path.join(ckp_fpath, f"{name}.output.csv")
            predA_rows.to_csv(predA_fpath, index=False)
            print("Save", predA_fpath, end="\n\n")
