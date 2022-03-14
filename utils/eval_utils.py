import os

import pandas as pd
import torch

from utils import data_utils, model_utils


def transform(model, dataset, index, device=None):
    audio, query, info = dataset[index]

    audio = torch.unsqueeze(audio, dim=0).to(device=device)
    query = torch.unsqueeze(query, dim=0).to(device=device)

    audio_emb, query_emb = model(audio, query, [query.size(-1)])

    audio_emb = torch.squeeze(audio_emb, dim=0).to(device=device)
    query_emb = torch.squeeze(query_emb, dim=0).to(device=device)

    return audio_emb, query_emb, info


def audio_retrieval(model, caption_dataset, K=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)

    model.eval()

    with torch.no_grad():

        fid_embs, fid_fnames = {}, {}
        cid_embs, cid_infos = {}, {}

        # Encode audio signals and captions
        for cap_ind in range(len(caption_dataset)):
            audio_emb, query_emb, info = transform(model, caption_dataset, cap_ind, device)

            fid_embs[info["fid"]] = audio_emb
            fid_fnames[info["fid"]] = info["fname"]

            cid_embs[info["cid"]] = query_emb
            cid_infos[info["cid"]] = info

        # Stack audio embeddings
        audio_embs, fnames = [], []
        for fid in fid_embs:
            audio_embs.append(fid_embs[fid])
            fnames.append(fid_fnames[fid])

        audio_embs = torch.vstack(audio_embs)  # dim [N, E]

        # Compute similarities
        output_rows = []
        for cid in cid_embs:

            sims = torch.mm(torch.vstack([cid_embs[cid]]), audio_embs.T).flatten().to(device=device)

            sorted_idx = torch.argsort(sims, dim=-1, descending=True)

            csv_row = [cid_infos[cid]["caption"]]  # caption
            for idx in sorted_idx[:K]:  # top-K retrieved fnames
                csv_row.append(fnames[idx])

            output_rows.append(csv_row)

        return output_rows


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

    # Retrieve audio files for evaluation captions
    for split in ["test"]:
        output = audio_retrieval(model, caption_datasets[split], K=10)

        csv_fields = ["caption",
                      "file_name_1",
                      "file_name_2",
                      "file_name_3",
                      "file_name_4",
                      "file_name_5",
                      "file_name_6",
                      "file_name_7",
                      "file_name_8",
                      "file_name_9",
                      "file_name_10"]

        output = pd.DataFrame(data=output, columns=csv_fields)
        output.to_csv(os.path.join(checkpoint_dir, "{}.output.csv".format(split)),
                      index=False)
        print("Saved", "{}.output.csv".format(split))
