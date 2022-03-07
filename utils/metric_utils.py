import torch
import torchmetrics


def retrieval_metrics(model, caption_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)

    model.eval()

    # Initialize retrieval metrics
    R1 = torchmetrics.RetrievalRecall(empty_target_action="neg", compute_on_step=False, k=1).to(device=device)
    R5 = torchmetrics.RetrievalRecall(empty_target_action="neg", compute_on_step=False, k=5).to(device=device)
    R10 = torchmetrics.RetrievalRecall(empty_target_action="neg", compute_on_step=False, k=10).to(device=device)
    mAP = torchmetrics.RetrievalMAP(empty_target_action="neg", compute_on_step=False).to(device=device)

    with torch.no_grad():

        fid_embs, cid_embs, cid_fid = {}, {}, {}

        # Encode audio and captions
        for cap_ind in range(len(caption_dataset)):

            audio_emb, query_emb, info = transform(model, caption_dataset, cap_ind, device)

            if fid_embs.get(info["fid"]) is None:
                fid_embs[info["fid"]] = audio_emb

            cid_embs[info["cid"]] = query_emb
            cid_fid[info["cid"]] = info["fid"]

        # Stack audio embeddings
        fid_values = []
        for fid in fid_embs:
            fid_values.append(fid_embs[fid])

        fid_values = torch.vstack(fid_values)  # dim [N, E]

        # Compute similarities
        for cid in cid_embs:
            preds = torch.mm(torch.vstack([cid_embs[cid]]), fid_values.T).flatten().to(device=device)
            targets = torch.as_tensor([cid_fid[cid] == fid for fid in fid_embs], device=device, dtype=torch.bool)
            indexes = torch.as_tensor([cid for fid in fid_embs], device=device, dtype=torch.long)

            # Update retrieval metrics
            R1(preds, targets, indexes=indexes)
            R5(preds, targets, indexes=indexes)
            R10(preds, targets, indexes=indexes)

            sorted_idx = torch.argsort(preds, dim=-1, descending=True)
            mAP(preds[sorted_idx][:3], targets[sorted_idx][:3], indexes=indexes[sorted_idx][:3])

        return {"R1": R1.compute().item(), "R5": R5.compute().item(), "R10": R10.compute().item(),
                "mAP": mAP.compute().item()}


def transform(model, dataset, index, device=None):
    audio, query, info = dataset[index]

    audio = torch.unsqueeze(audio, dim=0).to(device=device)
    query = torch.unsqueeze(query, dim=0).to(device=device)

    audio_emb, query_emb = model(audio, query, [query.size(-1)])

    audio_emb = torch.squeeze(audio_emb, dim=0).to(device=device)
    query_emb = torch.squeeze(query_emb, dim=0).to(device=device)

    return audio_emb, query_emb, info
