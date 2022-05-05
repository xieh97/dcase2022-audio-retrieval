import copy
import os

import torch

from models import core


def get_model(params, vocab):
    params = copy.deepcopy(params)

    if params["name"] in ["CRNNWordModel"]:
        word_args = params["text_enc"]["word_enc"]
        word_args["num_word"] = len(vocab)
        word_args["word_embeds"] = vocab.get_weights() if word_args["init"] == "prior" else None

        return getattr(core, params["name"], None)(**params)

    return None


def train(model, data_loader, criterion, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion.to(device=device)
    model.to(device=device)

    model.train()

    for batch_idx, data in enumerate(data_loader, 0):
        audio_vecs, audio_lens, text_vecs, text_lens, add_infos = data

        audio_vecs = audio_vecs.to(device)
        text_vecs = text_vecs.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        audio_embeds, text_embeds = model(audio_vecs, text_vecs, text_lens)
        loss = criterion(audio_embeds, text_embeds, add_infos)
        loss.backward()
        optimizer.step()


def eval(model, data_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion.to(device=device)
    model.to(device=device)

    model.eval()

    eval_loss, eval_steps = 0.0, 0

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader, 0):
            audio_vecs, audio_lens, text_vecs, text_lens, add_infos = data

            audio_vecs = audio_vecs.to(device)
            text_vecs = text_vecs.to(device)

            audio_embeds, text_embeds = model(audio_vecs, text_vecs, text_lens)
            loss = criterion(audio_embeds, text_embeds, add_infos)
            eval_loss += loss.cpu().numpy()
            eval_steps += 1

    return eval_loss / (eval_steps + 1e-20)


def restore(model, ckp_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state, optimizer_state = torch.load(os.path.join(ckp_dir, "checkpoint"),
                                              map_location=device)
    model.load_state_dict(model_state)
    return model
