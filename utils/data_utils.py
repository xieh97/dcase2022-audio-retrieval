import os
import pickle

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Vocabulary(object):

    def __init__(self):
        self.word2vec = {}
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        self.weights = None

    def add_word(self, word, word_vector):
        if word not in self.word2idx:
            self.word2vec[word] = word_vector
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def get_word(self, idx):
        return self.idx2word.get(idx, None)

    def get_vec(self, word):
        return self.word2vec.get(word, None)

    def get_weights(self):
        for idx in range(self.idx):
            if self.weights is None:
                self.weights = self.word2vec[self.idx2word[idx]]
            else:
                self.weights = np.vstack((self.weights, self.word2vec[self.idx2word[idx]]))

        return self.weights

    def __call__(self, word):
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class AudioTextDataset(Dataset):

    def __init__(self, **kwargs):
        self.audio_data = kwargs["audio_data"]

        self.text_data = kwargs["text_data"]
        self.text_col = kwargs["text_col"]
        self.text_vocab = kwargs["text_vocab"]

        self.win_shift = kwargs["win_shift"]

        self.kwargs = kwargs  # parameters

    def __getitem__(self, index):
        item = self.text_data.iloc[index]

        audio_vec = torch.as_tensor(self.audio_data[item["fid"]][()])
        text_vec = torch.as_tensor([self.text_vocab(token) for token in item[self.text_col]])

        add_info = {  # additional info
            "cid": item["cid"],
            "fid": item["fid"],
            "fname": item["fname"],
            "caption": item["original"]
        }

        return audio_vec, text_vec, add_info

    def __len__(self):
        return len(self.text_data)


def collate_fn(data_batch):
    """
    :param data_batch: a list of tuples (audio_vec, text_vec, add_info).
    """

    audio_vecs, text_vecs, add_infos = [], [], []

    for a, t, i in data_batch:
        audio_vecs.append(a)
        text_vecs.append(t)
        add_infos.append(i)

    audio_vecs, audio_lens = pad_tensors(audio_vecs, dtype=torch.float)
    text_vecs, text_lens = pad_tensors(text_vecs, dtype=torch.long)
    # audio_labels, _ = pad_tensors(audio_labels, dtype=torch.float)

    return audio_vecs, audio_lens, text_vecs, text_lens, add_infos


def pad_tensors(tensor_list, dtype=torch.float):
    tensor_lens = [tensor.shape for tensor in tensor_list]

    dim_max_lens = tuple(np.max(tensor_lens, axis=0))

    tensor_lens = np.array(tensor_lens)[:, 0]

    padded_tensor = torch.zeros((len(tensor_list),) + dim_max_lens)
    for i, t in enumerate(tensor_list):
        end = tensor_lens[i]
        padded_tensor[i, :end] = t[:end]

    return padded_tensor.to(dtype), tensor_lens


def load_data(conf):
    kwargs = {}

    # Load audio data
    audio_fpath = os.path.join(conf["dataset"], conf["audio_data"])
    audio_data = h5py.File(audio_fpath, "r")
    print("Load", audio_fpath)

    kwargs["audio_data"] = audio_data

    # Load caption data
    caption_fpath = os.path.join(conf["dataset"], conf["caption_data"])
    caption_data = pd.read_json(caption_fpath)
    print("Load", caption_fpath)

    kwargs["text_data"] = caption_data

    # Load word embeddings
    embed_fpath = os.path.join(conf["dataset"], conf["word_embeds"])
    with open(embed_fpath, "rb") as stream:
        word_embeds = pickle.load(stream)
    print("Load", embed_fpath)

    # Build vocabulary
    text_vocab = Vocabulary()
    for word in word_embeds:
        if len(text_vocab) == 0:
            text_vocab.add_word("<pad>", np.zeros_like(word_embeds[word]))
        text_vocab.add_word(word, word_embeds[word])

    kwargs["text_vocab"] = text_vocab

    # Add additional parameters
    kwargs["text_col"] = "tokens"
    kwargs["win_shift"] = 0.02

    # Enclose dataset
    dataset = AudioTextDataset(**kwargs)

    return dataset
