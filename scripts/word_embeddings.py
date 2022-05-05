import os
import pickle

import numpy as np

from scripts import global_params
from utils import word_utils

# %% Collect pre-trained word embeddings

word_embeds = {}
embed_mat, embed_shape = None, None

# Load vocabulary
with open(os.path.join(global_params["dataset_dir"], "vocab_info.pkl"), "rb") as stream:
    vocab_info = pickle.load(stream)
    vocabulary = vocab_info["vocabulary"]

# Gather pretrained word embeddings
for word in vocabulary:
    if word != word_utils.UNK_token:
        word_embeds[word] = word_utils.word_vector(word)

        if embed_shape is None:
            embed_shape = word_embeds[word].shape

        if embed_mat is None:
            embed_mat = word_embeds[word]
        else:
            embed_mat = np.vstack((embed_mat, word_embeds[word]))

mean, std = np.mean(embed_mat, axis=0), np.std(embed_mat, axis=0)

# Generate UNK_token embedding
dot_product = 1.
UNK_embed = np.zeros_like(mean)

while np.all(np.abs(dot_product) > 0.01):
    UNK_embed = mean + std * np.random.randn(embed_shape[0])
    dot_product = np.dot(embed_mat, UNK_embed)
    print(dot_product.shape)

word_embeds[word_utils.UNK_token] = UNK_embed

# Save pretrained embeddings
with open(os.path.join(global_params["dataset_dir"], "word2vec_embeds.pkl"), "wb") as stream:
    pickle.dump(word_embeds, stream)
print("Save", "word2vec_embeds.pkl")
