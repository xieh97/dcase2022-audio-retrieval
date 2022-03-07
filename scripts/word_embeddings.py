import os
import pickle

import numpy as np

from utils import word_utils

global_params = {
    "dataset_dir": "~/Clotho.v2.1",
    "audio_splits": ["development", "validation", "evaluation"],
}

word_embs = {}
emb_matrix, emb_shape = None, None

# Load vocabulary
with open(os.path.join(global_params["dataset_dir"], "vocab_info.pkl"), "rb") as store:
    vocab_info = pickle.load(store)
    vocabulary = vocab_info["vocabulary"]

# Gather pretrained word embeddings
for word in vocabulary:
    if word != word_utils.UNK_token:
        word_embs[word] = word_utils.word_vector(word)

        if emb_shape is None:
            emb_shape = word_embs[word].shape

        if emb_matrix is None:
            emb_matrix = word_embs[word]
        else:
            emb_matrix = np.vstack((emb_matrix, word_embs[word]))

mean, std = np.mean(emb_matrix, axis=0), np.std(emb_matrix, axis=0)

# Generate UNK_token embedding
dot_product = 1.
UNK_emb = np.zeros_like(mean)

while np.all(np.abs(dot_product) > 0.01):
    UNK_emb = mean + std * np.random.randn(emb_shape[0])
    dot_product = np.dot(emb_matrix, UNK_emb)
    print(dot_product.shape)

word_embs[word_utils.UNK_token] = UNK_emb

# Save pretrained embeddings
with open(os.path.join(global_params["dataset_dir"], "word2vec_emb.pkl"), "wb") as store:
    pickle.dump(word_embs, store)
print("Saved pretrained embeddings info")
