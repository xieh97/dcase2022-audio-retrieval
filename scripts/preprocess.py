import glob
import os
import pickle

import pandas as pd
from mutagen.wave import WAVE

from utils import word_utils

global_params = {
    "dataset_dir": "~/Clotho.v2.1",
    "audio_splits": ["development", "validation", "evaluation"],
    "caption_files": ["clotho_captions_development.csv",
                      "clotho_captions_validation.csv",
                      "clotho_captions_evaluation.csv"],
    "metadata_files": ["clotho_metadata_development.csv",
                       "clotho_metadata_validation.csv",
                       "clotho_metadata_evaluation.csv"]
}

# %% 1. Check audio clips

fid_count = 0
audio_fids, audio_fnames, audio_durations = {}, {}, {}

for split in global_params["audio_splits"]:
    fids, fnames, durations = {}, [], []

    for fpath in glob.glob(r"{}/*.wav".format(os.path.join(global_params["dataset_dir"], split))):
        try:
            clip = WAVE(fpath)

            if clip.info.length > 0.0:
                fid_count += 1
                fname = os.path.basename(fpath)

                fids[fname] = fid_count
                fnames.append(fname)
                durations.append(clip.info.length)
        except:
            print("Error audio file: {}.".format(fpath))
    audio_fids[split] = fids
    audio_fnames[split] = fnames
    audio_durations[split] = durations

# Save audio info
with open(os.path.join(global_params["dataset_dir"], "audio_info.pkl"), "wb") as store:
    pickle.dump({"audio_fids": audio_fids, "audio_fnames": audio_fnames, "audio_durations": audio_durations}, store)
print("Saved audio info")

# %% 2. Load and preprocess captions

cid_count = 0

for split, caption_file in zip(global_params["audio_splits"], global_params["caption_files"]):
    captions = pd.read_csv(os.path.join(global_params["dataset_dir"], caption_file))
    split_fids = audio_fids[split]

    audio_captions = []
    for i in captions.index:
        fname = captions.iloc[i].file_name.strip()
        c1 = captions.iloc[i].caption_1
        c2 = captions.iloc[i].caption_2
        c3 = captions.iloc[i].caption_3
        c4 = captions.iloc[i].caption_4
        c5 = captions.iloc[i].caption_5

        fid = split_fids[fname]

        for cap in [c1, c2, c3, c4, c5]:
            cid_count += 1

            print(split, cid_count, fid, cap)

            text, words = word_utils.clean_text(cap)

            # cid, fid, fname, original, caption, tokens
            audio_captions.append([cid_count, fid, fname, cap, text, words])

    audio_captions = pd.DataFrame(data=audio_captions, columns=["cid", "fid", "fname", "original", "caption", "tokens"])
    audio_captions.to_json(os.path.join(global_params["dataset_dir"], "{}_captions.json".format(split)))
    print("Saved", "{}_captions.json".format(split))

# %% 3. Gather data statistics

# 1) clips
# 2) captions
# 3) word frequencies

vocabulary = set()
word_bags = {}
split_infos = {}

for split in global_params["audio_splits"]:
    captions = pd.read_json(os.path.join(global_params["dataset_dir"], "{}_captions.json".format(split)))
    split_fids = audio_fids[split]

    num_clips = len(split_fids)
    num_captions = captions.caption.size

    bag = []
    for tokens in captions["tokens"]:
        bag.extend(tokens)
        vocabulary = vocabulary.union(tokens)

    num_words = len(bag)
    word_bags[split] = bag
    split_infos[split] = {
        "num_clips": num_clips,
        "num_captions": num_captions,
        "num_words": num_words
    }

# Save vocabulary
with open(os.path.join(global_params["dataset_dir"], "vocab_info.pkl"), "wb") as store:
    pickle.dump({
        "vocabulary": vocabulary,
        "word_bags": word_bags,
        "split_infos": split_infos
    }, store)
print("Saved vocabulary info")
