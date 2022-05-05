import base64
import glob
import os
import pickle

import pandas as pd
from mutagen.wave import WAVE

from scripts import global_params
from utils import word_utils

# %% 1. Check audio clips

audio_fids, audio_fnames, audio_durations = {}, {}, {}

for split in global_params["audio_splits"]:

    fid_fnames, fnames, durations = {}, [], []

    audio_dir = os.path.join(global_params["dataset_dir"], split)

    for fpath in glob.glob(r"{}/*.wav".format(audio_dir)):
        try:
            clip = WAVE(fpath)

            if clip.info.length > 0.0:
                fid = base64.urlsafe_b64encode(os.urandom(8)).decode("ascii")
                fname = os.path.basename(fpath)

                fid_fnames[fid] = fname
                fnames.append(fname)
                durations.append(clip.info.length)
        except:
            print("Error audio file:", fpath)

    assert len(fid_fnames) == len(fnames)

    audio_fids[split] = fid_fnames
    audio_fnames[split] = fnames
    audio_durations[split] = durations

# Save audio info
with open(os.path.join(global_params["dataset_dir"], "audio_info.pkl"), "wb") as store:
    pickle.dump({"audio_fids": audio_fids, "audio_fnames": audio_fnames, "audio_durations": audio_durations}, store)
print("Save audio info")

# %% 2. Load and preprocess captions

for split, cap_fname in zip(global_params["audio_splits"], global_params["caption_files"]):

    fid_fnames = audio_fids[split]
    fname_fids = {fid_fnames[fid]: fid for fid in fid_fnames}

    cap_fpath = os.path.join(global_params["dataset_dir"], cap_fname)
    cap_data = pd.read_csv(cap_fpath)

    cap_rows = []

    for i in cap_data.index:
        fname = cap_data.iloc[i].file_name.strip()
        cap_list = [cap_data.iloc[i].get(key) for key in cap_data.columns if key.startswith("caption")]

        fid = fname_fids[fname]

        for cap in cap_list:
            cid = base64.urlsafe_b64encode(os.urandom(8)).decode("ascii")

            text, words = word_utils.clean_text(cap)

            # cid, fid, fname, original, caption, tokens
            cap_rows.append([cid, fid, fname, cap, text, words])

    cap_rows = pd.DataFrame(data=cap_rows, columns=["cid", "fid", "fname", "original", "text", "tokens"])

    cap_fpath = os.path.join(global_params["dataset_dir"], f"{split}_captions.json")
    cap_rows.to_json(cap_fpath)
    print("Save", cap_fpath)

# %% 3. Gather data statistics

# 1) clips
# 2) captions
# 3) word frequencies

vocabulary = set()
word_bags = {}
split_infos = {}

for split in global_params["audio_splits"]:

    fid_fnames = audio_fids[split]
    fname_fids = {fid_fnames[fid]: fid for fid in fid_fnames}

    cap_fpath = os.path.join(global_params["dataset_dir"], f"{split}_captions.json")
    cap_data = pd.read_json(cap_fpath)

    num_clips = len(fid_fnames)
    num_captions = cap_data.cid.size

    bag = []
    for tokens in cap_data["tokens"]:
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
print("Save vocabulary info")
