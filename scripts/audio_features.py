import glob
import os
import pickle

import h5py
import librosa
import numpy as np


def log_mel_spectrogram(y,
                        sample_rate=44100,
                        window_length_secs=0.025,
                        hop_length_secs=0.010,
                        num_mels=128,
                        fmin=12.0,
                        fmax=8000,
                        log_offset=0.0):
    """Convert waveform to a log magnitude mel-frequency spectrogram.

    :param y: 1D np.array of waveform data.
    :param sample_rate: The sampling rate of data.
    :param window_length_secs: Duration of each window to analyze.
    :param hop_length_secs: Advance between successive analysis windows.
    :param num_mels: Number of Mel bands.
    :param fmin: Lower bound on the frequencies to be included in the mel spectrum.
    :param fmax: The desired top edge of the highest frequency band.
    :param log_offset: Add this to values when taking log to avoid -Infs.
    :return:
    """
    window_length = int(round(sample_rate * window_length_secs))
    hop_length = int(round(sample_rate * hop_length_secs))
    fft_length = 2 ** int(np.ceil(np.log(window_length) / np.log(2.0)))

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_fft=fft_length, hop_length=hop_length,
                                                     win_length=window_length, n_mels=num_mels, fmin=fmin, fmax=fmax)

    return np.log(mel_spectrogram + log_offset)


def extract_log_mel(params):
    output_file = os.path.join(params["dataset_dir"], params["audio_feats"])

    with h5py.File(output_file, "w") as feature_store:
        for split in params["audio_splits"]:

            subset_dir = os.path.join(params["dataset_dir"], split)
            print(subset_dir)

            for fpath in glob.glob("{}/*.wav".format(subset_dir)):
                try:
                    fname = os.path.basename(fpath)
                    fid = global_params["audio_fids"][split][fname]

                    y, sr = librosa.load(fpath, sr=None, mono=True)
                    log_mel = log_mel_spectrogram(y=y, sample_rate=sr, window_length_secs=0.040,
                                                  hop_length_secs=0.020, num_mels=64,
                                                  log_offset=np.spacing(1))

                    feat = np.vstack(log_mel).transpose()  # [Time, Mel]

                    feature_store[str(fid)] = feat
                    print(fid)
                except:
                    print("Error file: {}.".format(fpath))


#
# %% Pre-computed stuff
#

global_params = {
    "dataset_dir": "~/Clotho.v2.1",
    "audio_splits": ["development", "validation", "evaluation"],
    "audio_feats": "audio_logmel.hdf5"
}

with open(os.path.join(global_params["dataset_dir"], "audio_info.pkl"), "rb") as store:
    global_params["audio_fids"] = pickle.load(store)["audio_fids"]

# Extract log mel features
extract_log_mel(global_params)
