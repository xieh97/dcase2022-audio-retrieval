## Language-based audio retrieval baseline system for DCASE 2022 Task 6

Welcome to the repository of the language-based audio retrieval baseline system for **task 6b in DCASE 2022 challenge**.
For detailed introduction of this task, please see the [task webpage](https://dcase.community/challenge2022/task-language-based-audio-retrieval).

**2022/12/29 Update:**
The DCASE 2022 Challenge has ended.
Full challenge results for this task can be found in the [result webpage](https://dcase.community/challenge2022/task-language-based-audio-retrieval-results), and a full summary of the task submissions can be found in [our paper](https://arxiv.org/abs/2206.06108):

```
@InProceedings{Xie2022Language,
    author = {Xie, Huang and Lipping, Samuel and Virtanen, Tuomas},
    title = {Language-Based Audio Retrieval Task in {DCASE} 2022 Challenge},
    booktitle = {Proceedings of the 7th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE)},
    year = {2022},
    pages = {216-220}
}
```

This repository includes:

  1. source code of the baseline system,
  2. training/evaluation script(s) for the baseline system,
  3. source code of evaluation metrics,
  4. pre-processing/feature extraction scripts for the Clotho dataset.


## Baseline system description

The baseline system is a simpler version of the **audio-text aligning framework** presented in [this paper](http://arxiv.org/abs/2110.02939), which calculates relevant scores between encoded textual descriptions and encoded audio signals.
A **convolutional recurrent neural network** (CRNN) is used as the audio encoder, which extracts frame-wise acoustic embeddings from audio signals.
Then, an audio signal is represented by the average of its frame-wise acoustic embeddings.
A **pre-trained Word2Vec** (published [here](https://code.google.com/archive/p/word2vec/) by Google) is used as the text encoder, which converts textual descriptions into sequences of word embeddings.
A textual description is represented by the average of its word embeddings.
The relevant score between an audio signal and a textual description is calculated by the dot product of their vector representations.
The baseline system is optimized with a **triplet ranking loss** criterion.

For more details on the **audio-text aligning framework**, please see:

_H. Xie, O. Räsänen, K. Drossos, and T. Virtanen, “Unsupervised Audio-Caption Aligning Learns Correspondences between Individual Sound Events and Textual Phrases,” Accepted at IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2022. Available: http://arxiv.org/abs/2110.02939._

For details on the **CRNN audio encoder**, please see:

_X. Xu, H. Dinkel, M. Wu, and K. Yu, “A CRNN-GRU Based Reinforcement Learning Approach to Audio Captioning,” in Proceedings of the Detection and Classification of Acoustic Scenes and Events Workshop (DCASE), 2020, pp. 225–229._

For the **triplet ranking loss** used to train the baseline system, please see:

_J. Bromley, I. Guyon, Y. LeCun, E. Säckinger, and R. Shah, “Signature Verification Using a ‘Siamese’ Time Delay Neural Network,” in Proceedings of the 6th International Conference on Neural Information Processing Systems, 1993, pp. 737–744._


## Code Tutorial

This repository is developed with Python 3.8.

1. Checkout the code and install the required python packages.

```
> git clone https://github.com/xieh97/dcase2022-audio-retrieval.git
> pip install -r requirements.txt
```

2. Download and extract audio files and captions of the **Clotho v2.1** dataset from Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4783391.svg)](https://doi.org/10.5281/zenodo.4783391). Below is the expanded directory layout:

```
Clotho.v2.1
├─clotho_captions_development.csv
├─clotho_captions_validation.csv
├─clotho_captions_evaluation.csv
├─development
│  └─...(3839 wavs)
├─validation
│  └─...(1045 wavs)
└─evaluation
   └─...(1045 wavs)
```

3. Preprocess audio and caption data.

```
> python3 scripts/preprocess.py
```

This step will output four files:

  * `audio_info.pkl`: audio file names and durations.
  * `development_captions.json`: preprocessed captions of Clotho development split.
  * `validation_captions.json`: preprocessed captions of Clotho validation split.
  * `evaluation_captions.json`: preprocessed captions of Clotho evaluation split.
  * `vocab_info.pkl`: vocabulary statistics of Clotho dataset.

4. Extract audio features (**log-mel energies**).

```
> python3 scripts/audio_features.py
```

The final log-mel energies of each Clotho audio file will be saved into `audio_logmel.hdf5`.

5. Generate word embeddings.

```
> python3 scripts/word_embeddings.py
```

Each word in Clotho captions will be encoded with [pretrained Word2Vec](https://code.google.com/archive/p/word2vec/) word embeddings.
This step will output word embedding file `word2vec_emb.pkl`.

6. Edit `conf.yaml`.

```
experiment: audio-retrieval

output_path: "?"  # Your output data dir

# Ray-tune configurations
ray_conf:
    ...

# Data configurations
train_data:
    input_path: "?"  # Your input data dir
    ...

# Training configurations
training:
    ...

# Model hyper-parameters
CRNNWordModel:
    ...

# Algorithm hyper-parameters

# Losses
TripletRankingLoss:
    ...

# Optimizers
AdamOptimizer:
    ...

# Evaluation data
eval_data:
    input_path: "?"  # Your evaluation data dir
    ...
```

**Note:** The baseline system is trained with the tool [Ray Tune](https://www.ray.io/ray-tune).

7. Run `main.py` to train/validate/evaluate the baseline system with Clotho dataset.

```
> python3 main.py
```

**Note:** The `test.output.csv` files contains file names of the top-10 retrieved audio files for each caption in the Clotho v2 evaluation data.

8. Calculate evaluation metrics (i.e., **recall@{1, 5, 10}**, **mAP@10**) with `eval_script.py`.

```
> python3 eval_script.py
```


## Results for the Clotho evaluation split

The results of the baseline system for the Clotho v2 evaluation split are:

| Metric                  | Value                 |
|-------------------------|-----------------------|
| recall@1                | 0.03                  |
| recall@5                | 0.11                  |
| recall@10               | 0.19                  |
| <strong>mAP@10</strong> | <strong>0.07</strong> |
