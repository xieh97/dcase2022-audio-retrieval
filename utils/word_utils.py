import re
import string

import contractions
import gensim.downloader as downloader
import nltk
from nltk.corpus import wordnet

stopwords = nltk.corpus.stopwords.words("english")

UNK_token = "<UNK>"

wordnet_pos_tag = {
    "J": wordnet.ADJ,
    "N": wordnet.NOUN,
    "V": wordnet.VERB,
    "R": wordnet.ADV
}

embedding_model = None
word2vec = "word2vec-google-news-300"


def word_vector(word):
    global embedding_model

    if embedding_model is None:
        embedding_model = downloader.load(word2vec)

    try:
        return embedding_model.get_vector(word)
    except KeyError:
        return None


def clean_text(text):
    words = []

    # Remove redundant spaces
    text = re.sub(r"\s+", " ", text)

    # Expand contractions (n't, 's, 've, 'm, etc.)
    text = " ".join([contractions.fix(t) for t in re.split(r"\s", text)])

    # Replace dashes with spaces - task-specific operation
    text = re.sub(r"-+", " ", text)

    # Remove punctuations
    # text = "".join([i for i in text if i not in string.punctuation])
    # text = re.sub(r"[^0-9A-Za-z\s]", "", text)

    # Normalize case
    # text = text.lower()
    # text = text.upper()

    # Tokenization
    tokens = nltk.word_tokenize(text)  # an apostrophe is not considered as punctuation.
    # tokens = nltk.TweetTokenizer().tokenize(text)  # for text data from social media consisting of #, @, emoticons.
    # tokens = re.split(r"\s", text)  # split by spaces

    # Remove stopwords
    # tokens = [t for t in tokens if t not in stopwords]

    # POS Tagging
    tokens_tag = nltk.pos_tag(tokens)

    # Recognize named entity
    chunked_tree = nltk.ne_chunk(tokens_tag, binary=True)
    tokens_tag = nltk.tree2conlltags(chunked_tree)

    # Introduce UNK_token
    for token, tag, IOB_tag in tokens_tag:
        if IOB_tag != "O":  # named entity
            words.append(UNK_token)
            print(token, "-->", UNK_token)

        elif tag in ["POS"]:  # possessive endings ('s, ', etc.)
            print(token, "-->")
            pass

        elif token in string.punctuation:  # punctuations
            print(token, "-->")
            pass

        elif word_vector(token) is not None:  # has pretrained embeddings
            words.append(token)
            print(token, "-->", token)

        else:
            # Lemmatization
            lemmatizer = nltk.WordNetLemmatizer()
            lemma = lemmatizer.lemmatize(token, pos=wordnet_pos_tag.get(tag[0], wordnet.NOUN))

            if word_vector(lemma) is not None:  # has pretrained embeddings
                words.append(lemma)
                print(token, "-->", lemma)
            else:
                # Stemming
                # stemmer = nltk.PorterStemmer()
                stemmer = nltk.SnowballStemmer(language="english")
                root = stemmer.stem(lemma)

                if word_vector(root) is not None:  # has pretrained embeddings
                    words.append(root)
                    print(token, "-->", root)
                else:
                    words.append(UNK_token)
                    print(token, "-->", UNK_token)

    return text, words
