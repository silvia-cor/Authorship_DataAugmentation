from collections import Counter

import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
import re


# convert n-class labels to binary
def labels_for_AV(labels, author):
    return [1 if label == author else 0 for label in labels]


# convert text (string of keys) -> encoding (list of values)
def text_encod(text, vocab):
    return [vocab.get(item, vocab.get('<unk>')) for item in word_tokenize(text)]


# convert encoding (list of values) -> text (string of keys)
def encod_text(encod, vocab):
    return ' '.join([str(list(vocab)[item]) for item in encod])


# convert encoding (list of values) -> one-hot (tensor (S, E))
def encod_onehot(encod, vocab):
    if not torch.is_tensor(encod):
        encod = torch.Tensor(encod).long()
    return F.one_hot(encod, num_classes=len(vocab))


# convert one-hot (tensor) -> encoding (list of values)
def onehot_encod(one_hot):
    return torch.argmax(one_hot, dim=1)


# convert one-hot (tensor) -> text
def onehot_text(onehot, vocab):
    encod = onehot_encod(onehot)
    text = encod_text(encod, vocab)
    return delete_gpt2tokenizer_unwanted_symbols(text)


def get_maxlen(docs):
    return len(max(docs, key=len))


def get_splits(text):
    words = word_tokenize(text)  # with punctuation
    splits = [' '.join(words[i:i + 100]) for i in range(0, len(words), 100)]
    splits_purged = [split for split in splits if len(tokenize_nopunct(split)) >= 25]
    return splits_purged


def tokenize_nopunct(text):
    unmod_tokens = word_tokenize(text)
    return [token.lower() for token in unmod_tokens if
            any(char.isalpha() for char in token)]  # checks whether all the chars are alphabetic


# delete authors with less than 10 samples
def purge_authors(texts, labels, to_keep=None):
    c = Counter(labels)
    if not to_keep:
        to_keep = [key for key in c.keys() if c[key] >= 10]
    new_texts = [text for i, text in enumerate(texts) if labels[i] in to_keep]
    new_labels = [label for label in labels if label in to_keep]
    return new_texts, new_labels, to_keep


# very light polishing of GPT2 output
def delete_gpt2tokenizer_unwanted_symbols(text):
    symbol_todelete = "Ġ"
    text = re.sub(symbol_todelete, " ", text)
    text = re.sub(r"\s+", " ", text)
    return text
