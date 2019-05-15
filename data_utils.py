import os
import json
import numpy as np
import random
from keras.utils import Progbar
import h5py


with open('en_tokens.json', 'r') as f:
    en_tokens = json.load(f)

with open('mal_tokens.json', 'r') as f:
    mal_tokens = json.load(f)

with open('en_maxlen.txt', 'r') as f:
    en_maxlen = int(f.read())

with open('mal_maxlen.txt', 'r') as f:
    mal_maxlen = int(f.read())


START_TOKEN = '\t'
STOP_TOKEN = '\n'

mal_tokens.append(ord(START_TOKEN))
mal_tokens.append(ord(STOP_TOKEN))
mal_maxlen += 2

_en_token2idx = {en_tokens[i]: i + 1 for i in range(len(en_tokens))}
_mal_token2idx = {mal_tokens[i]: i + 1 for i in range(len(mal_tokens))}

def en_idx2token(i):
    return en_tokens[i - 1]

def mal_idx2token(i):
    return mal_tokens[i - 1]

def en_token2idx(c):
    return _en_token2idx[c]

def mal_token2idx(c):
    return _mal_token2idx[ord(c)]


def vectorize_en_word(word):
    return [en_token2idx(c) for c in word]


def vectorize_mal_word(word):
    return [mal_token2idx(c) for c in word]


def get_data(file_name='malayalam_map_unique.csv', min_frequency=1000, test_split=0.1):
    X = []
    Y = []
    print('Loading data...')
    lines = []
    with  open(file_name, 'r', encoding='utf8') as f:
        for line in f:
            freq = int(f.readline().split(',')[3][1:-2])
            if freq >= min_frequency:
                lines.append(line)
    lines.pop(0)
    print('Done.')
    print('Processing...')
    pbar = Progbar(len(lines))
    for line in lines:
        sp = line.split(',')
        word_en = sp[1][1:-1]
        word_mal = ','.join(sp[2: -1])[1:-1]
        word_mal = '\t' + word_mal + '\n'
        X.append(vectorize_en_word(word_en))
        Y.append(vectorize_mal_word(word_mal))
        pbar.add(1)
    idxs = list(range(len(X)))
    random.shuffle(idxs)
    X = [X[i] for i in idxs]
    Y = [Y[i] for i in idxs]
    print('Done.')

    if test_split is None:
        return X, Y

    return _train_test_split(X, Y, test_split)


def _to_s2s(X, Y):
    encoder_inputs = X
    decoder_inputs = Y
    decoder_targets = []
    for y in decoder_inputs:
        y = y[:]
        for j in range(1, len(y)):
            y[j - 1] = y[j]
        y.pop(-1)
        decoder_targets.append(y)
    return encoder_inputs, decoder_inputs, decoder_targets


def _to_one_hot(X, num_classes):
    Y = []
    for x in X:
        y = []
        for i in x:
            oh = [0] * num_classes
            oh[i] = 1
            y.append(oh)
        Y.append(y)
    return Y


def _train_test_split(X, Y, test_split):
    test_count = int(len(X) * test_split)
    X_train = X[:-test_count]
    Y_train = Y[:-test_count]
    X_test = X[-test_count:]
    Y_test = Y[-test_count:]
    return (X_train, Y_train), (X_test, Y_test)


def _s2s_to_one_hot(encoder_inputs, decoder_inputs, decoder_targets):
    encoder_inputs = _to_one_hot(encoder_inputs, len(en_tokens) + 1)
    decoder_inputs = _to_one_hot(decoder_inputs, len(mal_tokens) + 1)
    decoder_targets =_to_one_hot(decoder_targets, len(mal_tokens) + 1)
    return encoder_inputs, decoder_inputs, decoder_targets


def _pad(X, maxlen, pad_left=True):
    dim = len(X[0][0])
    zeros = [0] * dim
    for i, x in enumerate(X):
        if pad_left:
            x += [zeros] * (maxlen - len(x))
        else:
            X[i] = x + [zeros] * (maxlen - len(x))


def get_seq2seq_data(data):
    s2s_data = _s2s_to_one_hot(*_to_s2s(*data))
    _pad(s2s_data[0], en_maxlen, True)
    _pad(s2s_data[1], mal_maxlen, True)
    _pad(s2s_data[2], mal_maxlen, False)
    s2s_data = tuple(map(np.array, s2s_data))
    return s2s_data


def seq2seq_data_generator(data, batch_size):
    idx = 0
    while True:
        batch = data[0][idx: idx + batch_size], data[1][idx: idx + batch_size]
        if len(batch[0]) < batch_size:
            reminder = data[0][: batch_size - len(batch)], data[1][: batch_size - len(batch)]
            batch = np.concatenate([batch[0], reminder[0]]), np.concatenate([batch[1], reminder[1]])
            idx = len(reminder[0])
        idx += batch_size
        if idx == len(data[0]):
            idx = 0
        s2s_data = get_seq2seq_data(batch)
        yield list(s2s_data[:2]), s2s_data[2]