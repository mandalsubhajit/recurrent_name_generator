# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 16:52:07 2017

@author: subhajit
"""

import numpy as np
import os

# read names from dataset
def get_data():
    data = set()
    for filename in ['male.txt', 'female.txt']:
        for line in open(os.path.join(os.path.dirname(__file__),filename)):
            if len(line.strip()) and not line.startswith('#'):
                data.add(line.strip().lower())
    return data

# get the char to index mapping (both ways) for creating one-hot vectors
def get_character_mapping(data):
    chars = sorted(list(set(''.join(data))))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    return char_indices, indices_char


# convert names to one-hot vectors (1 or -1)
def vectorize(string, maxlen, char_indices):
    vec = -1*np.ones((1, maxlen, len(char_indices)+1), dtype=np.bool)
    # PADDING
    vec[0, :, len(char_indices)] = 1
    for t, char in enumerate(string[:maxlen]):
        vec[0, t, len(char_indices)] = -1
        vec[0, t, char_indices[char]] = 1
    return vec

# convert predicted output to names
def characterize(vec, maxlen, indices_char):
    string = ''
    vec = vec.reshape(maxlen, len(indices_char)+1)
    for t in range(vec.shape[0]):
        best = np.argmax(vec[t])
        if best == len(indices_char):
            break
        string = string + indices_char[best]
    return string