
"""
# from nltk.corpus import reuters
from nltk import bigrams, trigrams
from collections import Counter, defaultdict
# Create a placeholder for model
model = Sequential()
# Count frequency of co-occurrence
for sentence in data:
    for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
        model[(w1, w2)][w3] += 1
 # Let's transform the counts to probabilities
for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count
"""

import numpy as np
import pandas as pd
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint


def get_words(sentence):
    """
    Get a list of all words which are found in a given sentence.

    Parameters
    ----------
    sentence : str
        The sentence to find words in.

    Returns
    -------
    list
        A list containing all words found in the sentence.
    """
    return sentence.split()


def calculate_distance(word1, word2):
    """
    Calculate the distance between two words. This method could be replaced by a method calculating the Levenshtein
    distance.

    Parameters
    ----------
    word1 : str
        First word.
    word2 : str
        Second word.

    Returns
    -------
    float
        Distance between word1 and word2
    """
    return len(word1.replace(word2, ''))


training_sentences = ["cookies", "chocolate", "chip", "chocolate chip cookie", "chocolate chip cookies", "mister", "gospel", "quilter", "apostle", "middle", "classes", "glad",
                      "welcome", "festive", "season", "year", "with", "christmas", "roast", "beef", "looming", "before", "similes", "drawn", "from", "eating", "results", "occur", "readily"]
