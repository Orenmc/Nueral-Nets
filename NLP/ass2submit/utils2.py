import numpy as np


EMBEDDING_WORD = {}
WORD_VECTOR = np.loadtxt("data/wordVectors.txt")


def set_word_embedding():
    global EMBEDDING_WORD, WORD_VECTOR
    dictionary = {}

    word_vectors = WORD_VECTOR
    with open("data/vocab.txt", 'r') as f:
        lines = f.readlines()

    for word, vec in zip(lines, word_vectors):
        word = word.strip().strip('\n')
        dictionary[word] = vec

    EMBEDDING_WORD = dictionary

