
import numpy as np
from numpy import linalg as la
import utils2 as utils

STUDENT = {'name': "Oren Cohen",
           'ID': "305164295"}


def most_similar(word, k):
    word_embedding_dict = utils.EMBEDDING_WORD
    u = word_embedding_dict[word]
    words_distances = []
    for w in word_embedding_dict:
        v = word_embedding_dict[w]
        dist = cosine_distance(u, v)
        words_distances.append([w, dist])

    words_distances.sort(key=distance, reverse=True)

    top_k = []
    for i in range(k):
        top_k.append(words_distances[i+1])

    return top_k


def cosine_distance(u, v):
    n = np.dot(u, v)
    d = la.norm(u) * la.norm(v)
    return n / d


def distance(word_and_distance):

    return word_and_distance[1]


def get_word_embedding():
    dictionary = {}

    word_vectors = np.loadtxt("data/wordVectors.txt")
    with open("data//vocab.txt", 'r') as f:
        lines = f.readlines()

    for word, vec in zip(lines, word_vectors):
        word = word.strip().strip('\n')
        dictionary[word] = vec

    return dictionary


if __name__ == "__main__":
    utils.set_word_embedding()
    word_list = ['dog', 'england', 'john', 'explode', 'office']

    for word in word_list:
        print("top 5 most similar to " + word + " are:")
        print(str(most_similar(word, 5)) + "\n")
