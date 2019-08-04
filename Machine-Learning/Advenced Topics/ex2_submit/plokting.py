import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def plot_acc():
    acc_1 = pickle.load(open('model1_acc.p', 'rb'))
    acc_2 = pickle.load(open('model2_acc.p', 'rb'))
    acc_3 = pickle.load(open('model3_acc.p', 'rb'))
    plt.clf()
    x_range = list(range(1, len(acc_1) + 1))
    plt.plot(x_range, acc_1, label='model 1')
    plt.plot(x_range, acc_2, label='model 2')
    plt.plot(x_range, acc_3, label='model 3')

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode="expand", borderaxespad=0.)

    plt.xlabel('Epochs')
    plt.ylabel('test accuracy [%]')
    # plt.title('models accuracy')
    plt.savefig("models_accuracy.png")

    plt.clf()


def plot_bigram():
    W_pred = pickle.load(open('model3_W_pred.p', 'rb'))
    W_bi = W_pred[1:27, 129:155]

    letters = list(LETTERS)

    fig, ax = plt.subplots()
    im = ax.imshow(W_bi, cmap='binary')

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(letters)))
    ax.set_yticks(np.arange(len(letters)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(letters)
    ax.set_yticklabels(letters)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    ax.set_title("hitmap of bigrams")
    fig.tight_layout()
    plt.savefig("bigrams.png")
    plt.clf()


if __name__ == '__main__':
    plot_acc()
    plot_bigram()
    print('End test')
