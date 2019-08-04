import time
import numpy as np

startTime = 0


def tic():
    global startTime
    startTime = time.time()


def toc():
    global startTime
    t = time.time() - startTime
    if t < 61:
        print('{:2.2f} seconds'.format(t))
    elif t < 60 * 60 + 1:
        print('{minuts} min and {sec} seconds'.format(minuts=int(t / 60), sec=int(t % 60)))
    else:
        print('{hours} hours and {minuts} minuts'.format(hours=int(t / 3600), minuts=int((t % 3600) / 60)))

    return t


def shuffle(data, labels):
    indices = np.arange(labels.shape[0])
    np.random.shuffle(indices)

    return data[indices], labels[indices]


def ReLU(x):
    return x * (x > 0)


def dReLU(x):
    return 1. * (x > 0)


def softmax(x):
    """
    should be satble softmax
    :param x:
    :return: softmax on 1D ndarray
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def cross_entropy_loss(x):
    """
    in classification: sum of y_t*log(yp), where t is target and p id prediction,
    y_t: 1-hot-vector i.e: [0,0,1,0,0]  so: at the end the loss is: 1 * log(y_p), where y_p index is the index
    such that i := np.where(y_t == 1)
    :param x: value of y_p
    :return: cross entropy loss
    """
    return -1 * np.log(x)


def L2_cost(W_array, lambd):
    cost = 0.0

    for W in W_array:
        cost += np.sum(0.5 * lambd * np.square(W))

    return cost


def dL2(W_array, lambd):
    dl2 = []
    for W in W_array:
        dl2.append(lambd * W)

    return dl2
