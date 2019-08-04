import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle

#################
#   globals     #
#################

# read data
mnist = fetch_mldata("MNIST original", data_home="./data")

X, Y = mnist.data[:60000] / 255., mnist.target[:60000]
train_data = [ex for ex, ey in zip(X, Y) if ey in [0, 1, 2, 3]]
train_label = [ey for ey in Y if ey in [0, 1, 2, 3]]
x_test = np.loadtxt("./data/x_test.txt")
y_test = np.loadtxt("./data/y_test.txt")
x4pred = np.loadtxt("./data/x4pred.txt")

EPOCHS = 50  # number of epochs
eta_0 = 0.1  # learning rate (at t = 0)
lambada = 0.1  # regularization

# all pairs matrix classifiers
AP = np.array([[1, 1, 1, 0, 0, 0], [-1, 0, 0, 1, 1, 0],
               [0, -1, 0, -1, 0, 1], [0, 0, -1, 0, -1, -1]], dtype=float)
# one VS all matrix (diagonal is 1, other -1)
OVA = np.array([[1, -1, -1, -1], [-1, 1, -1, -1],
                [-1, -1, 1, -1], [-1, -1, -1, 1]], dtype=float)
# random 16 classifiers matrix
RAND = np.around(2 * np.random.rand(4, 16) - 1)

# W_i is matrix if all together classifiers such that:
# W1 is classifier of One Vs All
# W2 is classifier of All Pairs
# W3 is classifier of random classifiers
W1 = np.zeros((784, 4))
W2 = np.zeros((784, 6))
W3 = np.zeros((784, 16))

ova_hmm_good_pred_count = 0.0
ap_hmm_good_pred_count = 0.0
random_hmm_good_pred_count = 0.0
ova_loss_good_pred_count = 0.0
ap_loss_good_pred_count = 0.0
random_loss_good_pred_count = 0.0


def train_ova_classifier(classifier_number, data, label, lr):
    global W1, OVA

    # TODO: write as if 1-W[label][0]*np.dot(W_0,data)
    y_pred = np.dot(W1[:, classifier_number], data)
    y_i = OVA[label, classifier_number]
    if 1 - y_pred * y_i >= 0:
        W1[:, classifier_number] = (1 - lambada * lr) * W1[:, classifier_number] + lr * y_i * data
    else:
        W1[:, classifier_number] = (1 - lambada * lr) * W1[:, classifier_number]


def train_ap_classifier(classifier_number, data, label, lr):
    global W2, AP

    y_i = AP[label, classifier_number]
    if y_i != 0:
        # TODO: write as if 1-W[label][0]*np.dot(W_0,data)
        y_pred = np.dot(W2[:, classifier_number], data)
        if 1 - y_pred * y_i >= 0:
            W2[:, classifier_number] = (1 - lambada * lr) * W2[:, classifier_number] + lr * y_i * data
        else:
            W2[:, classifier_number] = (1 - lambada * lr) * W2[:, classifier_number]


def train_rand_classifier(classifier_number, data, label, lr):
    global W3, RAND

    y_i = RAND[label, classifier_number]
    if y_i != 0:
        # TODO: write as if 1-W[label][0]*np.dot(W_0,data)
        y_pred = np.dot(W3[:, classifier_number], data)
        if 1 - y_pred * y_i >= 0:
            W3[:, classifier_number] = (1 - lambada * lr) * W3[:, classifier_number] + lr * y_i * data
        else:
            W3[:, classifier_number] = (1 - lambada * lr) * W3[:, classifier_number]


def test_ova(data, label):
    global OVA, ova_loss_good_pred_count, ova_hmm_good_pred_count
    # f(i) is the prediction of classifier i.
    f = np.array([np.dot(W1[:, 0], data), np.dot(W1[:, 1], data), np.dot(W1[:, 2], data), np.dot(W1[:, 3], data)])
    hamm_dist = np.sum(1 - (np.sign(np.multiply(f, OVA))) / 2, axis=1)
    hamm_pred = np.argmin(hamm_dist)
    if hamm_pred == label:
        ova_hmm_good_pred_count += 1

    loss_dist = np.sum(np.maximum(0, 1 - np.multiply(f, OVA)), axis=1)
    loss_pred = np.argmin(loss_dist)
    if loss_pred == label:
        ova_loss_good_pred_count += 1


def test_ap(data, label):
    global AP, ap_hmm_good_pred_count, ap_loss_good_pred_count

    # f(i) is the prediction of classifier i.
    f = np.array([np.dot(W2[:, 0], data), np.dot(W2[:, 1], data), np.dot(W2[:, 2], data),
                  np.dot(W2[:, 3], data), np.dot(W2[:, 4], data), np.dot(W2[:, 5], data)])

    hamm_dist = np.sum(1 - (np.sign(np.multiply(f, AP))) / 2, axis=1)
    hamm_pred = np.argmin(hamm_dist)
    if hamm_pred == label:
        ap_hmm_good_pred_count += 1

    loss_dist = np.sum(np.maximum(0, 1 - np.multiply(f, AP)), axis=1)
    loss_pred = np.argmin(loss_dist)
    if loss_pred == label:
        ap_loss_good_pred_count += 1


def test_random(data, label):
    global RAND, random_hmm_good_pred_count, random_loss_good_pred_count

    # f(i) is the prediction of classifier i.
    f = np.array([np.dot(W3[:, 0], data), np.dot(W3[:, 1], data), np.dot(W3[:, 2], data),
                  np.dot(W3[:, 3], data), np.dot(W3[:, 4], data), np.dot(W3[:, 5], data),
                  np.dot(W3[:, 6], data), np.dot(W3[:, 7], data), np.dot(W3[:, 8], data), np.dot(W3[:, 9], data),
                  np.dot(W3[:, 10], data),
                  np.dot(W3[:, 11], data), np.dot(W3[:, 12], data), np.dot(W3[:, 13], data),
                  np.dot(W3[:, 14], data), np.dot(W3[:, 15], data)])
    hamm_dist = np.sum(1 - (np.sign(np.multiply(f, RAND))) / 2, axis=1)
    hamm_pred = np.argmin(hamm_dist)
    if hamm_pred == label:
        random_hmm_good_pred_count += 1

    loss_dist = np.sum(np.maximum(0, 1 - np.multiply(f, RAND)), axis=1)
    loss_pred = np.argmin(loss_dist)
    if loss_pred == label:
        random_loss_good_pred_count += 1


def train_ova(data, label, lr_t):
    """
    in ova there is 4 classifiers - train all of them.
    :param data: input data
    :param label: correct label
    :param lr_t: learning rate at epoch t
    """
    for i in range(4):
        # train classifier # i:
        train_ova_classifier(int(i), data, int(label), lr_t)


def train_ap(data, label, lr_t):
    """
    in All pairs there is 6 classifiers - train all of them.
    :param data: input data
    :param label: correct label
    :param lr_t: learning rate at epoch t
    """
    for i in range(6):
        # train classifier # i:
        train_ap_classifier(int(i), data, int(label), lr_t)


def train_random(data, label, lr_t):
    """
    in random there is 16 classifiers - train all of them.
    :param data: input data
    :param label: correct label
    :param lr_t: learning rate at epoch t
    """
    for i in range(16):
        # train classifier # i:
        train_rand_classifier(int(i), data, int(label), lr_t)


def train():
    global EPOCHS, eta_0
    global train_data, train_label

    for epoch in range(EPOCHS):
        # learning rate
        lr_t = eta_0 / np.sqrt(epoch + 1)
        x, y = shuffle(train_data, train_label)
        for data, label in zip(x, y):
            train_ova(data, label, lr_t)
            train_ap(data, label, lr_t)
            train_random(data, label, lr_t)

        print("############\tepoch: %d\t############" % (epoch + 1))
        test()


def test():
    global x_test, y_test
    global ova_hmm_good_pred_count, ova_loss_good_pred_count, ap_loss_good_pred_count, ap_hmm_good_pred_count
    global random_hmm_good_pred_count, random_loss_good_pred_count
    test_size = float(len(x_test))
    ova_hmm_good_pred_count = 0.0
    ap_hmm_good_pred_count = 0.0
    random_hmm_good_pred_count = 0.0
    ova_loss_good_pred_count = 0.0
    ap_loss_good_pred_count = 0.0
    random_loss_good_pred_count = 0.0

    for data, label in zip(x_test, y_test):
        test_ova(data, label)
        test_ap(data, label)
        test_random(data, label)
    print("\n##OVA##")
    print("hamming accuracy is: %.2f %%" % (ova_hmm_good_pred_count * 100 / test_size))
    print("loss accuracy is: %.2f %%" % (ova_loss_good_pred_count * 100 / test_size))
    print("\n##AP##")
    print("hamming accuracy is: %.2f %%" % (ap_hmm_good_pred_count * 100 / test_size))
    print("loss accuracy is: %.2f %%" % (ap_loss_good_pred_count * 100 / test_size))
    print("\n##random##")
    print("hamming accuracy is: %.2f %%" % (random_hmm_good_pred_count * 100 / test_size))
    print("loss accuracy is: %.2f %%" % (random_loss_good_pred_count * 100 / test_size))


def write_to_file():
    global OVA, AP, RAND
    global W1, W2, W3
    file1 = open("test.onevall.ham.pred", 'w')
    file2 = open("test.allpairs.ham.pred", 'w')
    file3 = open("test.randm.ham.pred", 'w')
    file4 = open("test.onevall.loss.pred", 'w')
    file5 = open("test.allpairs.loss.pred", 'w')
    file6 = open("test.randm.loss.pred", 'w')

    for data in x4pred:
        f_ova = np.array(
            [np.dot(W1[:, 0], data), np.dot(W1[:, 1], data), np.dot(W1[:, 2], data), np.dot(W1[:, 3], data)])
        f_ap = np.array([np.dot(W2[:, 0], data), np.dot(W2[:, 1], data), np.dot(W2[:, 2], data),
                         np.dot(W2[:, 3], data), np.dot(W2[:, 4], data), np.dot(W2[:, 5], data)])
        f_rand = np.array([np.dot(W3[:, 0], data), np.dot(W3[:, 1], data), np.dot(W3[:, 2], data),
                           np.dot(W3[:, 3], data), np.dot(W3[:, 4], data), np.dot(W3[:, 5], data),
                           np.dot(W3[:, 6], data), np.dot(W3[:, 7], data), np.dot(W3[:, 8], data),
                           np.dot(W3[:, 9], data), np.dot(W3[:, 10], data), np.dot(W3[:, 11], data),
                           np.dot(W3[:, 12], data), np.dot(W3[:, 13], data), np.dot(W3[:, 14], data),
                           np.dot(W3[:, 15], data)])

        hamm_dist = np.sum(1 - (np.sign(np.multiply(f_ova, OVA))) / 2, axis=1)
        ova_hamm_pred = np.argmin(hamm_dist)
        file1.write(str(ova_hamm_pred) + "\n")

        hamm_dist = np.sum(1 - (np.sign(np.multiply(f_ap, AP))) / 2, axis=1)
        ap_hamm_pred = np.argmin(hamm_dist)
        file2.write(str(ap_hamm_pred) + "\n")

        hamm_dist = np.sum(1 - (np.sign(np.multiply(f_rand, RAND))) / 2, axis=1)
        rand_hamm_pred = np.argmin(hamm_dist)
        file3.write(str(rand_hamm_pred) + "\n")

        loss_dist = np.sum(np.maximum(0, 1 - np.multiply(f_ova, OVA)), axis=1)
        ova_loss_pred = np.argmin(loss_dist)
        file4.write(str(ova_loss_pred) + "\n")

        loss_dist = np.sum(np.maximum(0, 1 - np.multiply(f_ap, AP)), axis=1)
        ap_loss_pred = np.argmin(loss_dist)
        file5.write(str(ap_loss_pred) + "\n")

        loss_dist = np.sum(np.maximum(0, 1 - np.multiply(f_rand, RAND)), axis=1)
        rand_loss_pred = np.argmin(loss_dist)
        file6.write(str(rand_loss_pred) + "\n")

    file1.close()
    file2.close()
    file3.close()
    file4.close()
    file5.close()
    file6.close()


if __name__ == '__main__':
    train()
    write_to_file()
