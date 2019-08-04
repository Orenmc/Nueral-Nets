import numpy as np
import pickle

# GLOBALS:
LETTERS = 'abcdefghijklmnopqrstuvwxyz'  # english alphabet
CLASSES = 26
INPUT = 128
EPOCHS = 3
W_sum = W = np.random.rand(CLASSES, INPUT) - 0.5  # normalized
C2I = {c: i for i, c in enumerate(LETTERS)}  # letter to index
I2C = {i: c for c, i in C2I.items()}  # index to letter


def phi(x_i, y_i):
    t1 = np.zeros((CLASSES, INPUT))
    t1[y_i] = x_i
    return t1


def split_sample(line):
    """
    split line to input (128x1) and label (index of letter 0-25)
    :param line: line to split has more than only those fields..
    :return: x - data (OCR 128x1), label - 0-25 (based on index of letter)
    """
    line = line.strip()
    letter_id, letter, next_id, word_id, possition, fold, x = line.split('\t', 6)
    x = np.fromstring(x, dtype=int, sep='\t')
    # return letter_id, letter, next_id, fold, x
    label = C2I[letter]
    return x, label


def read_files():
    """
    read files
    :return:
    """
    with open("data/letters.train.data", 'r') as f:
        train_lines = f.readlines()
    with open("data/letters.test.data", 'r') as f:
        test_lines = f.readlines()
    return train_lines, test_lines


def create_sets():
    """
    creates train file and test file - split them to data and label, and store them in the set. (as tuple)
    :return: train set and test set. [(x1,l1), (x2,l2),... (xn, yn)]
    """
    train_labeled = []
    test_labeled = []
    train_lines, test_lines = read_files()
    for line in train_lines:
        data, label = split_sample(line)
        train_labeled.append((data, label))

    for line in test_lines:
        data, label = split_sample(line)
        test_labeled.append((data, label))

    return train_labeled, test_labeled


def prediction(data, weights):
    pred = []
    for i in range(CLASSES):
        p_i = phi(data, i)
        pred_i = np.inner(weights, p_i)
        t = pred_i.trace()
        pred.append(t)
    return np.array(pred)


if __name__ == '__main__':
    print('#' * 30 + "\tSTART model 2\t" + '#' * 30)
    accuracy = []
    train_set, test_set = create_sets()

    # TRAIN:

    np.random.shuffle(train_set)
    counter = 0
    for epoch in range(EPOCHS):
        # train:

        t_good = 0.0
        t_bad = 0.0

        for data, label in train_set:
            pred = prediction(data, W)
            y_hat = np.argmax(pred)
            if y_hat != label:
                counter += 1
                t_bad += 1
                t1 = phi(data, label)
                t2 = phi(data, y_hat)
                W = W + t1 - t2
                W_sum = W_sum + W  # W1 + w2 + w3 ... wt

            else:
                t_good += 1

        # end train in this epoch
        """
        print('epoch #{:d}:\ttrain acc = {:d}/{:d}, {:.2f}'.format(epoch + 1, int(t_good), int(t_good + t_bad),
                                                                   t_good * 100 / (t_bad + t_good)))
        """
        # TEST :
        if counter != 0:
            W_pred = W_sum / counter
        else:
            W_pred = W_sum
        good = bad = 0.0
        for d, l in test_set:
            p1 = prediction(d, W_pred)
            y1 = np.argmax(p1)
            # print('pred : {}, label: {}'.format(I2C[y1], I2C[l]))
            if y1 == l:
                good += 1
            else:
                bad += 1
        print('epoch #{:d}:\ttest acc = {:d}/{:d}, {:.2f}'.format(epoch + 1, int(good), int(good + bad),
                                                                  good * 100 / (bad + good)))
        acc = good * 100 / (bad + good)
        accuracy.append(acc)
    # end of Epochs
    # print(accuracy)
    pickle.dump(accuracy, open("model2_acc.p", 'wb'))
    pickle.dump(W, open("model2_W.p", 'wb'))
    pickle.dump(W_pred, open("model2_W_pred.p", 'wb'))

    print('#' * 30 + "\tEND model 2\t" + '#' * 30)
