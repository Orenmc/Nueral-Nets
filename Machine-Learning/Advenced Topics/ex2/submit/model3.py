import numpy as np
import pickle

# GLOBALS:
ENG_ALPHABET = '$abcdefghijklmnopqrstuvwxyz'  # english alphabet
# LABELS = 'abcdefghijklmnopqrstuvwxyz'  # class alphabet
# NUM_CLASSES = 26
NUM_ENG_CHR = 27

INPUT = 128
EPOCHS = 3

W_sum = W = np.random.rand(NUM_ENG_CHR, INPUT + NUM_ENG_CHR) - 0.5

ENG2I = {c: i for i, c in enumerate(ENG_ALPHABET)}  # eng_alphabet to index
I2ENG = {i: c for c, i in ENG2I.items()}  # index to eng_alphabet
"""
L2I = {c: i for i, c in enumerate(LABELS)}  # letter to index
I2L = {i: c for c, i in L2I.items()}  # index to letter
"""


def build_phi(x_i, y_hat):
    y_prev, y_curr = y_hat  # y_curr is the row, y_prev is the col.

    phi = np.zeros((NUM_ENG_CHR, INPUT + NUM_ENG_CHR))  # 27 x 155
    np.put(phi[y_curr], range(INPUT), x_i)  # phi_1
    phi[y_prev][INPUT + y_curr] = 1  # phi_2

    return phi


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
    label = ENG2I[letter]
    return x, label, next_id


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
    word = []
    for line in train_lines:
        data, label, next_id = split_sample(line)
        if next_id == '-1':
            word.append((data, label))
            train_labeled.append(word)
            word = []
        else:
            word.append((data, label))
    word = []
    for line in test_lines:
        data, label, next_id = split_sample(line)
        if next_id == '-1':
            word.append((data, label))
            test_labeled.append(word)
            word = []
        else:
            word.append((data, label))

    return train_labeled, test_labeled


def split_word(word_to_split):
    inputs = []
    labels = []

    for c in word_to_split:
        x, y = c
        inputs.append(x)
        labels.append(y)
    return inputs, labels


def word_prediction(word_inputs, weights):
    D_S = np.zeros((len(word_inputs), NUM_ENG_CHR))  # use for scores - size of word x 27
    D_PI = np.zeros((len(word_inputs), NUM_ENG_CHR))  # use to save the prev char index - size of word x 27
    ##################### INITIALIZATION #####################
    prev_char = ENG2I['$']  # should be 0

    x = word_inputs[0]
    for curr_char in range(1, NUM_ENG_CHR):
        y_hat = [prev_char, curr_char]
        phi = build_phi(x, y_hat)
        s = np.inner(weights, phi).trace()
        D_S[0][curr_char] = s
        D_PI[0][curr_char] = -1  # remember that PI is between 0-25! (not include $)!

    ##################### RECURSION #####################
    size = len(word_inputs)
    for i in range(1, size):
        x = inputs[i]
        for curr_char in range(1, NUM_ENG_CHR):  # remember! curr_char range is: 1-26 (a-z)
            tmp_d = d_best = i_best = -1
            for prev_char in range(1, NUM_ENG_CHR):  # remember! prev_char range is: 0-26 ($ + a-z)
                y_hat = [prev_char, curr_char]
                phi = build_phi(x, y_hat)
                tmp_d = np.inner(weights, phi).trace() + D_S[i - 1][prev_char]  # -1 because prev start with 1 not 0
                if tmp_d > d_best:
                    d_best = tmp_d
                    i_best = prev_char
            D_S[i][curr_char] = d_best
            D_PI[i][curr_char] = i_best
            # todo: check what to do if i_best is -1.. may cause problems!
    ##################### BACK-TRACK #####################
    # todo: change "shadow" + remove kinds of t1, o1 and so on
    pred = np.zeros(size, dtype=int)
    d_best = -1

    test = D_S[len(inputs) - 1]
    # t2 = np.array(test)
    y_last = np.argmax(test)
    pred[size - 1] = int(y_last)  # +1 because of $, 0-25

    for i in reversed(range(size - 1)):
        t1 = pred[i + 1]
        pred[i] = int(D_PI[i + 1][int(pred[i + 1])])  # each cell is 1-26

    return list(pred)


if __name__ == '__main__':
    print('#' * 30 + "\tSTART model 3\t" + '#' * 30)
    accuracy = []
    train_set, test_set = create_sets()

    # TRAIN:

    np.random.shuffle(train_set)
    counter = 0
    for epoch in range(EPOCHS):

        # train:

        t_good = 0.0
        t_bad = 0.0

        for word in train_set:
            inputs, labels = split_word(word)
            pred = word_prediction(inputs, W)

            for i, data in enumerate(inputs):
                if pred[i] != labels[i]:
                    counter += 1
                    t_bad += 1
                    # phi for y_label
                    if i == 0:
                        y_i = [0, labels[i]]
                    else:
                        y_i = [labels[i - 1], labels[i]]

                    t1 = build_phi(data, y_i)

                    # phi for y_pred
                    if i == 0:
                        y_i = [0, pred[i]]
                    else:
                        y_i = [pred[i - 1], pred[i]]

                    t2 = build_phi(data, y_i)

                    W = W + t1 - t2
                    W_sum = W_sum + W  # W1 + w2 + w3 ... wt

                else:
                    t_good += 1

        # end train in this epoch

        print('epoch #{:d}:\ttrain acc = {:d}/{:d}, {:.2f}'.format(epoch + 1, int(t_good), int(t_good + t_bad),
                                                                   t_good * 100 / (t_bad + t_good)))

        # TEST :
        if counter != 0:
            W_pred = W_sum / counter
        else:
            W_pred = W_sum
        good = bad = 0.0
        for word in test_set:
            inputs, labels = split_word(word)
            pred = word_prediction(inputs, W_pred)

            for i in range(len(pred)):
                if pred[i] == labels[i]:
                    good += 1
                else:
                    bad += 1

        print('epoch #{:d}:\ttest acc = {:d}/{:d}, {:.2f}'.format(epoch + 1, int(good), int(good + bad),
                                                                  good * 100 / (bad + good)))
        acc = good * 100 / (bad + good)
        accuracy.append(acc)
    # end of Epochs
    # print(accuracy)
    pickle.dump(accuracy, open("model3_acc.p", 'wb'))
    pickle.dump(W, open("model3_W.p", 'wb'))
    pickle.dump(W_pred, open("model3_W_pred.p", 'wb'))

    # SAVE PREDICTION
    with open("structured.pred", 'w') as f:
        for word in test_set:
            inputs, labels = split_word(word)
            pred = word_prediction(inputs, W_pred)

            for i in range(len(pred)):
                y_hat = I2ENG[pred[i]]
                f.write(y_hat + '\n')

    print('#' * 30 + "\tEND model 3\t" + '#' * 30)
