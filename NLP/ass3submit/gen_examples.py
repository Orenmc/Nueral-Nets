import rstr
import random
import string
from numpy import random as rand


def create_regular_expression(a1, a2, a3, a4):
    """
    creates regular expression: digits+a1+digits+a2+digits+a3+digits+a4+digits
    notice - the default length of rstr is 1-10 (so the avg is 5) so the expression should
    be in length of 9*5=  45 +-
    :param a1: first separator
    :param a2: second separator
    :param a3: third separator
    :param a4: fourth separator
    :return: regular expression: [1-9]+a1+[1-9]+a2+[1-9]+a3+[1-9]+a4+[1-9]+
    """
    temp = rstr.rstr(string.digits)
    temp += rstr.rstr(a1)
    temp += rstr.rstr(string.digits)
    temp += rstr.rstr(a2)
    temp += rstr.rstr(string.digits)
    temp += rstr.rstr(a3)
    temp += rstr.rstr(string.digits)
    temp += rstr.rstr(a4)
    temp += rstr.rstr(string.digits)
    return temp


def create_x_examples(amount, a1, a2, a3, a4):
    """
    creates x times regular expression
    :param amount: amount of examples to create
    :param a1: first separator
    :param a2: second separator
    :param a3: third separator
    :param a4: fourth separator
    :return: regular expression: [1-9]+a1+[1-9]+a2+[1-9]+a3+[1-9]+a4+[1-9]+
    """
    examples = []
    for i in range(amount):
        examples.append(create_regular_expression(a1, a2, a3, a4))
    return examples


def create_files():
    """
    creates 2 files: pos_examples and neg_examples with 500 examples each
    :return: void
    """
    with open("pos_examples", 'w') as f:
        lines = '\n'.join(create_x_examples(500, 'a', 'b', 'c', 'd'))
        f.write(lines)
    with open("neg_examples", 'w') as f:
        lines = '\n'.join(create_x_examples(500, 'a', 'c', 'b', 'd'))
        f.write(lines)


def get_train_and_test(pos_examples_file_name, neg_examples_file_name):
    """
    creates train_set and test_set from files with examples.
    :return: train and test
    """
    with open(neg_examples_file_name, "r") as f1, open(pos_examples_file_name, "r") as f2:
        f1_lines, f2_lines = f1.readlines(), f2.readlines()
        pos_labeled = [line.strip('\n') + ' ' + "1" for line in f2_lines]
        neg_labeled = [line.strip('\n') + ' ' + "0" for line in f1_lines]
        labeled_examples = pos_labeled + neg_labeled
        random.shuffle(labeled_examples)
    index = int(0.8 * len(labeled_examples))
    train_set = labeled_examples[:index]
    test_set = labeled_examples[index:]
    return train_set, test_set


def create_fails_examples():
    pos_examples = []
    neg_examples = []

    # creates language W#W
    for i in range(500):
        w1 = rstr.rstr(string.digits, 1, 50)
        w2 = rstr.rstr(string.digits, len(w1))
        while w1 == w2:
            w2 = rstr.rstr(string.digits, len(w1))

        pos_examples.append(w1 + '#' + w1)
        neg_examples.append(w1 + '#' + w2)

    # till here I have 500 good examples and 500 bad examples
    with open("fail_1_pos_examples", 'w') as f:
        lines = '\n'.join(pos_examples)
        f.write(lines)
    with open("fail_1_neg_examples", 'w') as f:
        lines = '\n'.join(neg_examples)
        f.write(lines)

    # creates language w#w^r (i.e, 1234554321)
    pos_examples = []
    neg_examples = []
    for i in range(500):
        w1 = rstr.rstr(string.digits, 1, 50)
        w2 = rstr.rstr(string.digits, len(w1))
        w3 = w1[::-1]

        while w3 == w2:
            w2 = rstr.rstr(string.digits, len(w1))

        pos_examples.append(w1 + '#' + w3)
        neg_examples.append(w1 + '#' + w2)

    # till here I have 500 good examples and 500 bad examples
    with open("fail_2_pos_examples", 'w') as f:
        lines = '\n'.join(pos_examples)
        f.write(lines)
    with open("fail_2_neg_examples", 'w') as f:
        lines = '\n'.join(neg_examples)
        f.write(lines)

    # third fail lang
    pos_examples = []
    neg_examples = []
    for i in range(500):
        w1 = rstr.digits() + rstr.rstr('a') + rstr.rstr(string.digits, 100, 300) + rstr.rstr('b') + \
             rstr.rstr(string.digits, 100, 300) + rstr.rstr('c') + rstr.rstr(string.digits, 100, 300) + \
             rstr.rstr('d') + rstr.digits()

        w2 = rstr.digits() + rstr.rstr('a') + rstr.rstr(string.digits, 100, 300) + rstr.rstr('c') + \
             rstr.rstr(string.digits, 100, 300) + rstr.rstr('b') + rstr.rstr(string.digits, 100, 300) + \
             rstr.rstr('d') + rstr.digits()

        while w1 == w2:
            w2 = rstr.digits() + rstr.rstr('b') + rstr.rstr(string.digits, 200, 500) + rstr.rstr('a') + rstr.digits()

        pos_examples.append(w1)
        neg_examples.append(w2)

    # till here I have 500 good examples and 500 bad examples
    with open("fail_3_pos_examples", 'w') as f:
        lines = '\n'.join(pos_examples)
        f.write(lines)
    with open("fail_3_neg_examples", 'w') as f:
        lines = '\n'.join(neg_examples)
        f.write(lines)


if __name__ == '__main__':
    print("gen_examples - START!")
    print("\ncreates files")
    create_files()
    create_fails_examples()
    print("gen_examples - DONE!")
