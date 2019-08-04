import sys
import neural_networks as nn
import utils
import pickle
from zipfile import ZipFile
import os
import matplotlib.pyplot as plt


def load_model(rep):
    with ZipFile(modelFile) as myzip:
        myzip.extractall(os.getcwd())

    dicts = pickle.load(open("dicts.p", "rb"))
    utils.set_dictionaries(dicts)
    net = nn.Model(rep)
    net.model.populate("model.dy")
    return net


def plot_acc():
    acc_a = pickle.load(open('pos_acc_of_a', 'rb'))
    acc_b = pickle.load(open('pos_acc_of_b', 'rb'))
    acc_c = pickle.load(open('pos_acc_of_c', 'rb'))
    acc_d = pickle.load(open('pos_acc_of_d', 'rb'))

    plt.clf()
    x_a_range = list(range(1, len(acc_a) + 1))
    x_a_range = [x * 5 for x in x_a_range]
    x_b_range = list(range(1, len(acc_b) + 1))
    x_b_range = [x * 5 for x in x_b_range]
    x_c_range = list(range(1, len(acc_c) + 1))
    x_c_range = [x * 5 for x in x_c_range]
    x_d_range = list(range(1, len(acc_d) + 1))
    x_d_range = [x * 5 for x in x_d_range]

    plt.plot(x_a_range, acc_a, label='model a')
    plt.plot(x_b_range, acc_b, label='model b')
    plt.plot(x_c_range, acc_c, label='model c')
    plt.plot(x_d_range, acc_d, label='model d')

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('sentences/100')
    plt.ylabel('dev accuracy')
    plt.title('POS accuracy per sentences')
    plt.savefig("POS.png")

    plt.clf()

    acc_a = pickle.load(open('ner_acc_of_a', 'rb'))
    acc_b = pickle.load(open('ner_acc_of_b', 'rb'))
    acc_c = pickle.load(open('ner_acc_of_c', 'rb'))
    acc_d = pickle.load(open('ner_acc_of_d', 'rb'))

    x_a_range = list(range(1, len(acc_a) + 1))
    x_a_range = [x * 5 for x in x_a_range]
    x_b_range = list(range(1, len(acc_b) + 1))
    x_b_range = [x * 5 for x in x_b_range]
    x_c_range = list(range(1, len(acc_c) + 1))
    x_c_range = [x * 5 for x in x_c_range]
    x_d_range = list(range(1, len(acc_d) + 1))
    x_d_range = [x * 5 for x in x_d_range]

    plt.plot(x_a_range, acc_a, label='model a')
    plt.plot(x_b_range, acc_b, label='model b')
    plt.plot(x_c_range, acc_c, label='model c')
    plt.plot(x_d_range, acc_d, label='model d')

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)

    plt.xlabel('sentences/100')
    plt.ylabel('dev accuracy')
    plt.title('NER accuracy per sentences')
    plt.savefig("NER.png")


if __name__ == '__main__':
    rep = sys.argv[1]
    modelFile = sys.argv[2]
    inputFile = sys.argv[3]  # pos model
    test_type = inputFile.split('/')[0]
    plot_acc()


    # change argv inputs for POS and NER - manually
    model = load_model(rep)

    test_set = utils.get_untagged_set(inputFile)
    pred_list = []
    for sentence in test_set:
        pred = model.untagged_sent(sentence)
        pred_list.append((sentence, pred))

    with open("test4." + test_type, 'w') as test_pred_file:
        for item in pred_list:
            sentence, tags = item
            for w, tag in zip(sentence, tags):
                test_pred_file.write(w + " " + tag + "\n")
            test_pred_file.write("\n")
    print("write file test4." + test_type)


