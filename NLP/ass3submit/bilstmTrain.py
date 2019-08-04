import sys
import utils
import neural_networks as nn
import random
import pickle
from zipfile import ZipFile
import os

EPOCHS = 3


def save_model(model_file):
    # save the indexers of words, tags, chars, prefixes and suffixes sets in dicts.pkl
    dicts = [utils.W2I, utils.T2I, utils.C2I, utils.I2W, utils.I2T, utils.I2C, utils.P2I, utils.S2I]
    pickle.dump(dicts, open("dicts.p", "wb"), pickle.HIGHEST_PROTOCOL)
    # save the dy-net model in model.dy
    model.model.save("model.dy")
    zip_file = ZipFile(model_file, "w")
    # zip dicts.pkl and model.dy
    zip_file.write("dicts.p")
    zip_file.write("model.dy")
    zip_file.close()
    # remove the files after writing them to the model_file zip
    os.remove("dicts.p")
    os.remove("model.dy")


if __name__ == '__main__':
    print('#' * 20 + '\tstarts bilstmTrain\t' + '#' * 20)

    # get arguments from input
    rep = sys.argv[1]
    trainFile = sys.argv[2]
    modelFile = sys.argv[3]

    print(rep + '\t' + trainFile + '\t' + modelFile)

    train_set, dev_set = utils.get_sets_and_init_indexes(trainFile)
    # todo: remove

    # print('train size: {}, so {} points'.format(len(train_set), int(len(train_set) / 500)))
    model = nn.Model(rep)

    trainer = model.trainer

    tagged = loss = 0
    dev_acc = []
    for ITER in range(EPOCHS):
        print("Epoch #{}/{}:".format(ITER + 1, EPOCHS))
        random.shuffle(train_set)
        for i, s in enumerate(train_set, 1):
            if i % 500 == 0:
                good = bad = 0.0
                for sent in dev_set:
                    tags = model.tag_sent(sent)
                    golds = [t for w, t in sent]
                    for go, gu in zip(golds, tags):
                        if go == gu:
                            if go != 'O':  # this is for case 'O' == 'O'
                                good += 1
                        else:
                            bad += 1
                acc = good / (good + bad) * 100  # percents
                print(acc)
                dev_acc.append(acc)

            words, tags = utils.get_words_and_tags_from_sent(s)
            sum_errs = model.build_tagging_graph(words, tags)
            squared = -sum_errs  # * sum_errs
            loss += sum_errs.scalar_value()
            tagged += len(tags)
            sum_errs.backward()
            trainer.update()

    # save model:
    save_model(modelFile)
    # save accuracy
    train_type = trainFile.split('/')[0]
    with open(train_type + '_acc_of_' + rep, 'wb') as fp:
        pickle.dump(dev_acc, fp)

    print('#' * 20 + '\tend bilstmTrain\t' + '#' * 20)
