import torch
from torch.utils.data import Dataset
import utils
import numpy as np
import NNClass as nnClass
import cPickle as pickle
import NNTrainer


def get_train_dataset(tagger_type):
    sentences = pickle.load(open(tagger_type + "TrainSentences.p", "rb"))
    data, tag = utils.window_data_and_tag(sentences)
    x, y = np.asarray(data, np.float32), np.asarray(tag, np.int32)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    x, y = x.type(torch.LongTensor), y.type(torch.LongTensor)
    return torch.utils.data.TensorDataset(x, y)


def get_dev_dataset(tagger_type):
    sentences = pickle.load(open(tagger_type + "DevSentences.p", "rb"))
    data, tag = utils.window_data_and_tag(sentences)
    x, y = np.asarray(data, np.float32), np.asarray(tag, np.int32)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    x, y = x.type(torch.LongTensor), y.type(torch.LongTensor)
    return torch.utils.data.TensorDataset(x, y)


def get_test_dataset(tagger_type):
    sentences = pickle.load(open(tagger_type + "TestSentences.p", "rb"))
    data = utils.window_data(sentences)
    x = np.asarray(data, np.float32)
    x = torch.from_numpy(x)
    x = x.type(torch.LongTensor)
    return x


def init_sets(tagger_type):
    tag_set = pickle.load(open(tagger_type + "TrainTagSet.p", "rb"))
    word_set = pickle.load(open(tagger_type + "TrainWordSet.p", "rb"))
    utils.init_words_and_tags(word_set, tag_set)


if __name__ == '__main__':
    # train NER
    init_sets("NER")
    train_data = get_train_dataset("NER")
    dev_data = get_dev_dataset("NER")
    test_data = get_test_dataset("NER")

    model = nnClass.NET()
    trainer = NNTrainer.ModelTrainer(model, train_data, dev_data, test_data, "NER")
    trainer.run()

    # train NER
    init_sets("POS")
    train_data = get_train_dataset("POS")
    dev_data = get_dev_dataset("POS")
    test_data = get_test_dataset("POS")

    model = nnClass.NET()
    trainer = NNTrainer.ModelTrainer(model, train_data, dev_data, test_data, "POS")
    trainer.run()

    print("finish tagger1")
