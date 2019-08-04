import cPickle as pickle
import utils


def read_train_data(file_name, tagger_type):
    tags_set = set()
    words_set = set()
    sentences = []

    try:
        with open(file_name, 'r') as f:
            sentence = []
            lines = f.readlines()
            for line in lines:
                if line == '\n':  # end line - means we can save this sentence
                    # add list of sentences with tags
                    sentences.append(sentence)
                    sentence = []
                    continue
                # TODO try remove split("\t")
                line = line.strip("\n").strip().strip("\t")
                word, tag = line.split()
                sentence.append((word, tag))
                tags_set.add(tag)
                words_set.add(word)
            # add START,END and UNK symbols
            words_set.add(utils.UNK_SYMBOL)
            words_set.add(utils.START_SYMBOL)
            words_set.add(utils.END_SYMBOL)
            tags_set.add(utils.UNK_SYMBOL)
            # no need the TAG - i will not tagged then anyway

        f.close()

        pickle.dump(tags_set, open(tagger_type + "TrainTagSet.p", "wb"))
        pickle.dump(words_set, open(tagger_type + "TrainWordSet.p", "wb"))
        pickle.dump(sentences, open(tagger_type + "TrainSentences.p", "wb"))
        print("real size of %sTrain WORD: %d and %sTrain TAG: %d %sTrain SENTENCES %d" %
              (str(tagger_type), len(words_set), str(tagger_type), len(tags_set), str(tagger_type), len(sentences)))

    except IOError:
        print("cannot open file")


def read_dev_data(file_name, tagger_type):
    sentences = []

    try:
        with open(file_name, 'r') as f:
            sentence = []
            lines = f.readlines()
            for line in lines:
                if line == '\n':  # end line - means we can save this sentence
                    # add list of sentences with tags
                    sentences.append(sentence)
                    sentence = []
                    continue
                # TODO try remove split("\t")
                line = line.strip("\n").strip().strip("\t")
                word, tag = line.split()
                sentence.append((word, tag))

        f.close()

        pickle.dump(sentences, open(tagger_type + "DevSentences.p", "wb"))
        print("real size of %s DEV SENTENCES %d" %(str(tagger_type), len(sentences)))

    except IOError:
        print("cannot open file")


def read_test_data(file_name, tagger_type):
    # must be test.. the only one without tags
    sentences = []

    try:
        with open(file_name, 'r') as f:
            sentence = []
            lines = f.readlines()
            for line in lines:
                if line == '\n':  # end line - means we can save this sentence
                    # add list of sentences with tags
                    sentences.append(sentence)
                    sentence = []
                    continue
                w = line.strip("\n").strip()
                sentence.append(w)

        f.close()

        pickle.dump(sentences, open(tagger_type + "TestSentences.p", "wb"))
        print("real size of %s TEST SENTENCES %d" %(str(tagger_type), len(sentences)))

    except IOError:
        print("cannot open file")


if __name__ == '__main__':
    print("#########\tstart ReadFiles\t#########")

    print("reads NER")
    read_train_data("data/ner/train", "NER")
    read_dev_data("data/ner/dev", "NER")
    read_test_data("data/ner/test", "NER")
    print("reads POS")
    read_train_data("data/pos/train", "POS")
    read_dev_data("data/pos/dev", "POS")
    read_test_data("data/pos/test", "POS")

    print("#########\tfinish ReadFiles\t#########")
