import cPickle as pickle

###############
#   globals   #
###############

WORD_TO_INDEX = {}
INDEX_TO_WORD = {}
TAG_TO_INDEX = {}
INDEX_TO_TAG = {}

START_SYMBOL = '*START*'
END_SYMBOL = '*END*'
UNK_SYMBOL = "UUUNKKK"

WORD_SET = []
TAG_SET = []


def init_words_and_tags(word_set, tag_set):
    global WORD_TO_INDEX, INDEX_TO_WORD, TAG_TO_INDEX, INDEX_TO_TAG
    global WORD_SET, TAG_SET

    WORD_SET = word_set
    TAG_SET = tag_set

    WORD_TO_INDEX = {word: i for i, word in enumerate(word_set)}
    INDEX_TO_WORD = {i: word for word, i in WORD_TO_INDEX.iteritems()}
    TAG_TO_INDEX = {tag: i for i, tag in enumerate(tag_set)}
    INDEX_TO_TAG = {i: tag for tag, i in TAG_TO_INDEX.iteritems()}


def pad_sentence_with_tags(sentence):
    """
    get sentence (Word1,Tag1 Word2,Tag2, ...., Word n, Tag n)
    and pad with START START (sentence) END END
    :param sentence: (Word1,Tag1 Word2,Tag2, ...., Word n, Tag n)
    :return: padded sentence
    """
    global START_SYMBOL, END_SYMBOL
    pad = [(START_SYMBOL, START_SYMBOL), (START_SYMBOL, START_SYMBOL)]
    pad.extend(sentence)
    pad.extend([(END_SYMBOL, END_SYMBOL), (END_SYMBOL, END_SYMBOL)])
    return pad


def pad_sentence(sentence):
    """
    get sentence (Word1,Tag1 Word2,Tag2, ...., Word n, Tag n)
    and pad with START START (sentence) END END
    :param sentence: (Word1,Tag1 Word2,Tag2, ...., Word n, Tag n)
    :return: padded sentence
    """
    global START_SYMBOL, END_SYMBOL
    pad = [START_SYMBOL, START_SYMBOL]
    pad.extend(sentence)
    pad.extend([END_SYMBOL, END_SYMBOL])
    return pad


def words_to_index(words):
    global UNK_SYMBOL
    transform = []
    for word in words:
        if word in WORD_TO_INDEX:
            transform.append(WORD_TO_INDEX[word])
        else:
            # needed for the dev part
            transform.append(WORD_TO_INDEX[UNK_SYMBOL])
    return transform


def window_data_and_tag(sentences, lowercase=False):
    # this god for train and dev
    global START_SYMBOL, END_SYMBOL
    window_word_inputs = []
    window_tag_inputs = []

    for sentence in sentences:
        sentence = pad_sentence_with_tags(sentence)
        for i, (word, tag) in enumerate(sentence):
            if word != START_SYMBOL and word != END_SYMBOL:
                # it must that we have at least 5 words (start start Word end end)
                if lowercase:
                    window = [sentence[i - 2][0].lower(), sentence[i - 1][0].lower(),
                              word.lower(), sentence[i + 1][0].lower(), sentence[i + 2][0].lower()]
                else:
                    window = [sentence[i - 2][0], sentence[i - 1][0],
                              word, sentence[i + 1][0], sentence[i + 2][0]]

                window = words_to_index(window)
                window_word_inputs.append(window)
                window_tag_inputs.append(TAG_TO_INDEX[tag])
    return window_word_inputs, window_tag_inputs


def window_data(sentences, lowercase=False):
    # this good for test
    global START_SYMBOL, END_SYMBOL
    window_word_inputs = []

    for sentence in sentences:
        sentence = pad_sentence(sentence)
        for i, (word) in enumerate(sentence):
            if word != START_SYMBOL and word != END_SYMBOL:
                # it must that we have at least 5 words (start start Word end end)
                if lowercase:
                    window = [sentence[i - 2].lower(), sentence[i - 1].lower(),
                              word.lower(), sentence[i + 1].lower(), sentence[i + 2].lower()]
                else:
                    window = [sentence[i - 2], sentence[i - 1], word, sentence[i + 1], sentence[i + 2]]
                window = words_to_index(window)
                window_word_inputs.append(window)
    return window_word_inputs


if __name__ == '__main__':
    print("#####\tstart utils\t#####")
    print("test of sizes with pickle:")
    w = pickle.load(open("NERTrainWordSet.p", "rb"))
    t = pickle.load(open("NERTrainTagSet.p", "rb"))
    s = pickle.load(open("NERTrainSentences.p", "rb"))
    dev = pickle.load(open("NERDevSentences.p", "rb"))
    test = pickle.load(open("NERTestSentences.p", "rb"))

    print("NER train WORD size: %d, NER train TAG WORD size: %d ner train sentence: %d"
          % (len(w), len(t), len(s)))
    print("ner dev sentences: %d, ner Test sentences: %d\n\n" % (len(dev), len(test)))

    w = pickle.load(open("POSTrainWordSet.p", "rb"))
    t = pickle.load(open("POSTrainTagSet.p", "rb"))
    s = pickle.load(open("POSTrainSentences.p", "rb"))
    dev = pickle.load(open("POSDevSentences.p", "rb"))
    test = pickle.load(open("POSTestSentences.p", "rb"))

    print("POS train WORD size: %d, POS train TAG WORD size: %d POS train sentence: %d"
          % (len(w), len(t), len(s)))
    print("POS dev sentences: %d, POS Test sentences: %d" % (len(dev), len(test)))

    print("#####\tend utils\t#####")
