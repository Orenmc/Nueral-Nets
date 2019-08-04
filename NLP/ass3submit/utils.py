import pickle

# globals:
WORDS_SET = set()
TAGS_SET = set()
CHARS_SET = set()

UNK = 'uuunnnkk'
PREFIX_SIZE = 3
SUFFIX_SIZE = 3

W2I = {}
T2I = {}
C2I = {}
I2W = {}
I2T = {}
I2C = {}
P2I = {}
S2I = {}


def get_tagged_set(file_name, embedding_flag=True):
    global UNK
    sentences = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
        sentence = []
        for line in lines:

            if line == '\n':
                # end of sentence
                sentences.append(sentence)
                sentence = []  # clear
            else:
                word, tag = line.strip('\n').strip().split()
                sentence.append((word, tag))
                # time to do embedding!
                if embedding_flag:
                    # add embedding...
                    add_embedding(word, tag)
        if embedding_flag:
            # add UNK to embedding
            add_embedding(UNK, UNK)
    return sentences


def add_embedding(word, tag):
    global WORDS_SET, TAGS_SET, CHARS_SET

    WORDS_SET.add(word)
    TAGS_SET.add(tag)
    # TODO: create unknown char! in case there is a char in test that not found in train
    if word == UNK:
        CHARS_SET.add(UNK)
    else:
        for c in word:
            CHARS_SET.add(c)


def get_untagged_set(file_name):
    sentences = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
        sentence = []
        for line in lines:

            if line == '\n':
                # end of sentence
                sentences.append(sentence)
                sentence = []  # clear
            else:
                word = line.strip('\n').strip()
                sentence.append(word)

    return sentences


def update_indexing():
    global W2I, T2I, C2I, I2C, I2T, I2W, P2I, S2I
    W2I = {word: i for i, word in enumerate(WORDS_SET)}
    I2W = {i: word for word, i in W2I.items()}
    T2I = {tag: i for i, tag in enumerate(TAGS_SET)}
    I2T = {i: word for word, i in T2I.items()}
    C2I = {tag: i for i, tag in enumerate(CHARS_SET)}
    I2C = {i: word for word, i in C2I.items()}
    # initialize prefixes and suffixes
    prefixes = {word[:PREFIX_SIZE] for word in WORDS_SET}
    suffixes = {word[-SUFFIX_SIZE:] for word in WORDS_SET}
    P2I = {word[:PREFIX_SIZE]: i for i, word in enumerate(prefixes)}
    S2I = {word[-SUFFIX_SIZE:]: i for i, word in enumerate(suffixes)}


def get_sets_and_init_indexes(train_file_name):
    train_set = get_tagged_set(train_file_name, embedding_flag=True)
    update_indexing()
    find_dev_name = train_file_name.split('/')
    find_dev_name[-1] = 'dev'
    test_name = '/'.join(find_dev_name)
    dev_set = get_tagged_set(test_name, embedding_flag=False)

    return train_set, dev_set


def set_dictionaries(dicts):
    global W2I, T2I, C2I, I2C, I2T, I2W, P2I, S2I

    W2I = dicts[0]
    T2I = dicts[1]
    C2I = dicts[2]
    I2W = dicts[3]
    I2T = dicts[4]
    I2C = dicts[5]
    P2I = dicts[6]
    S2I = dicts[7]


def get_words_and_tags_from_sent(sent):
    words = [word for word, tag in sent]
    tags = [tag for word, tag in sent]
    return words, tags


if __name__ == '__main__':
    print('#' * 20 + '\tstarts utils\t' + '#' * 20)

    print('#' * 20 + '\tend utils\t' + '#' * 20)
