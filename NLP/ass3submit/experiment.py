import dynet as dy
import numpy as np
import gen_examples as examples
import random
import os
from time import time
import sys

# globals:

EMBED_SIZE = 100
characters = list("0123456789abcd#")
int2char = list(characters)
char2int = {c: i for i, c in enumerate(characters)}
VOCAB_SIZE = len(characters)
EPOCHS = 10


# acceptor LSTM
class LstmAcceptor(object):
    def __init__(self, in_dim, lstm_dim, out_dim, model):
        self.builder = dy.VanillaLSTMBuilder(1, in_dim, lstm_dim, model)
        self.W = model.add_parameters((out_dim, lstm_dim))

    def __call__(self, sequence):
        lstm = self.builder.initial_state()
        W = self.W.expr()  # convert the parameter into an Expession (add it to graph)
        outputs = lstm.transduce(sequence)
        result = W * outputs[-1]
        return result


if __name__ == '__main__':

    neg_examples = sys.argv[1]
    pos_examples = sys.argv[2]
    print('#'*20)
    if not (os.path.exists(pos_examples) and os.path.exists(neg_examples)):
        print("files not found - creates files.")
        examples.create_files()
    else:
        print("neg and pos examples files has been found.")

    print('#' * 20 + '\n')
    # model:
    m = dy.Model()
    trainer = dy.AdamTrainer(m)
    embeds = m.add_lookup_parameters((VOCAB_SIZE, EMBED_SIZE))
    acceptor = LstmAcceptor(EMBED_SIZE, 100, 2, m)

    train_set, test_set = examples.get_train_and_test(pos_examples, neg_examples)

    start_time = time()
    for epoch in range(EPOCHS):

        print("\nepoch : #{}".format(epoch + 1))
        # for each epoch shuffle train and test
        random.shuffle(train_set)
        random.shuffle(test_set)
        sum_of_losses = 0.0
        good = 0.0

        train_time = time()
        # training code
        for line in train_set:
            sequence, label = line.split(' ')
            dy.renew_cg()  # new computation graph
            vecs = [embeds[char2int[c]] for c in sequence]  # convert to n embeds words (n is the size of length of seq)
            preds = acceptor(vecs)  # preds has unchanged size (2- {0,1})
            loss = dy.pickneglogsoftmax(preds, int(label))  # label must be int.
            sum_of_losses += loss.npvalue()
            loss.backward()
            trainer.update()
        print('train took {}'.format(time() - train_time))
        print('train avg loss: {:.6f}'.format(int(sum_of_losses) / len(train_set)))  # calculate avg loss

        test_time = time()
        # prediction(test) code:
        for line in test_set:
            sequence, label = line.split(' ')
            dy.renew_cg()  # new computation graph
            vecs = [embeds[char2int[c]] for c in sequence]
            preds = dy.softmax(acceptor(vecs))
            vals = preds.npvalue()
            if np.argmax(vals) == int(label):
                good += 1
        print('test took {}'.format(time() - test_time))
        print("test accuracy: {:.2f}%".format(good * 100 / len(test_set)))
        print('time till now: {}'.format(time() - start_time))
