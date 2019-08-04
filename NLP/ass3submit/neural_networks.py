import dynet as dy
import utils
import numpy as np

LR = 0.001
WORD_EMBEDDING_DIM = 100
CHAR_EMBEDDING_DIM = 30
LSTM_DIM = 50
HID_DIM = 32


class Model(object):

    def __init__(self, rep):
        self.rep = rep

        self.W2I = utils.W2I
        self.T2I = utils.T2I
        self.C2I = utils.C2I
        self.I2W = utils.I2W
        self.I2T = utils.I2T
        self.I2C = utils.I2C
        self.P2I = utils.P2I
        self.S2I = utils.S2I

        self.model = dy.ParameterCollection()
        self.trainer = dy.AdamTrainer(self.model, LR)

        # word embedding matrix
        if rep == 'a':
            self.E = self.model.add_lookup_parameters((len(self.W2I), WORD_EMBEDDING_DIM))
        if rep == 'b':
            self.E_CHAR = self.model.add_lookup_parameters((len(self.C2I), CHAR_EMBEDDING_DIM))
            # acceptor, for chars to word size of 100
            self.char_LSTM = dy.LSTMBuilder(1, CHAR_EMBEDDING_DIM, WORD_EMBEDDING_DIM, self.model)
        if rep == 'c':
            self.E_PREF = self.model.add_lookup_parameters((len(self.P2I), WORD_EMBEDDING_DIM))
            self.E_SUFF = self.model.add_lookup_parameters((len(self.S2I), WORD_EMBEDDING_DIM))
        if rep == 'd':
            self.E = self.model.add_lookup_parameters((len(self.W2I), WORD_EMBEDDING_DIM))

            self.E_CHAR = self.model.add_lookup_parameters((len(self.C2I), CHAR_EMBEDDING_DIM))
            # acceptor, for chars to word size of 100
            self.char_LSTM = dy.LSTMBuilder(1, CHAR_EMBEDDING_DIM, WORD_EMBEDDING_DIM, self.model)

            self.W = self.model.add_parameters((WORD_EMBEDDING_DIM, WORD_EMBEDDING_DIM * 2))
            self.b = self.model.add_parameters(WORD_EMBEDDING_DIM)

        # builders = [first forward, first backward,....]

        self.builders = [
            dy.LSTMBuilder(1, WORD_EMBEDDING_DIM, LSTM_DIM, self.model),
            dy.LSTMBuilder(1, WORD_EMBEDDING_DIM, LSTM_DIM, self.model),
            dy.LSTMBuilder(1, WORD_EMBEDDING_DIM, LSTM_DIM, self.model),
            dy.LSTMBuilder(1, WORD_EMBEDDING_DIM, LSTM_DIM, self.model),
        ]

        # MLP mult on: b'1,..b'n
        # PH is from first BiLSTM so I choose the output dim
        # PO dim is same as TAG length - (classifier..)
        self.PH = self.model.add_parameters((HID_DIM, LSTM_DIM * 2))
        self.PO = self.model.add_parameters((len(self.T2I), HID_DIM))

    def build_tagging_graph(self, words, tags):
        dy.renew_cg()
        builders = self.builders
        f1_init, b1_init, f2_init, b2_init = [b.initial_state() for b in builders]
        # this is train graph! so every word I get I saw!

        wembs = []
        if self.rep == 'a':
            # all words here should be known!
            for w in words:
                wembs.append(self.E[self.W2I[w]])

        if self.rep == 'b':
            for word in words:
                char_indexes = []
                for c in word:
                    char_indexes.append(self.C2I[c])
                # till here got list of #chars * 30 in a WORD!!!
                char_embedding = [self.E_CHAR[i] for i in char_indexes]  # but this is a changing length. - do acceptor!
                char_lstm_init = self.char_LSTM.initial_state()
                wembs.append(char_lstm_init.transduce(char_embedding)[-1])

        if self.rep == 'c':
            for word in words:
                prefix = word[:3]
                suffix = word[-3:]
                pref_index = self.P2I[prefix]
                suff_index = self.S2I[suffix]
                wembs.append(dy.esum([self.E_PREF[pref_index], self.E_SUFF[suff_index]]))
        if self.rep == 'd':

            # making params for linear layer
            W = dy.parameter(self.W)
            b = dy.parameter(self.b)

            # all words here should be known!
            for w in words:
                temp1 = self.E[self.W2I[w]]
                temp2 = []
                char_indexes = []
                for c in w:
                    char_indexes.append(self.C2I[c])
                    # till here got list of #chars * 30 in a WORD!!!
                    char_embedding = [self.E_CHAR[i] for i in
                                      char_indexes]  # but this is a changing length. - do acceptor!
                    char_lstm_init = self.char_LSTM.initial_state()
                    temp2 = char_lstm_init.transduce(char_embedding)[-1]
                con = dy.concatenate([temp1, temp2])
                # LINER:
                res = ((W * con) + b)
                wembs.append(res)

        # first bilstm
        fw_1 = [x.output() for x in f1_init.add_inputs(wembs)]
        bw_1 = [x.output() for x in b1_init.add_inputs(reversed(wembs))]

        # concat the results
        b = [dy.concatenate([y1, y2]) for y1, y2 in zip(fw_1, bw_1)]

        fw_2 = [x.output() for x in f2_init.add_inputs(b)]
        bw_2 = [x.output() for x in b2_init.add_inputs(reversed(b))]

        H_mat = dy.parameter(self.PH)
        O_mat = dy.parameter(self.PO)

        errs = []
        for f, b, t in zip(fw_2, reversed(bw_2), tags):
            f_b = dy.concatenate([f, b])
            r_t = O_mat * (dy.tanh(H_mat * f_b))
            err = dy.pickneglogsoftmax(r_t, self.T2I[t])
            errs.append(err)
        return dy.esum(errs)

    def tag_sent(self, sent):
        dy.renew_cg()
        builders = self.builders
        f1_init, b1_init, f2_init, b2_init = [b.initial_state() for b in builders]

        wembs = []
        if self.rep == 'a':
            for w, t in sent:
                if w in self.W2I.keys():
                    wembs.append(self.E[self.W2I[w]])
                else:
                    wembs.append(self.E[self.W2I[utils.UNK]])
        if self.rep == 'b':

            for w, t in sent:
                char_indexes = []
                for c in w:
                    if c in self.C2I.keys():
                        char_indexes.append(self.C2I[c])
                    else:
                        char_indexes.append(self.C2I[utils.UNK])

                # till here got list of #chars * 30 in a WORD!!!
                char_embedding = [self.E_CHAR[i] for i in char_indexes]  # but this is a changing length. - do acceptor!
                char_lstm_init = self.char_LSTM.initial_state()
                wembs.append(char_lstm_init.transduce(char_embedding)[-1])

        if self.rep == 'c':
            for w, t in sent:
                prefix = w[:3]
                suffix = w[-3:]
                if prefix in self.P2I.keys():
                    pref_index = self.P2I[prefix]
                else:
                    pref_index = self.P2I[utils.UNK[:3]]

                if suffix in self.S2I.keys():
                    suff_index = self.S2I[suffix]
                else:
                    suff_index = self.S2I[utils.UNK[-3:]]
                wembs.append(dy.esum([self.E_PREF[pref_index], self.E_SUFF[suff_index]]))
        if self.rep == 'd':

            # making params for linear layer
            W = dy.parameter(self.W)
            b = dy.parameter(self.b)

            # all words here should be known!
            for w, t in sent:
                temp2 = []

                if w in self.W2I.keys():
                    temp1 = self.E[self.W2I[w]]
                else:
                    temp1 = self.E[self.W2I[utils.UNK]]

                char_indexes = []
                for c in w:
                    if c in self.C2I.keys():
                        char_indexes.append(self.C2I[c])
                    else:
                        char_indexes.append(self.C2I[utils.UNK])
                    # till here got list of #chars * 30 in a WORD!!!
                    char_embedding = [self.E_CHAR[i] for i in
                                      char_indexes]  # but this is a changing length. - do acceptor!
                    char_lstm_init = self.char_LSTM.initial_state()
                    temp2 = char_lstm_init.transduce(char_embedding)[-1]
                con = dy.concatenate([temp1, temp2])
                # LINER:
                res = ((W * con) + b)
                wembs.append(res)

        fw_1 = [x.output() for x in f1_init.add_inputs(wembs)]
        bw_1 = [x.output() for x in b1_init.add_inputs(reversed(wembs))]

        # concat the results
        b = [dy.concatenate([y1, y2]) for y1, y2 in zip(fw_1, bw_1)]

        fw_2 = [x.output() for x in f2_init.add_inputs(b)]
        bw_2 = [x.output() for x in b2_init.add_inputs(reversed(b))]

        H_mat = dy.parameter(self.PH)
        O_mat = dy.parameter(self.PO)

        tags = []
        for f, b, (w, t) in zip(fw_2, reversed(bw_2), sent):
            r_t = O_mat * (dy.tanh(H_mat * dy.concatenate([f, b])))
            out = dy.softmax(r_t)
            chosen = np.argmax(out.npvalue())
            tags.append(self.I2T[chosen])
        return tags

    def untagged_sent(self, sent):
        dy.renew_cg()
        builders = self.builders
        f1_init, b1_init, f2_init, b2_init = [b.initial_state() for b in builders]


        wembs = []
        if self.rep == 'a':
            for w in sent:
                if w in self.W2I.keys():
                    wembs.append(self.E[self.W2I[w]])
                else:
                    wembs.append(self.E[self.W2I[utils.UNK]])
        if self.rep == 'b':

            for w in sent:
                char_indexes = []
                for c in w:
                    if c in self.C2I.keys():
                        char_indexes.append(self.C2I[c])
                    else:
                        char_indexes.append(self.C2I[utils.UNK])

                # till here got list of #chars * 30 in a WORD!!!
                char_embedding = [self.E_CHAR[i] for i in char_indexes]  # but this is a changing length. - do acceptor!
                char_lstm_init = self.char_LSTM.initial_state()
                wembs.append(char_lstm_init.transduce(char_embedding)[-1])

        if self.rep == 'c':
            for w in sent:
                prefix = w[:3]
                suffix = w[-3:]
                if prefix in self.P2I.keys():
                    pref_index = self.P2I[prefix]
                else:
                    pref_index = self.P2I[utils.UNK[:3]]

                if suffix in self.S2I.keys():
                    suff_index = self.S2I[suffix]
                else:
                    suff_index = self.S2I[utils.UNK[-3:]]
                wembs.append(dy.esum([self.E_PREF[pref_index], self.E_SUFF[suff_index]]))
        if self.rep == 'd':

            # making params for linear layer
            W = dy.parameter(self.W)
            b = dy.parameter(self.b)

            # all words here should be known!
            for w in sent:
                temp2 = []

                if w in self.W2I.keys():
                    temp1 = self.E[self.W2I[w]]
                else:
                    temp1 = self.E[self.W2I[utils.UNK]]

                char_indexes = []
                for c in w:
                    if c in self.C2I.keys():
                        char_indexes.append(self.C2I[c])
                    else:
                        char_indexes.append(self.C2I[utils.UNK])
                    # till here got list of #chars * 30 in a WORD!!!
                    char_embedding = [self.E_CHAR[i] for i in
                                      char_indexes]  # but this is a changing length. - do acceptor!
                    char_lstm_init = self.char_LSTM.initial_state()
                    temp2 = char_lstm_init.transduce(char_embedding)[-1]
                con = dy.concatenate([temp1, temp2])
                # LINER:
                res = ((W * con) + b)
                wembs.append(res)


        fw_1 = [x.output() for x in f1_init.add_inputs(wembs)]
        bw_1 = [x.output() for x in b1_init.add_inputs(reversed(wembs))]

        # concat the results
        b = [dy.concatenate([y1, y2]) for y1, y2 in zip(fw_1, bw_1)]

        fw_2 = [x.output() for x in f2_init.add_inputs(b)]
        bw_2 = [x.output() for x in b2_init.add_inputs(reversed(b))]

        H_mat = dy.parameter(self.PH)
        O_mat = dy.parameter(self.PO)

        tags = []
        for f, b, w in zip(fw_2, reversed(bw_2), sent):
            r_t = O_mat * (dy.tanh(H_mat * dy.concatenate([f, b])))
            out = dy.softmax(r_t)
            chosen = np.argmax(out.npvalue())
            tags.append(self.I2T[chosen])
        return tags
