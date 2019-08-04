import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt

train_acc = []
val_acc = []
train_loss = []
val_loss = []


class NeuralNetwork:
    def __init__(self, model_name='last_model', hidden=None, epochs=50, lr=0.001, norm=False, L2=False):
        # self.lambd = None
        self.train_data = None
        self.train_labels = None
        self.val_data = None
        self.val_labels = None
        self.test_data = None
        self.parameters = None
        self.std = None
        self.mean = None
        self.norm = None
        self.lr = lr
        self.epochs = epochs
        self.hidden = hidden
        self.model_name = model_name
        self.norm = norm
        self.in_dropout = None
        self.L2 = L2

    def load_data(self, train_path, val_path, test_path):
        """
        load train, validation and test data (and labels), with option to normalize all data
        :param train_path: path to train file
        :param val_path: path to validation file
        :param test_path: path to test file
        """
        # train data load with pandas
        train_set = pd.read_csv(train_path, header=None).to_numpy()

        train_data = train_set[:, 1:]  # firs column is for labels
        train_labels = train_set[:, 0] - 1  # original is between 1-10 (argmax is 0-9)
        self.train_labels = train_labels
        self.train_data = train_data

        # validation data load with pandas
        validation = pd.read_csv(val_path, header=None).to_numpy()
        val_data = validation[:, 1:]  # firs column is for labels
        val_labels = validation[:, 0] - 1  # original is between 1-10 (argmax is 0-9)
        self.val_labels = val_labels
        self.val_data = val_data

        # test data load with pandas
        test = pd.read_csv(test_path, header=None).to_numpy()
        test_data = test[:, 1:]  # firs column is for labels
        self.test_data = test_data

        print('data loaded')
        if self.norm:
            self.norm_data()
            print('data normalized')

    def norm_data(self):
        self.mean = np.mean(self.train_data)
        self.std = np.std(self.train_data)
        norm_data = (self.train_data - self.mean) / self.std

        if abs(np.mean(norm_data) - 0) > 1e-5 or abs(np.std(norm_data) - 1) > 1e-5:  # check if normalization worked
            print('something went wrong with the normalization')
            print('values are: mean: {}, std: {}'.format(np.mean(norm_data), np.std(norm_data)))
            exit(1)
        self.train_data = norm_data
        self.val_data = (self.val_data - self.mean) / self.std
        self.test_data = (self.test_data - self.mean) / self.std

    def initialize_parameters(self):
        """
        :return: 
        """"""
        # """

        W = []
        B = []

        assert len(self.hidden) >= 1, 'hidden must be more then 1 layer'
        n_x = self.train_data.shape[1]  # size of input layer
        n_y = 10  # 10 classes
        prev_size = n_x
        for size in self.hidden:
            # w = np.random.randn(size, prev_size) * np.sqrt(2 / n_x) # TODO: not very good. do not converge
            w = np.random.randn(size, prev_size) * 0.01
            b = np.zeros((size, 1))

            W.append(w)
            B.append(b)
            prev_size = size
        w = np.random.randn(n_y, prev_size) * 0.01
        b = np.zeros((n_y, 1))

        W.append(w)
        B.append(b)

        # W1 = np.random.randn(n_h_1, n_x) * 0.01
        # b1 = np.zeros((n_h_1, 1))
        #
        # W2 = np.random.randn(n_h_2, n_h_1) * 0.01
        # b2 = np.zeros((n_h_2, 1))
        #
        # W3 = np.random.randn(n_y, n_h_2) * 0.01
        # b3 = np.zeros((n_y, 1))

        self.parameters = {"W": W,
                           "B": B}

    def fprop(self, data):
        if data.ndim == 1:
            data = data[:, np.newaxis]
        # Retrieve each parameter from the dictionary "parameters"
        W, B = self.parameters['W'], self.parameters['B']

        Z = []
        A = []
        input_data = data
        for i in range(len(W) - 1):
            z = np.dot(W[i], input_data) + B[i]
            a = utils.ReLU(z)  # activation func
            input_data = a  # for next layer
            Z.append(z)
            A.append(a)
        z = np.dot(W[-1], input_data) + B[-1]
        a = utils.softmax(z)  # activation func
        Z.append(z)
        A.append(a)

        # # TODO: sanity check
        # W1 = W[0]
        # b1 = B[0]
        # W2 = W[1]
        # b2 = B[1]
        #
        # Z1 = np.dot(W1, data) + b1
        # A1 = utils.ReLU(Z1)  # activation func
        # Z2 = np.dot(W2, A1) + b2
        # A2 = utils.softmax(Z2)  # activation func
        #
        # assert np.array_equal(Z1, Z[0]) and np.array_equal(Z2, Z[1]), 'sanity check'
        # assert np.array_equal(A1, A[0]) and np.array_equal(A2, A[1]), 'sanity check'

        cache = {"Z": Z,
                 "A": A}

        return a, cache

    def bprop(self, cache, x, label):
        y_tag = np.zeros((10, 1))
        y_tag[int(label)] += 1
        if x.ndim == 1:
            x = x[:, np.newaxis]
        # First, retrieve W1 and W2 from the dictionary "parameters".
        # parameters = self.parameters
        W, B = self.parameters["W"], self.parameters["B"]
        Z, A = cache["Z"], cache["A"]
        L2_reg = []
        if self.L2:
            L2_reg = utils.dL2(W, self.L2)

        # TODO: without L2
        dW = []
        dB = []
        dz = A[-1] - y_tag
        for i in reversed(range(1, len(W))):
            dw = np.dot(dz, A[i - 1].T)
            if self.L2:
                dw += L2_reg[i]
            db = dz
            dW.insert(0, dw)
            dB.insert(0, db)
            dz = np.dot(W[i].T, dz) * utils.dReLU(Z[i - 1])
        dw = np.dot(dz, x.T)
        db = dz
        dW.insert(0, dw)
        dB.insert(0, db)

        # TODO: for checking:
        # W1 = W[0]
        # b1 = B[0]
        # W2 = W[1]
        # b2 = B[1]
        #
        # # Retrieve also A1 and A2 from dictionary "cache".
        # Z1 = Z[0]
        # A1 = A[0]
        # A2 = A[1]
        #
        # # Backward propagation: calculate dW1, db1, dW2, db2. Add the regularization term to dW2,dW1
        # dZ2 = A2 - y_tag
        # dW2 = np.dot(dZ2, A1.T)
        # db2 = dZ2
        # dZ1 = np.dot(W2.T, dZ2) * utils.dReLU(Z1)
        # dW1 = np.dot(dZ1, x.T)
        # db1 = dZ1
        #
        # assert np.array_equal(dW2, dW[1]) and np.array_equal(dW1, dW[0]), 'sanity check'
        #
        # assert np.array_equal(db2, dB[1]) and np.array_equal(db1, dB[0]), 'sanity check'

        grads = {"dW": dW,
                 "dB": dB}
        return grads

        # TODO: last time - hard coded

        # W1 = parameters["W1"]
        # W2 = parameters["W2"]
        # W3 = parameters["W3"]

        # Retrieve also A1 and A2 from dictionary "cache".

        # Z1 = cache["Z1"]
        # Z2 = cache["Z2"]
        # A1 = cache["A1"]
        # A2 = cache["A2"]
        # A3 = cache["A3"]

        # if self.L2:
        #     # Backward propagation: calculate dW1, db1, dW2, db2. Add the regularization term to dW2,dW1
        #     dZ2 = A2 - y_tag
        #     # dW2 = (1.0 / m) * np.matmul(dZ2, np.transpose(A1)) + (lambd / m) * W2  ## add the regularization term
        #     dW2 = np.dot(dZ2, A1.T) + self.lambd * W2  ## add the regularization term
        #     db2 = dZ2
        #
        #     dZ1 = np.dot(W2.T, dZ2) * utils.dReLU(Z1)
        #     dW1 = np.dot(dZ1, x.T) + self.lambd * W1  ## add the regularization
        #     db1 = dZ1
        # else:
        # Backward propagation: calculate dW1, db1, dW2, db2. Add the regularization term to dW2,dW1
        # dZ2 = A2 - y_tag
        # dW2 = np.dot(dZ2, A1.T)
        # db2 = dZ2
        # dZ1 = np.dot(W2.T, dZ2) * utils.dReLU(Z1)
        # dW1 = np.dot(dZ1, x.T)
        # db1 = dZ1
        #
        # grads = {"dW1": dW1,
        #          "db1": db1,
        #          "dW2": dW2,
        #          "db2": db2}

        # if self.L2:
        #     dZ3 = A3 - y_tag
        #     # dW3 = np.dot(dZ3, A2.T) + self.L2 * W3
        #     # TODO: remove
        #     dW3 = np.dot(dZ3, A2.T)
        #     dW3 += self.L2 * W3
        #     db3 = dZ3
        #     dZ2 = np.dot(W3.T, dZ3) * utils.dReLU(Z2)
        #     dW2 = np.dot(dZ2, A1.T) + self.L2 * W2
        #     db2 = dZ2
        #     dZ1 = np.dot(W2.T, dZ2) * utils.dReLU(Z1)
        #     dW1 = np.dot(dZ1, x.T) + self.L2 * W1
        #     db1 = dZ1
        #
        # else:
        #     dZ3 = A3 - y_tag
        #     dW3 = np.dot(dZ3, A2.T)
        #     db3 = dZ3
        #     dZ2 = np.dot(W3.T, dZ3) * utils.dReLU(Z2)
        #     dW2 = np.dot(dZ2, A1.T)
        #     db2 = dZ2
        #     dZ1 = np.dot(W2.T, dZ2) * utils.dReLU(Z1)
        #     dW1 = np.dot(dZ1, x.T)
        #     db1 = dZ1
        #
        # grads = {"dW1": dW1,
        #          "db1": db1,
        #          "dW2": dW2,
        #          "db2": db2,
        #          "dW3": dW3,
        #          "db3": db3}
        # return grads

    def weights_updates(self, grads):
        W, B = self.parameters["W"], self.parameters["B"]
        dW, dB = grads["dW"], grads["dB"]

        assert len(W) == len(B) == len(dW) == len(dB), 'must be in same length'

        # #  TODO: check
        # cost_before = utils.L2_cost(W, self.L2)

        new_W = []
        new_B = []
        for w, b, dw, db in zip(W, B, dW, dB):
            w = w - self.lr * dw
            b = b - self.lr * db
            new_W.append(w)
            new_B.append(b)

        self.parameters = {"W": new_W,
                           "B": new_B}

        # # TODO: check
        # cost_after = utils.L2_cost(new_W, self.L2)
        #
        # print('check L2 before: {}, and after: {}'.format(cost_before, cost_after))
        # # TODO: sanity check
        # W1 = W[0]
        # b1 = B[0]
        # W2 = W[1]
        # b2 = B[1]
        #
        # dW1 = dW[0]
        # db1 = dB[0]
        # dW2 = dW[1]
        # db2 = dB[1]
        #
        # W1 = W1 - self.lr * dW1
        # b1 = b1 - self.lr * db1
        # W2 = W2 - self.lr * dW2
        # b2 = b2 - self.lr * db2
        #
        # assert np.array_equal(W1, new_W[0]) and np.array_equal(W2, new_W[1]), 'sanity check'
        # assert np.array_equal(b1, new_B[0]) and np.array_equal(b2, new_B[1]), 'sanity check'

        # Retrieve each parameter from the dictionary "parameters"

        # W1 = parameters["W1"]
        # b1 = parameters["b1"]
        # W2 = parameters["W2"]
        # b2 = parameters["b2"]
        # W3 = parameters["W3"]
        # b3 = parameters["b3"]

        # Retrieve each gradient from the dictionary "grads"
        # dW1 = grads["dW1"]
        # db1 = grads["db1"]
        # dW2 = grads["dW2"]
        # db2 = grads["db2"]
        # dW3 = grads["dW3"]
        # db3 = grads["db3"]

        # Update rule for each parameter

        # W1 = W1 - self.lr * dW1
        # b1 = b1 - self.lr * db1
        # W2 = W2 - self.lr * dW2
        # b2 = b2 - self.lr * db2
        # W3 = W3 - self.lr * dW3
        # b3 = b3 - self.lr * db3

    def train(self):
        data = self.train_data
        labels = self.train_labels
        for epoch in range(self.epochs):
            utils.tic()
            total_loss = 0.0  # every epoch loss should start with zero
            good = 0.0
            total_size = 0.0
            # TODO: shuffle?
            data, labels = utils.shuffle(data, labels)
            for d, l in zip(data, labels):
                total_size += 1
                pred, cache = self.fprop(d)
                # check the prediction
                y_hat = np.argmax(pred)
                if y_hat == l:
                    good += 1

                err_cost = float(pred[int(l)])  # loss = -1 * log(err_cost)

                cross_entropy = utils.cross_entropy_loss(err_cost)
                if self.L2:
                    cross_entropy += utils.L2_cost(self.parameters["W"], self.L2)
                total_loss += cross_entropy

                grads = self.bprop(cache, d, l)
                self.weights_updates(grads)

            print('epoch {}:'.format(epoch + 1))
            acc = good * 100 / total_size
            train_acc.append(acc)
            avg_loss = total_loss / total_size
            train_loss.append(avg_loss)

            print('train accuracy: {:2.2f}%'.format(acc))
            print('train AVG loss: {:2.2f}'.format(avg_loss))

            self.validation_acc()
            print('time:')
            utils.toc()
            # end of epoch
        # cache all about model
        trained_model = {
            "norm": self.norm,
            "parameters": self.parameters,
            "lr": self.lr
        }
        directory = str(len(self.hidden)) + 'Hidden/L2/'
        np.save(directory + 'model_' + self.model_name, trained_model)
        self.printGraph(directory)

    def validation_acc(self):
        total = 0.0
        good = 0.0
        total_loss = 0.0

        for d, l in zip(self.val_data, self.val_labels):
            total += 1
            pred, cache = self.fprop(d)
            y_hat = np.argmax(pred)
            if y_hat == int(l):
                good += 1
            err_cost = float(pred[int(l)])
            cross_entropy = utils.cross_entropy_loss(err_cost)
            if self.L2:
                cross_entropy += utils.L2_cost(self.parameters["W"], self.L2)
            total_loss += cross_entropy

        acc = good * 100 / total
        val_acc.append(acc)
        avg_loss = total_loss / total
        val_loss.append(avg_loss)

        print('val acc {:2.2f}%'.format(good / total * 100))
        print('val AVG loss: {:2.2f}'.format(avg_loss))

    def printGraph(self, directory):
        x = range(1, len(train_acc) + 1)
        plt.plot(x, train_acc, 'r', label='Training Set')
        plt.plot(x, val_acc, 'b', label='Validation Set')
        plt.legend()
        plt.title('Training')
        plt.savefig(directory + self.model_name + '_acc.png')

        plt.clf()
        plt.plot(x, train_loss, 'r', label='Training Set')
        plt.plot(x, val_loss, 'b', label='Validation Set')
        plt.legend()
        plt.title('Training')
        plt.savefig(directory + self.model_name + '_loss.png')

    def addInputDropout(self):
        self.in_dropout = True
