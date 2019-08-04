import numpy as np
import NN
import utils
import sys
import time

# Globals:
false_array = ['false', 'False', '0', 0, 0.0, '0.0']


def main(args):
    print('Start: {}'.format(time.ctime()))

    argv = {}
    for arg in args:
        key, val = arg.split(':')
        argv[key] = val

    hidden = np.fromstring(argv["hidden"], dtype=int, sep=',')
    epochs = int(argv["epoch"])
    lr = float(argv['lr'])
    if argv['norm'] in false_array:
        norm = False
    else:
        norm = True
    if argv["l2"] in false_array:
        l2 = False
    else:
        l2 = float(argv["l2"])
    name = 'hidden:' + str(hidden) + '_e:' + str(epochs) + '_lr:' + str(lr) + '_norm:' + str(norm) + '_L2:' + str(l2)
    print(name)

    net = NN.NeuralNetwork(model_name=name, hidden=hidden, epochs=epochs, lr=lr, norm=norm, L2=l2)  # 2 hidden + l2
    net.load_data('data/train.csv', 'data/validate.csv', 'data/test.csv')
    # net.load_data('data/validate.csv', 'data/validate.csv', 'data/validate.csv')  # TODO: remove
    net.initialize_parameters()
    net.train()

    print('End: {}'.format(time.ctime()))


if __name__ == '__main__':
    main(sys.argv[1::])
