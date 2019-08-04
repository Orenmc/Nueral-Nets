import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import utils
import utils2


HIDDEN_LAYER_SIZE = 100
EMBEDDING_VAC_LEN = 50
WINDOW_SIZE = 5


class NET(nn.Module):
    """
    First model version.
    two hidden layers.
    activation function betweeb the layers: Relu.
    using batch normalization.
    """

    def __init__(self):
        super(NET, self).__init__()

        word_vectors = utils2.WORD_VECTOR
        output_len = len(utils.TAG_SET)
        self.E = nn.Embedding(word_vectors.shape[0], word_vectors.shape[1])  # Embedding matrix _ * 50
        self.E.weight.data.copy_(torch.from_numpy(word_vectors))
        self.input_size = WINDOW_SIZE * EMBEDDING_VAC_LEN   # 5 * 50 = 250
        self.fc0 = nn.Linear(self.input_size, HIDDEN_LAYER_SIZE)
        self.fc1 = nn.Linear(HIDDEN_LAYER_SIZE, output_len)

    def forward(self, x):
        """
        forward pass
        :param x: input data (size 250)
        :return: prediction (in precision - softmax)
        """
        x = self.E(x).view(-1, self.input_size)
        x = torch.tanh(self.fc0(x))
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)      # activation function
