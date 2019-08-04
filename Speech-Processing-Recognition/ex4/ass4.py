import numpy as np
import gcommand_loader as loader
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
import torch.nn.functional as F
import cer as CER
import matplotlib.pyplot as plt
import os
import string

# Global Parameters:
batchSize = 64
labelMaxSize = 6
lr = 1e-3
epochs = 80
alphabet = string.ascii_lowercase  # english alphabet TODO: some of alphabet is not in our scope: do 23 +1
L2I = {c: i for i, c in enumerate(alphabet, 1)}  # indexing letters to index
I2L = {i: c for i, c in enumerate(alphabet, 1)}  # index to letters

numFeatureMaps = 20
filterSize = 3

trainLoss = []
validateLoss = []
validationAcc = []

# loaders - code of Yossi Adi

# I guess google colab.. so make the right paths
if torch.cuda.is_available():
    print('we have gpu:)')
    device = "cuda"
    datasetTest = loader.GCommandLoader('drive/My Drive/Colab Notebooks/data/test')
    datasetTrain = loader.GCommandLoader('drive/My Drive/Colab Notebooks/data/train')
    datasetValid = loader.GCommandLoader('drive/My Drive/Colab Notebooks/data/valid')

else:
    device = "cpu"
    datasetTest = loader.GCommandLoader('data/test')
    datasetTrain = loader.GCommandLoader('data/train')
    datasetValid = loader.GCommandLoader('data/valid')

test_loader = DataLoader(datasetTest, batch_size=1, shuffle=False, num_workers=20,
                         pin_memory=True)
train_loader = DataLoader(datasetTrain, batch_size=batchSize, shuffle=True, num_workers=20,
                          pin_memory=True)
valid_loader = DataLoader(datasetValid, batch_size=batchSize, shuffle=False, num_workers=20,
                          pin_memory=True)


class ASR(nn.Module):
    global numFeatureMaps, filterSize

    def __init__(self):
        super(ASR, self).__init__()

        self.conv1 = nn.Conv2d(1, numFeatureMaps, kernel_size=filterSize)
        self.conv1_bn = nn.BatchNorm2d(numFeatureMaps)
        self.conv1_drop = nn.Dropout(p=0.3)

        self.conv2 = nn.Conv2d(numFeatureMaps, 1, kernel_size=filterSize)
        self.conv2_bn = nn.BatchNorm2d(1)
        self.conv2_drop = nn.Dropout(p=0.5)

        hidden_lstm = 300
        seq_size_lstm = round((161 - 2 * (filterSize - 1)) / 2)
        self.lstm = nn.LSTM(seq_size_lstm, 300, num_layers=4, bias=True, batch_first=False, dropout=0.,
                            bidirectional=True)
        self.fc0 = nn.Linear(2 * hidden_lstm, 150)
        self.fc1 = nn.Linear(150, len(alphabet) + 1)
        self.ctc_loss = nn.CTCLoss()

    def forward(self, x):
        # original 161x101

        x = self.conv1(x)  # batches x10x159x99
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)

        x = F.relu(x)

        x = self.conv2(x)  # batches x1x157x97
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)

        x = F.max_pool2d(x, kernel_size=2)  # batches x 1x78x48
        x = x.squeeze(1)  # back to 1 "filter" - so now we have: batches x features x timestamps [batches x 78x48]
        x = x.permute(2, 0, 1)  # lstm input shape (seq_len, batch, input_size), so lets permute!

        x, _ = self.lstm(x)  # 48x batches x 600 (2* hidden- 2 because of the bidirectional)
        x = x.permute(1, 0, 2)  # permute for FC NN, for each timestamp we have 600 features
        x = self.fc0(x)
        x = self.fc1(x)

        x = F.log_softmax(x, dim=2)
        x = x.permute(1, 0, 2)  # input to ctc
        return x


def index_label_to_seq_table(labels):
    """
    gets arg of labels indexes, each index is convert to class (from DataLoader), and then convert each class
    to chars, and by using L2I convert it to sequence of ints.
    :param labels:
    :return: sequence table, size table as torchInt
    """
    global labelMaxSize, device
    seq_table = np.zeros((len(labels), labelMaxSize))
    size_table = np.zeros((len(labels),))
    for i, label_idx in enumerate(labels):
        row_seq = np.zeros((1, labelMaxSize))  # initialized with zeros
        label_as_class = datasetTrain.classes[label_idx.item()]  # label_idx is tensor!-> convert to index
        size_table[i] = len(label_as_class)
        for j, c in enumerate(label_as_class):
            row_seq[0, j] = L2I[c]
        seq_table[i] = row_seq
    # TODO: check if intTensor is needed
    return torch.IntTensor(seq_table).to(device), torch.IntTensor(size_table).to(device)


def train(model, optimizer, epoch):
    global device, trainLoss
    model.train()
    count = 0
    total_loss: float = 0.0
    for batch_index, (data, labels, _) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # CTC needs 4 args: input, target, input_length, target_length
        # input: TxNxC s.t: T -Input sequence length, N -Batch size, C - #classes (in my case: 48x2x27)
        model_output = model(data)
        # target: NxS s.t: N - Batch size, S - max length of label (target)
        # target_length (N)
        # and target_length (length of each label in the target!)
        # in the function I already transfer to device!
        target, target_length = index_label_to_seq_table(labels)
        # input_length (as described at torch documentation) (N)
        N = model_output.size()[1]
        T = model_output.size()[0]
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long).to(device)

        # operate the ctc loss
        loss = model.ctc_loss(model_output, target, input_lengths, target_length)
        if epoch % 10 == 9:
            print("epoch{}#{} loss: {}".format(epoch, batch_index, str(loss.data.cpu().numpy())))
        # TODO: remove
        print("epoch{}#{} loss: {}".format(epoch, batch_index, str(loss.data.cpu().numpy())))

        total_loss += loss.data.cpu().numpy()  # get only the tensor data and convert it to numpy arrey.
        loss.backward()  # retain_graph=True)
        optimizer.step()
        count += 1
    total_loss /= count
    print(str(count))
    print('train avg loss: ' + str(total_loss))
    trainLoss.append(total_loss)


def reduced_prediction(words):
    """
    reduced repetitions and blanks
    :param words:
    :return:
    """
    reduced = []
    for word in words:
        last = w = ""
        for letter in word:
            if letter != "#" and last != letter:
                w += letter
            last = letter
        reduced.append(w)
    return reduced


def pred_word_from_seq(sequences):
    """
    with repetitions
    :param sequences:
    :return:
    """
    pred = []
    for seq in sequences:
        w = ''
        for timestamp in seq:
            index = torch.argmax(timestamp).item()
            if index == 0:
                # this blank
                w += '#'
            else:
                w += I2L[index]
        pred.append(w)

    return reduced_prediction(pred)


def label_idx_to_classes(labels):
    """
    return list of classes from list of label indexes
    :param labels:
    :return:
    """
    classes = []
    for label in labels:
        classes.append(datasetTrain.classes[label.item()])
    return classes


def validate(model, epoch):
    global validateLoss, labelMaxSize, device, validationAcc
    model.eval()
    count = 0
    validation_loss = 0.0
    validation_acc = 0.0
    with torch.no_grad():
        for batch_idx, (data, labels, _) in enumerate(valid_loader):
            data = data.to(device)
            labels = labels.to(device)

            output = model(data)
            pred = pred_word_from_seq(output.permute(1, 0, 2))
            # TODO: remove
            if epoch % 10 == 9:
                print(pred)
            target_classes = label_idx_to_classes(labels)
            accuracy = CER.cer(pred, target_classes)
            validation_acc += accuracy
            # again we want ctc loss!

            # target: NxS s.t: N - Batch size, S - max length of label (target)
            # target_length (N)
            # and target_length (length of each label in the target!)
            # in the function I already transfer to device!
            target, target_length = index_label_to_seq_table(labels)
            # input_length (as described at torch documentation) (N)
            N = output.size()[1]
            T = output.size()[0]
            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long).to(device)

            # operate the ctc loss
            loss = model.ctc_loss(output, target, input_lengths, target_length)

            validation_loss += loss.data.cpu().numpy()
            count += 1
        validation_loss /= count
        print('validation loss: ' + str(validation_loss))
        print("cer rate: " + str(validation_acc / count))

        validateLoss.append(validation_loss)
        validationAcc.append(float(validation_acc) / float(count) * 100)


def print_graphs(path):
    global epochs, validateLoss, trainLoss, validationAcc
    x = range(1, epochs + 1)
    plt.plot(x, trainLoss, 'r', label='train')
    plt.plot(x, validateLoss, 'b', label='validation')
    plt.legend()
    plt.title('Losses')
    plt.savefig(path + '/losses.png')

    plt.clf()
    plt.plot(x, validationAcc, 'r', label='validation')
    plt.legend()
    plt.title('accuracy')
    plt.savefig(path + '/accuracy.png')


def write_test(model, path):
    with torch.no_grad():
        model.eval()
        with open(path + '/test_y', 'w') as f:
            for data, _, path in test_loader:
                _, file_name = os.path.split(path[0])
                data = data.to(device)
                output = model(data)
                pred = pred_word_from_seq(output.permute(1, 0, 2))
                for w in pred:
                    f.write(file_name + ', ' + w + '\n')


def main():
    print('Start')

    model = ASR()
    model = model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    for epoch in range(0, epochs):
        print("start " + 'epoch number ' + str(epoch + 1))
        train(model, optimizer, epoch)
        validate(model, epoch)

    print('save model')
    if device == 'cpu':
        torch.save(model, "model.pt")
        print('print graphs')
        print_graphs("data")
        print('make file')
        write_test(model, "data")
    else:
        torch.save(model, "drive/My Drive/Colab Notebooks/ass4_outs/model.pt")
        print('print graphs')
        print_graphs("drive/My Drive/Colab Notebooks/ass4_outs")
        print('make file')
        write_test(model, "drive/My Drive/Colab Notebooks/ass4_outs")

    print('End')


if __name__ == '__main__':
    main()
