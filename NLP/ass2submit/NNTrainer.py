import torch.optim as optim
import torch
import utils
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

LEARN_RATE = 0.05
EPOCHS = 6
BATCH_SIZE = 1024

avg_validation_loss_per_epoch = {}
validation_accuracy_per_epoch = {}
avg_train_loss_per_epoch = {}


class ModelTrainer(object):

    def __init__(self, net, train_loader, validation_loader, test_loader, tagger_type):

        self.train_loader = train_loader
        self.dev_loader = validation_loader
        self.test_loader = test_loader
        self.net = net
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARN_RATE)
        self.tagger_type = tagger_type

    def run(self):
        global avg_validation_loss_per_epoch, validation_accuracy_per_epoch
        global avg_train_loss_per_epoch

        avg_validation_loss_per_epoch = {}
        validation_accuracy_per_epoch = {}
        avg_train_loss_per_epoch = {}
        for epoch in range(EPOCHS):
            print("epoch # %d" % int(epoch + 1))
            self.train(epoch)
            self.dev(epoch)
        # TODO REMOVE!!!
        print("end epochs")
        self.test()
        plot_graphs(self.tagger_type, avg_train_loss_per_epoch,
                    avg_validation_loss_per_epoch, validation_accuracy_per_epoch)

    def train(self, epoch):

        self.net.train()
        train_loss = 0.0
        correct = 0.0
        criterion = nn.CrossEntropyLoss()

        shuffled_train = DataLoader(self.train_loader, BATCH_SIZE, shuffle=True)
        for data, labels in shuffled_train:
            self.optimizer.zero_grad()
            output = self.net(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
            # negative log likelihood loss
            loss = criterion(output, labels)
            train_loss += loss.item()
            # calculating gradients
            loss.backward()
            # updating parameters
            self.optimizer.step()
        train_loss /= (len(shuffled_train))
        avg_train_loss_per_epoch[epoch] = train_loss
        print('\ntrain set: Average loss: {:.4f}\n'.format(train_loss))

    def dev(self, epoch):
        global avg_validation_loss_per_epoch, validation_accuracy_per_epoch
        self.net.eval()
        s = DataLoader(self.dev_loader, 1)
        validation_loss = 0.0
        correct = 0.0
        total = 0.0
        criterion = nn.CrossEntropyLoss()
        for data, target in s:
            output = self.net(data)
            validation_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            if self.tagger_type == "NER":
                if not (utils.INDEX_TO_TAG[pred.item()] == 'O'
                        and utils.INDEX_TO_TAG[target.item()] == 'O'):
                    correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                    total += 1
            else:
                total += 1
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        validation_loss /= len(s)
        accuracy = 100. * correct / total
        avg_validation_loss_per_epoch[epoch] = validation_loss
        validation_accuracy_per_epoch[epoch] = accuracy

        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(validation_loss, correct, total, accuracy))

    def test(self):
        s = DataLoader(self.test_loader, 1)

        self.net.eval()
        pred_list = []
        for data in s:
            t1 = torch.LongTensor(data)
            t2 = data
            output = self.net(t1)
            # get the predicted class out of output tensor
            pred = output.data.max(1, keepdim=True)[1]
            # add current prediction to predictions list
            tag = utils.INDEX_TO_TAG[pred.item()]
            pred_list.append(tag)

        write_test_results_file(self.tagger_type, pred_list)


def write_test_results_file(tagger_type, predictions_list):
    with open("data/" + tagger_type.lower() + "/test", 'r') as test_file, open("test1." + tagger_type.lower(), 'w') as output_file:
        content = test_file.readlines()
        i = 0
        for line in content:
            if line == '\n':
                output_file.write(line)
            else:
                output_file.write(predictions_list[i] + "\n")
                i += 1
        test_file.close()
        output_file.close()


def plot_graphs(tagger_type, train_loss, dev_loss, dev_accuracy):
    plt.clf()
    plt.plot(train_loss.keys(), train_loss.values(), label='train loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('avg train loss per epoch')
    plt.savefig(tagger_type + "1 train_loss.png")

    plt.clf()
    plt.plot(dev_loss.keys(), dev_loss.values(), label='dev loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('avg dev loss per epoch')
    plt.savefig(tagger_type + "1 dev_loss.png")

    plt.clf()
    plt.plot(dev_accuracy.keys(), dev_accuracy.values(), label='dev loss')
    plt.xlabel('epoch')
    plt.ylabel('percent')
    plt.title('dev accuracy per epoch')
    plt.savefig(tagger_type + "1 dev_acc.png")

