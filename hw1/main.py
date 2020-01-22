import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import tqdm

from nn import Network
from nn import layers
from nn.layers import losses
from nn.optimizers import SGDOptimizer, MomentumSGDOptimizer


class MNISTNetwork(Network):
    def __init__(self):
        self.network = layers.SequentialLayer(
            [
                layers.LinearLayer(28 * 28, 1000),
                layers.PReLULayer(1000),
                layers.LinearLayer(1000, 100),
                layers.PReLULayer(100),
                layers.LinearLayer(100, 10)
            ]
        )
        loss_layer = losses.SoftmaxCrossEntropyLossLayer(parent=self.network)
        super(MNISTNetwork, self).__init__(loss_layer)

    def forward(self, data):
        return self.network(data)

    def loss(self, predictions, labels):
        return self.loss_layer(predictions, labels)


def train(train_data, train_labels, test_data, test_labels, optimizer_type="sgd"):
    network = MNISTNetwork()
    print(network)
    if optimizer_type == "sgd":
        optimizer = SGDOptimizer(network.parameters(), lr)
    elif optimizer_type == "momentum":
        optimizer = MomentumSGDOptimizer(network.parameters(), lr, weight_decay=0.0005)
    else:
        raise NotImplementedError("This optimizer does not exist.")

    iteration = -1
    print("-" * 50)
    output = network(test_data)
    accuracy = (np.argmax(output, 1) == test_labels).mean()
    loss = network.loss(output, test_labels)
    print("initial test accuracy %.3f" % accuracy, "loss %.3f" % loss)
    print("-" * 50)

    for epoch in range(30):
        train_accuracy = 0
        train_loss = 0
        train_iters = 0
        for ii in tqdm.tqdm(range(0, len(train_data), batch_size)):
            train_iters += 1
            iteration += 1
            data = train_data[ii : min(ii + batch_size, len(train_data))]
            labels = train_labels[ii : min(ii + batch_size, len(train_data))]
            optimizer.zero_grad()
            output = network(data)
            train_accuracy += (np.argmax(output, 1) == labels).mean()
            train_loss += network.loss(output, labels)
            network.backward()
            optimizer.step()
        train_accuracy /= train_iters
        train_loss /= train_iters
        print("train accuracy %.3f" % train_accuracy, "loss %.3f" % train_loss)
        output = network(test_data)
        accuracy = (np.argmax(output, 1) == test_labels).mean()
        loss = network.loss(output, test_labels)
        print("epoch", (epoch + 1), "accuracy %.3f" % accuracy, "loss %.3f" % loss)
        print("-" * 50)
    print("done")


if __name__ == "__main__":
    batch_size = 1000
    lr = 0.01

    train_dataset = np.load("../datasets/mnist/train.npz")
    train_data = train_dataset["data"].astype(np.float32) / 255
    train_data = train_data.reshape(-1, 28 ** 2)
    train_labels = train_dataset["labels"]

    test_dataset = np.load("../datasets/mnist/test.npz")
    test_data = test_dataset["data"].astype(np.float32) / 255
    test_data = test_data.reshape(-1, 28 ** 2)
    test_labels = test_dataset["labels"]

    train(train_data, train_labels, test_data, test_labels, "momentum")

