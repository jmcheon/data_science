import numpy as np
from nn1 import NeuralNet, Layer
import matplotlib.pyplot as plt
import sys, os
import torch
import torch.nn as nn
import torch.optim as optim

path = os.path.join(os.path.dirname(__file__), '..', 'srcs')
sys.path.insert(1, path)

from activations import LeakyReLU, Sigmoid, Softmax
from losses import MSELoss, CrossEntropyLoss, BCELoss
from utils import load_topology, load_split_data, load_parameters

def load_data():
    TRAIN_FILE = "./datasets/WDBC/data_train.csv"
    TEST_FILE =  "./datasets/WDBC/data_test.csv"

    x_train, y_train = load_split_data(TRAIN_FILE)
    x_test, y_test = load_split_data(TEST_FILE)

    x_train, x_test = x_train / 255.0, x_test / 255.0

    print("x train shape:", x_train.shape, x_train.dtype)
    print("y train shape:", y_train.shape, y_train.dtype)
    print("x test shape:", x_test.shape, x_test.dtype)
    print("y test shape:", y_test.shape, y_test.dtype)

    train_data = np.hstack((y_train, x_train))
    print(train_data.shape)
    test_data = np.hstack((y_test, x_test))
    print(test_data.shape)
    return x_train, y_train, x_test, y_test

# test
def to_col(x):
    return x.reshape((x.size, 1))

def test(net, test_data):
    correct = 0
    for i, test_row in enumerate(test_data):

        y = test_row[0]
        x = to_col(test_row[1:])
        out = net.forward(x)
        y_pred = np.argmax(out)
        # if not i % 3000:
        #     print('pred:', y_pred, 'true:', y)
        if y == y_pred:
            correct += 1

    return correct/test_data.shape[0]

def create_net(shapes, activation_func, loss_func, lr):
    layers = [
        Layer(shapes[0], shapes[1], LeakyReLU()),
        Layer(shapes[1], shapes[2], LeakyReLU()),
        Layer(shapes[2], shapes[3], LeakyReLU()),
        Layer(shapes[3], shapes[4], activation_func),
    ]
    net = NeuralNet(layers, loss_func, lr)
    net.print_parameter_shape()
    return net

def plot_lc(train_loss):
    plt.plot(train_loss)
    plt.show()

class SimpleNN2(nn.Module):
    def __init__(self, shapes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.LeakyReLU(shapes[0], shapes[1])
        self.fc2 = nn.LeakyReLU(shapes[1], shapes[2])
        self.fc3 = nn.LeakyReLU(shapes[2], shapes[3])
        self.fc4 = nn.LeakyReLU(shapes[3], shapes[4])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

class SimpleNN(nn.Module):
    def __init__(self, shapes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(30, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 5)
        self.fc4 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

def norch(x, y, shapes, lr, epochs):
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    # Define the loss function
    criterion = nn.BCELoss()

    # Initialize the neural network
    net = SimpleNN(shapes)
    # print(net.parameters())

    # Define the optimizer
    optimizer = optim.SGD(net.parameters(), lr)

    optimizer.zero_grad()
    # Forward pass
    outputs = net(x)
    print(outputs.shape)
    # print(list(net.parameters()))
    # Training loop
    for epoch in range(epochs):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(x)

        # Compute the loss
        loss = criterion(outputs, y)

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        if epoch % 100 == 0:  # Print loss every 100 epochs
            print(f"Epoch {epoch}, Loss: {loss.item()}")

def morch(shapes, lr, epochs):
    # net = create_net(shapes, LeakyReLU(), MSELoss(), lr)
    net = create_net(shapes, LeakyReLU(), BCELoss(), lr)
    # net = create_net(shapes, LeakyReLU(), CrossEntropyLoss(), lr)

    # net = create_net(shapes, Sigmoid(), MSELoss(), lr)
    # net = create_net(shapes, Sigmoid(), BCELoss(), lr)
    # net = create_net(shapes, Sigmoid(), CrossEntropyLoss(), lr)

    # net = create_net(shapes, Softmax(), MSELoss(), lr)
    # net = create_net(shapes, Softmax(), BCELoss(), lr)
    # net = create_net(shapes, Softmax(), CrossEntropyLoss(), lr)

    train_loss = net.fit(x_train, y_train, epochs)
    y_pred = net.predict(x_test)

    plot_lc(train_loss)
    print(y_pred, y_test)


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()

    input_shape = x_train.shape[1]
    output_shape = 1#len(np.unique(y_train))

    x_train = x_train.T
    y_train = y_train.T
    x_test = x_test.T
    y_test = y_test.T

    print("input shape:", input_shape)
    print("output shape:", output_shape)

    shapes = [input_shape, 20, 10, 5, output_shape]
    lr = 0.01
    epochs = 100

    norch(x_train.T, y_train.T, shapes, lr, epochs)