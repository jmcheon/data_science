import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


import tensorflow as tf
from tensorflow.keras import layers, initializers

from multilayer_perceptron.ModelTrainer import ModelTrainer
from multilayer_perceptron.NeuralNet import NeuralNet
from multilayer_perceptron.srcs.utils import load_parameters, load_topology, load_split_data

import multilayer_perceptron.srcs.optimizers as optimizers
import multilayer_perceptron.srcs.losses as losses

n_inputs = 784

class TensorflowNN(tf.keras.Model):
    def __init__(self):
        super(TensorflowNN, self).__init__()
        self.fc1 = layers.Dense(20, activation='relu', kernel_initializer=initializers.GlorotUniform(), input_shape=(n_inputs,))
        self.fc2 = layers.Dense(16, activation='relu', kernel_initializer=initializers.GlorotUniform())
        self.fc3 = layers.Dense(10, activation='softmax', kernel_initializer=initializers.GlorotUniform())
    
    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class TorchNN(nn.Module):
    def __init__(self):
        super(TorchNN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 20)
        self.fc2 = nn.Linear(20, 16)
        self.fc3 = nn.Linear(16, 10)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, x):
        # x = x.float()

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


def create_torch_net():
    # pytorch network
    torch_net = TorchNN()
    # parameters: weights and biases
    parameters = load_parameters("./parameters/initial_mnist")
    
    for i in range(len(parameters)):
        if i % 2 == 0:
            parameters[i] = torch.tensor(parameters[i].T)
        else:
            parameters[i] = torch.tensor(parameters[i].reshape(parameters[i].shape[1],))
        # print(parameters[i].shape, torch_net_parmas[i].shape, tensorflow_net_params[i].shape)
        # print(parameters[i].dtype, torch_net_parmas[i].dtype, tensorflow_net_params[i].dtype)
    # print((parameters))
    
    with torch.no_grad():  # Disable gradient computation
        torch_net.fc1.weight.copy_(parameters[0])
        torch_net.fc1.bias.copy_(parameters[1])
        torch_net.fc2.weight.copy_(parameters[2])
        torch_net.fc2.bias.copy_(parameters[3])
        torch_net.fc3.weight.copy_(parameters[4])
        torch_net.fc3.bias.copy_(parameters[5])
    print("\nPytorch model creating...\n", torch_net)
    return torch_net


def train_torch_net(torch_net, x, y):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(torch_net.parameters(), lr=lr)

    dataset = TensorDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("\nPytorch model training...\n")
    for epoch in range(epochs):
        for x_batch, y_batch in data_loader:
            optimizer.zero_grad()
            outputs = torch_net(x_batch)
            # print(outputs.shape, outputs.dtype)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            print(f"Epoch {epoch + 1} - loss: {loss.item()}")

    # extract parameters from PyTorch model
    torch_parameters = []
    for param in torch_net.parameters():
        torch_parameters.append(param.detach().numpy())

    return torch_parameters


def create_tensorflow_net():
    # tensorflow network
    tensorflow_net = TensorflowNN()
    tensorflow_net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])
    dummy_input = tf.random.normal([1, n_inputs])
    tensorflow_net(dummy_input)
    tensorflow_net_params = tensorflow_net.get_weights()
    # print(len(tensorflow_net_params))
    parameters = load_parameters("./parameters/initial_mnist")
    for i in range(len(parameters)):
        if i % 2 != 0:
            parameters[i] = parameters[i].reshape(parameters[i].shape[1],)

    tensorflow_net.fc1.set_weights([parameters[0], parameters[1]])
    tensorflow_net.fc2.set_weights([parameters[2], parameters[3]])
    tensorflow_net.fc3.set_weights([parameters[4], parameters[5]])
    tensorflow_net.summary()
    print("\nTensorflow model creating...\n")
    return tensorflow_net

def train_tensorflow_net(tensorflow_net, x, y):
    tensorflow_parameters = tensorflow_net.get_weights()
    print("\nTensorflow model training...\n")
    tensorflow_net.fit(x, y, epochs=epochs, batch_size=batch_size)
    return tensorflow_parameters

def create_my_mlp_net():
    # my network
    topology = load_topology("./topologies/mnist.json")
    parameters = load_parameters("./parameters/initial_mnist")

    my_net = NeuralNet()
    my_net.set_topology(topology)
    my_net.set_parameters(parameters)
    print("\nMLP model creating...\n", my_net)
    return my_net

def train_my_mlp_net(my_net, x, y):
    print("\nMLP model training...\n")
    trainer = ModelTrainer()
    trainer.model_list.append(my_net)
    histories, model_names = trainer.train(trainer.model_list,
                  x,
                  y,
                  [optimizers.SGD(learning_rate=lr)],
                  losses.CrossEntropyLoss(),
                  metrics=['accuracy'],
                  batch_size=batch_size,
                  epochs=epochs
                  )
    y = my_net.one_hot_encode_labels(y)
    return y

def load_data():
    train_path = "./datasets/MNIST/data_train.csv"
    x, y = load_split_data(train_path)
    # print(x.shape, y.shape, x.dtype, y.dtype)

    return x, y

def compare_parameters(torch_parameters, tensorflow_parameters):
    # compare parameters
    same = True
    for tw, tfw in zip(torch_parameters, tensorflow_parameters):
        tw = tw.T
        # print(tw, tfw)
        print(tw.shape, tfw.shape)
        if not np.array_equal(tw, tfw):
            same = False
            break
    print(same)


if __name__ == "__main__":
    # hyperparameters 
    epochs = 2
    batch_size = 32
    lr = 1e-3
    print(f"epochs: {epochs}, batch size: {batch_size}, learning rate: {lr}")

    x, y = load_data()

    my_net = create_my_mlp_net()
    torch_net = create_torch_net()
    tensorflow_net = create_tensorflow_net()

    y = train_my_mlp_net(my_net, x, y)
    torch_parameters = train_torch_net(torch_net, torch.from_numpy(x), torch.from_numpy(y))
    tensorflow_parameters = train_tensorflow_net(tensorflow_net, x, y)