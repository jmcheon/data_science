import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import tensorflow as tf
from tensorflow.keras import layers, initializers

import sys, os
path = os.path.join(os.path.dirname(__file__), '', 'multilayer_perceptron')
print(path)

sys.path.insert(1, path)

from ModelTrainer import ModelTrainer
import srcs.optimizers as optimizers
import srcs.losses as losses
from srcs.utils import load_parameters, load_topology, load_split_data

class TensorflowNN(tf.keras.Model):
    def __init__(self):
        super(TensorflowNN, self).__init__()
        self.fc1 = layers.Dense(20, activation='relu', kernel_initializer=initializers.GlorotUniform(), input_shape=(30,))
        self.fc2 = layers.Dense(10, activation='relu', kernel_initializer=initializers.GlorotUniform())
        self.fc3 = layers.Dense(5, activation='relu', kernel_initializer=initializers.GlorotUniform())
        self.fc4 = layers.Dense(1, activation='sigmoid', kernel_initializer=initializers.GlorotUniform())
    
    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class TorchNN(nn.Module):
    def __init__(self):
        super(TorchNN, self).__init__()
        self.fc1 = nn.Linear(30, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 5)
        self.fc4 = nn.Linear(5, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
    
    def forward(self, x):
        # x = x.float()

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x


def pytorch_network(x, y):
    # pytorch network
    torch_net = TorchNN()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(torch_net.parameters(), lr=lr)
    # parameters: weights and biases
    parameters = load_parameters("./parameters/initial_parameters")
    
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
        torch_net.fc4.weight.copy_(parameters[6])
        torch_net.fc4.bias.copy_(parameters[7])
    
    print("\nPytorch model training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = torch_net(x)
        # print(outputs.shape, outputs.dtype)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if epoch % 1 == 0:
            print(f"Epoch {epoch + 1} - loss: {loss.item()}")

    # extract parameters from PyTorch model
    torch_parameters = []
    for param in torch_net.parameters():
        torch_parameters.append(param.detach().numpy())

    return torch_parameters


def tensorflow_network(x, y):
    # tensorflow network
    tensorflow_net = TensorflowNN()
    tensorflow_net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=['accuracy'])
    dummy_input = tf.random.normal([1, 30])
    tensorflow_net(dummy_input)
    tensorflow_net_params = tensorflow_net.get_weights()
    # print(len(tensorflow_net_params))
    parameters = load_parameters("./parameters/initial_parameters")
    for i in range(len(parameters)):
        if i % 2 != 0:
            parameters[i] = parameters[i].reshape(parameters[i].shape[1],)

    tensorflow_net.fc1.set_weights([parameters[0], parameters[1]])
    tensorflow_net.fc2.set_weights([parameters[2], parameters[3]])
    tensorflow_net.fc3.set_weights([parameters[4], parameters[5]])
    tensorflow_net.fc4.set_weights([parameters[6], parameters[7]])


    tensorflow_parameters = tensorflow_net.get_weights()

    print("\nTensorflow model training...")
    tensorflow_net.fit(x, y, epochs=epochs)
    return tensorflow_parameters

def my_mlp_network(x, y):
    # my network
    topology = load_topology("./topologies/binary_sigmoid.json")
    parameters = load_parameters("./parameters/initial_parameters")

    trainer = ModelTrainer()
    my_net = trainer.create(topology)
    my_net.set_parameters(parameters)
    print("\nMLP model training...")
    histories, model_names = trainer.train(trainer.model_list,
                  x,
                  y,
                  [optimizers.SGD(learning_rate=lr)],
                  losses.BCELoss(),
                  metrics=['accuracy'],
                  batch_size=None,
                  epochs=epochs
                  )

def load_data():
    train_path = "./datasets/WDBC/data_train.csv"
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
    epochs = 10
    lr = 1e-3

    x, y = load_data()

    my_mlp_network(x, y)
    pytorch_network(torch.from_numpy(x), torch.from_numpy(y))
    tensorflow_network(x, y)