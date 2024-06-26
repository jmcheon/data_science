import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm 

def initialize_weights(X):
    return np.random.randn(X.shape[1], 1)

def initialize_bias():
    """create a column vector as a matrix"""
    return np.random.randn(1)

def forward(X, W, b):
    Z = np.dot(X, W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


def log_loss(A, y, eps=1e-15):
    A = np.clip(A, eps, 1 - eps)
    return - 1/len(y) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))


def gradients(X, A, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return dW, db


def update(dW, db, W, b, lr):
    W = W - lr * dW
    b = b - lr * db
    return W, b


def predict(X, W, b):
    A = forward(X, W, b)
    return (A >= 0.5).astype(int)


def nn(x_train, y_train, validation_data=None, lr=0.01, epochs=1000, plot_graph=False):
    W = initialize_weights(x_train)
    b = initialize_bias()

    history = {}
    history["loss"] = []
    history["accuracy"] = []

    for epoch in tqdm(range(epochs)):
        A = forward(x_train, W, b)
        dW, db = gradients(x_train, A, y_train)
        W, b = update(dW, db, W, b, lr)

        if epoch % 10 == 0:
            # trian loss
            history["loss"].append(log_loss(A, y_train))
            # accuracy
            y_pred = predict(x_train, W, b)
            history["accuracy"].append(accuracy_score(y_train, y_pred))

            if validation_data:
                x_val, y_val = validation_data
                if "val_loss" not in history:
                    history["val_loss"] = []
                if "val_accuracy" not in history:
                    history["val_accuracy"] = []
                # valid loss
                A_val = forward(x_val, W, b)
                history["val_loss"].append(log_loss(A_val, y_val))
                # valid accuracy
                y_pred = predict(x_val, W, b)
                history["val_accuracy"].append(accuracy_score(y_val, y_pred))
    if plot_graph:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history["loss"], label="train loss")
        plt.plot(history["val_loss"] , label="valid loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history["accuracy"], label="train acc")
        plt.plot(history["val_accuracy"], label="valid acc")
        plt.legend()
        plt.show()

    return W, b, history 

def plot_learning_curves(X, y, W, b, history):
    # Generate input data for decision boundary plot
    x1_range = np.linspace(-1.5, 1.5, 100)
    x2_range = np.linspace(-1.5, 1.5, 100)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

    # Calculate output for each input pair
    z_grid = np.array([[predict(np.array([[x1], [x2]]), W, b)[0, 0] for x1 in x1_range] for x2 in x2_range])

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history["loss"], label="train loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="valid loss")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(history["accuracy"], label="train accuracy")
    if "val_loss" in history:
        plt.plot(history["val_accuracy"], label="valid accuracy")
    plt.title("Accuracy")
    plt.legend()
    
    # Plotting decision boundary
    plt.subplot(1, 3, 3)
    plt.contourf(x1_grid, x2_grid, z_grid, levels=50, cmap='viridis', alpha=0.7)
    plt.colorbar()

    # Plot the dataset points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="summer", edgecolor='k')
    plt.title("Decision Boundary")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()