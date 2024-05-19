from abc import abstractmethod
import numpy as np

def create_weight_matrix(nrows, ncols):
    # return np.zeros((nrows, ncols))
    return np.random.default_rng().normal(loc=0, scale=1/(nrows * ncols), size=(nrows, ncols))

def create_bias_vector(nrows):
    """create a column vector as a matrix"""
    # return np.zeros((nrows, 1))
    return create_weight_matrix(nrows, 1)


class NeuralNet:
    """
    A series of layers connected and compatible.
    """
    def __init__(self, layers, loss_function, lr) -> None:
        self._layers = layers
        self._loss_function = loss_function
        self.lr = lr
        self.check_layer_compatibility()

    def check_layer_compatibility(self):
        for from_, to_ in zip(self._layers[:-1], self._layers[1:]):
            print("from, to:", from_.ins, to_.ins)
            if from_.outs != to_.ins:
                raise ValueError("Layers should have compatible shapes.")

    def forward(self, x):
        # xs = [x]
        # for layer in self._layers:
        #     xs.append(layer.forward(xs[-1]))
        # return xs
        out = x
        for layer in self._layers:
            out = layer.forward(out)
        return out

    def loss(self, y_pred, y):
        return self._loss_function.loss(y_pred, y)

    def train(self, x, t):
        """
        Train the network on input x and expected output t.
        """
        # Accumulate intermediate results during forward pass.
        xs = [x]
        for layer in self._layers:
            xs.append(layer.forward(xs[-1]))

        x = xs.pop()
        # print("x in net train:", x)
        dx = self._loss_function.dloss(x, t)
        for layer, x in zip(self._layers[::-1], xs[::-1]):

            # Compute the derivatives
            y = np.dot(layer._W, x) + layer._b
            db = layer.act_function.df(y) * dx
            dx = np.dot(layer._W.T, db)
            dW = np.dot(db, x.T)

            # Update parameters
            layer._W -= self.lr * dW
            layer._b -= self.lr * db


class Activation:
    """
    Class to be inherited by activation functions.
    """
    @abstractmethod
    def f(self, x):
        """
        Method that implements the function.
        """
        pass

    @abstractmethod
    def df(self, x):
        """
        Derivative of the function with respect to its input.
        """
        pass

class Sigmoid(Activation):
    def f(self, x):
        return 1/(1 + np.exp(-x))

    def df(self, x):
        return self.f(x) * (1 - self.f(x))

class Softmax(Activation):
    def f(self, x):
        exp_x = np.exp(x - np.max(x))  # Subtract np.max(x) for numerical stability
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def df(self, x):
        # df here is the Jacobian matrix of softmax function
        s = self.f(x).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

class LeakyRelu(Activation):
    """
    Leaky Rectified Linear Unit.
    """
    def __init__(self, leaky_param=0.1):
        self.alpha = leaky_param

    def f(self, x):
        return np.maximum(x, x * self.alpha)

    def df(self, x):
        return np.maximum(x > 0, self.alpha)

class Loss:
    """
    Class to be inherited by loss functions.
    """
    @abstractmethod
    def loss(self, y_pred, y):
        """
        """
        pass

    @abstractmethod
    def dloss(self, y_pred, y):
        pass

class MSE(Loss):
    def loss(self, y_pred, y):
        return np.mean((y_pred - y) ** 2)

    def dloss(self, y_pred, y):
        return 2 * (y_pred - y) / y_pred.size

class CrossEntropyLoss(Loss):
    def loss(self, y_pred, y):
        # predicted should be a probability distribution (output of softmax)
        return -np.sum(y * np.log(y_pred))

    def dloss(self, y_pred, y):
        # Gradient of the loss with respect to the predicted values
        return y_pred - y 

class BinaryCrossEntropyLoss(Loss):
    """Cross entropy loss function following the pytorch docs."""
    def __init__(self, eps=1e-15):
        self.eps = eps
        
    def loss(self, y_pred, y):
        # Ensure predicted is a single probability value
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)

        return - (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def dloss(self, y_pred, y):
        # Gradient of the loss with respect to the predicted value
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        return - (y / y_pred) + (1 - y) / (1 - y_pred)

class Layer:
    """
    Layer class that represents the connections and the flow of information between a column of neurons and the next.
    It deals with what happens in between two columns of neurons instead of having the layer specifially represent the neurons of each vertical column
    """
    def __init__(self, ins, outs, act_function) -> None:
        self.ins = ins
        self.outs = outs
        self.act_function = act_function

        self._W = create_weight_matrix(self.outs, self.ins)
        self._b = create_bias_vector(self.outs)

    def forward(self, x):
        """
        helper method that computes the forward pass in the layer

        Parameters:
        x: a set of neuron states

        Returns:
        the next set of neuron states
        """
        return self.act_function.f(np.dot(self._W, x) + self._b)


def generic_train_demo():
    """
    Demo of a network as a serias of layers.
    """
    net = NeuralNet(
        [
            Layer(2, 4, LeakyRelu()),
            Layer(4, 4, LeakyRelu()),
            Layer(4, 3, LeakyRelu()),
        ],
        MSE(), 0.001
    )
    t = np.zeros(shape=(3, 1))
    print("t:\n", t)


    loss = 0
    for _ in range(100):
        x = np.random.normal(size=(2, 1))
        loss += net.loss(net.forward(x)[-1], t)
    print("loss:\n", loss)

    for _ in range(10000):
        x = np.random.normal(size=(2, 1))
        net.train(x, t)

    loss = 0
    for _ in range(100):
        x = np.random.normal(size=(2, 1))
        loss += net.loss(net.forward(x)[-1], t)
    print("loss:\n", loss)

# generic_train_demo()