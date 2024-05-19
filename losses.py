from abc import abstractmethod
import numpy as np

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


class BinaryCrossEntropyLoss:
    def loss(self, predicted, target):
        return -np.sum(target * np.log(predicted) + (1 - target) * np.log(1 - predicted))