import numpy as np

def initialize_weights(outs, ins):
    limit = np.sqrt(6 / (outs + ins))
    return np.random.uniform(-limit, limit, (outs, ins))
    return np.random.default_rng().normal(loc=0, scale=1/(outs * ins), size=(outs, ins))
    # return np.zeros((outs, ins))

def initialize_bias(outs):
    """create a column vector as a matrix"""
    # return np.zeros((outs, 1))
    return initialize_weights(outs, 1)

class Layer:
    """
    Layer class that represents the connections and the flow of information between a column of neurons and the next.
    It deals with what happens in between two columns of neurons instead of having the layer specifially represent the neurons of each vertical column
    """
    def __init__(self, ins, outs, act_function) -> None:
        self.ins = ins
        self.outs = outs
        self.a = None
        self.dz = None
        self.act_function = act_function

        self._W = initialize_weights(self.outs, self.ins)
        self._b = initialize_bias(self.outs)

    def forward(self, x):
        """
        helper method that computes the forward pass in the layer

        Parameters:
        x: a set of neuron states

        Returns:
        the next set of neuron states
        """
        a = self.act_function.f(np.dot(self._W, x) + self._b)
        self.a = a
        return  a

class NeuralNet:
    """
    A series of layers connected and compatible.
    """
    def __init__(self, layers, loss_function, lr) -> None:
        self._layers = layers
        self._loss_function = loss_function
        self.lr = lr
        self.activations = []
        self.check_layer_compatibility()

    def check_layer_compatibility(self):
        for from_, to_ in zip(self._layers[:-1], self._layers[1:]):
            print("from, to:", from_.ins, to_.ins)
            if from_.outs != to_.ins:
                raise ValueError("Layers should have compatible shapes.")
    
    def print_parameter_shape(self):
        for i, layer in enumerate(self._layers):
            print(f"W{i} ", layer._W.shape)
            print(f"b{i} ", layer._b.shape)

    def forward(self, x):
        # xs = [x]
        # for layer in self._layers:
        #     xs.append(layer.forward(xs[-1]))
        # return xs
        out = x
        self.activations.append(x)
        n_layers = len(self._layers)
        for layer in self._layers:
        # for l in range(n_layers):
            out = layer.forward(out)
            # out = self._layers[l].forward(self.activations[l - 1])
            self.activations.append(out)
        return out

    def predict(self, x):
        a = self.forward(x)
        print("a.shape: ", a, a.shape)
        return (a >= 0.5).astype(int)
        # return np.argmax(a, axis=0)

    def loss(self, y, y_pred):
        return self._loss_function.loss(y, y_pred)

    def backward(self, y):
        # dz = self._layers[-1].a - y
        dz = self.activations[-1] - y

        # for i in range(len(self.activations)):
            # print(f"A{i}.shape: ", self.activations[i].shape)
        m = y.shape[1]
        n_layers = len(self._layers)
        # print("n_layers:", n_layers)
        # for i, layer in enumerate(reversed(self._layers)):
        for l in reversed(range(0, n_layers)):
            # layer_num = n_layers - i - 2
            # print("layer num:", l)
            # a = self._layers[l - 1].a
            a = self.activations[l]
            # print(f"{l}th a.shape: ", a.shape)

            # Compute the derivatives
            # print(i)
            dW = 1 / m * np.dot(dz, a.T)
            db = 1 / m * np.sum(dz, axis=1, keepdims=True)
            if l > 0: 
                dz = np.dot(self._layers[l]._W.T, dz) * a * (1 - a)

            # print("dW.shape: ", dW.shape)
            # print("W.shape: ", self._layers[l]._W.shape)
            # Update parameters
            self._layers[l]._W -= self.lr * dW
            self._layers[l]._b -= self.lr * db

    def update(self):
        pass

    def backward2(self, a, y):
        dz = a.pop() - y
        m = y.shape[1]
        n_layers = len(self._layers)
        # print("m examples:", m)
        for i, (layer, a) in enumerate(zip(self._layers[::-1], a[::-1])):

            # Compute the derivatives
            # print(i)
            dW = 1 / m * np.dot(dz, a.T)
            db = 1 / m * np.sum(dz, axis=1, keepdims=True)
            if i < n_layers: 
                dz = np.dot(layer._W.T, dz) * a * (1 - a)

            # Update parameters
            layer._W -= self.lr * dW
            layer._b -= self.lr * db

    
    def fit(self, x_train, y_train, epochs=10):
        # x_train = train_data[:30000, 1:].T
        # y_train = train_data[:30000, 0:1].T

        # print(train_data.shape)
        print("x_train.shape", x_train.shape)
        print("y_train.shape", y_train.shape)
        train_loss = []

        for epoch in range(epochs):
            # self.fit(x_train, y_train)
            self.forward(x_train)
            self.backward(y_train)
            if epoch % 100 == 0:
                print(epoch)
            train_loss.append(self._loss_function.compute_loss(self.forward(x_train), y_train))
            # y_pred = net.forward(x_train)#.reshape(1, -1)
            # print(y_pred, y_pred.shape)
            # print(y_pred.shape, y_train.flatten().shape)
            # print(y_train, y_train.shape)
            # train_loss.append(loss.loss(y_pred, y_train))

        '''
        for i, train_row in enumerate(train_data):
            # if not i%1000:
            #     print(i)
            x = to_col(train_row[1:])
            y = np.array(train_row[0], ndmin=2)
            # if not i%10000:
            #     print('y:', y)
            # print(y, y.shape)
            net.train(x, y)
            if i % 10 == 0:
                train_loss.append(loss(net.forward(x), y))
        '''

        return train_loss

    def fit2(self, x, y):
        """
        Train the network on input x and expected output y.
        """
        # activations during forward pass
        a = [x]
        for layer in self._layers:
            a.append(layer.forward(a[-1]))

        # backpropagation
        self.backward2(a, y)