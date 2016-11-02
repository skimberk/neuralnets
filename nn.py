import numpy as np

def tanh(x):
    output = np.tanh(x)
    return output

def tanh_output_to_derivative(output):
    return 1 - np.power(output, 2)

class NeuralNet:
    def __init__(self, dimensions, sigma=0.1):
        np.random.seed(0)

        self.dimensions = dimensions

        self.epsilon = 0.01
        self.reg_lambda = 0.01

        self.weights = []
        self.biases = []
        self.sigma = sigma

        for index, current in enumerate(dimensions[1:]):
            previous = dimensions[index]

            rows = previous
            cols = current

            self.weights.append(np.random.randn(rows, cols) / np.sqrt(rows))
            self.biases.append(np.zeros((1, current)))

    def _forward(self, X):
        if X.shape[1] != self.dimensions[0]:
            raise ValueError('X has dimensions different from those of neural network')

        layers = [X]

        for weights, biases in zip(self.weights, self.biases):
            layer = tanh(np.dot(layers[-1], weights) + biases)
            layers.append(layer)

        return layers

    def predict(self, X):
        return self._forward(X)[-1]

    def learn(self, X, y):
        layers = self._forward(X)

        last_error = layers[-1] - y
        last_delta = last_error * tanh_output_to_derivative(layers[-1])

        deltas = [last_delta]

        info = list(zip(layers[1:-1], self.weights[1:]))[::-1]

        for layer, weights in info:
            error = np.dot(deltas[-1], weights.T)
            delta = error * tanh_output_to_derivative(layer)
            deltas.append(delta)

        deltas.reverse()

        for layer, delta, weights, biases in zip(layers[:-1], deltas, self.weights, self.biases):
            weights -= self.sigma * np.dot(layer.T, delta)
            biases -= self.sigma * np.sum(delta)
