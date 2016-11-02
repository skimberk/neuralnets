import numpy as np
from nn import NeuralNet
import matplotlib.pyplot as plt

neuralnet = NeuralNet([2, 2, 1], sigma=0.3)

# Inputs
X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

# Outputs
y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

def plot_decision_boundary(X, y, pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

for x in range(10000):
    neuralnet.learn(X, y)

def predict(network, X):
    return (1 * (network.predict(X) > 0.5)).flatten()

print(neuralnet.predict(X))

plot_decision_boundary(X, y, lambda x: predict(neuralnet, x))
