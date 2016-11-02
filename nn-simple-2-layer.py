import numpy as np

# Hyperbolic tan function
def tanh(x):
    output = np.tanh(x)
    return output

# Derivative of hyperbolic tan given output of tanh
def tanh_output_to_derivative(output):
    return 1 - np.power(output, 2)

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

# Seed random number generator
np.random.seed(0)

# Generate random weights distributed along normal curve
weights_0 = np.random.randn(2, 3)
weights_1 = np.random.randn(3, 1)

# Learn
for i in range(1000):
    layer_0 = X
    layer_1 = tanh(np.dot(layer_0, weights_0))
    layer_2 = tanh(np.dot(layer_1, weights_1))

    layer_2_error = layer_2 - y
    layer_2_delta = layer_2_error * tanh_output_to_derivative(layer_2)

    layer_1_error = np.dot(layer_2_delta, weights_1.T)
    layer_1_delta = layer_1_error * tanh_output_to_derivative(layer_1)

    weights_1 -= 0.1 * np.dot(layer_1.T, layer_2_delta)
    weights_0 -= 0.1 * np.dot(layer_0.T, layer_1_delta)

print('Outputs after training:')
print(layer_2)
