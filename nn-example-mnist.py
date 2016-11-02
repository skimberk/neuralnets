import numpy as np
from sklearn import datasets
from nn import NeuralNet
import pickle

digits = datasets.load_digits()

images = (digits.images / np.max(digits.images)).reshape(-1, 64)

targets = np.zeros((digits.target.size, 10))

for i in range(digits.target.size):
    targets[i][digits.target[i]] = 1

# nn = NeuralNet([64, 300, 10], 0.00006)

# for x in range(1000):
#     nn.learn(images, targets)

# pickle.dump(nn, open('digits-nn.pickle', 'wb'))

nn = pickle.load(open('digits-nn.pickle', 'rb'))

prediction = nn.predict(images)

failures = 0

for actual, guessed in zip(digits.target, np.argmax(prediction, axis=1)):
    if actual != guessed:
        failures += 1

print(1 - failures / images.shape[0])
