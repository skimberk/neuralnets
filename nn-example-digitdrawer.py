import numpy as np
from digitdrawer import DigitDrawer
from nn import NeuralNet
import pickle

nn = pickle.load(open('digits-nn.pickle', 'rb'))

def drawing_update(image):
    reformatted = (image / np.max(image)).reshape(-1, 64)
    # print(np.argmax(nn.predict(reformatted), axis=1))
    predicted = nn.predict(reformatted)
    ordered = predicted.argsort(axis=1)[0][::-1]

    print('Most likely a', ordered[0])
    print('But it could also be a', ordered[1])

dd = DigitDrawer(drawing_update)
