import numpy as np
from Network import Network
import pickle

# loads learned model parameters
d = pickle.load(open('model.pk', 'rb'))
w = d['w']
b = d['b']

# initializes model with learned parameters
model = Network(w, b)

# loads datasets
test_inputs = model.onehot(np.load('data/data/train_inputs.npy'))
test_targets = np.load('data/data/train_targets.npy')

# evaluates the model and accuracy
pred = model.evaluate(test_inputs)
acc = model.accuracy(pred, test_targets)

print('Test accuracy:', acc)