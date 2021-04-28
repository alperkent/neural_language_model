import numpy as np
import matplotlib.pyplot as plt
from Network import Network
import pickle

# initializes network and hyperparameters
model = Network(alpha=0.01)
n_epoch = 66
batch_size = 256

# loads datasets
train_inputs = np.load('data/data/test_inputs.npy')
train_targets = np.load('data/data/test_targets.npy')
valid_inputs = model.onehot(np.load('data/data/valid_inputs.npy'))
valid_targets = np.load('data/data/valid_targets.npy')

# shuffles training data
np.random.seed(0)
train = np.concatenate((train_inputs, train_targets.reshape(-1, 1)), axis=1)
np.random.shuffle(train)

# divides training data into minibatches
n_batch = train.shape[0] // batch_size 
batches = []
for i in range(n_batch):
    batch = train[batch_size*i:batch_size*(i+1), :]
    batches.append(batch)

# initializes lists for training statistics
costs = []
tr_accs = []
val_accs = []

# runs the model
for epoch in range(n_epoch):
    for i, batch in enumerate(batches):
        X = model.onehot(batch[:, :-1])
        y = batch[:, -1].reshape((-1, 1))
        loss, pred = model.train(X, y)
        if i == len(batches)-1:
            costs.append(loss)
            tr_accs.append(model.accuracy(pred, y))
            val_accs.append(model.accuracy(model.evaluate(valid_inputs), valid_targets))

# outputs the model parameters
pickle.dump({'w': model.w, 'b': model.b}, open('model.pk', 'wb'))

# plots costs/accuracies over epoch
def plot(x, title):
    plt.plot(range(len(x)), x)
    plt.xlabel('Epoch')
    if title == 'Cost':
        plt.title('Cost per epoch')
        plt.ylabel('Cost')
    else:
        plt.title(title + ' accuracy per epoch')
        plt.ylabel('Accuracy')
    plt.show()

plot(costs, 'Cost')
plot(tr_accs, 'Training')
plot(val_accs, 'Validation')

print('Training accuracy:', tr_accs[-1])
print('Validation accuracy:', val_accs[-1])