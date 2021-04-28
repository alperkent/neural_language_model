import numpy as np

class Network:
    
    def __init__(self, w=None, b=None, alpha=0.01):
        np.random.seed(0)
        if w != None:
            self.w = w
        else:
            self.w = [np.random.randn(250, 16),
                      np.random.randn(48, 128),
                      np.random.randn(128, 250)]
        if b != None:
            self.b = b
        else:
            self.b = [np.zeros((128)),
                      np.zeros((250))]
        self.alpha = alpha
        
    # forward propagates the model
    def forward(self, X, y):
        h1 = X[:, 0, :] @ self.w[0]
        h2 = X[:, 1, :] @ self.w[0]
        h3 = X[:, 2, :] @ self.w[0]
        h_all = np.concatenate((h1, h2 ,h3), axis=1)
        g = h_all @ self.w[1] + self.b[0]
        f = self.sigmoid(g)
        e = f @ self.w[2] + self.b[1]
        p = self.softmax(e)
        pred = np.argmax(p, axis=1).reshape((-1, 1))
        loss = self.xent(p, y)
        return h_all, g, f, p, loss, pred
    
    # backpropagates the model and updates parameters
    def backward(self, g, p, y):
        delta1 = self.delta_xent(p, y)
        delta2 = np.multiply((delta1 @ self.w[2].T), self.delta_sigmoid(g))
        delta3 = delta2 @ self.w[1].T
        return [delta1, delta2, delta3]
    
    # updates parameters
    def update(self, X, f, h_all, deltas):
        self.w[2] -= self.alpha * f.T @ deltas[0]
        self.b[1] -= self.alpha * deltas[0].sum(axis=0)
        
        self.w[1] -= self.alpha * h_all.T @ deltas[1]
        self.b[0] -= self.alpha * deltas[1].sum(axis=0)
        
        self.w[0] -= self.alpha * (X[:, 0, :].T @ deltas[2][:, :16] + X[:, 1, :].T @ deltas[2][:, 16:32] + X[:, 2, :].T @ deltas[2][:, 32:])
    
    # trains the model
    def train(self, X, y):
        h_all, g, f, p, loss, pred = self.forward(X, y)
        deltas = self.backward(g, p, y)
        self.update(X, f, h_all, deltas)
        return loss, pred
    
    # activates with sigmoid function
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
    # activates with softmax function
    def softmax(self, X):
        exp = np.exp(X - np.max(X))
        return exp / exp.sum(axis=1).reshape(-1, 1)
    
    # calculates cross-entropy
    def xent(self, p, y):
        y_hot = np.zeros((y.shape[0], 250))
        for i, truth in enumerate(y):
            y_hot[i][truth] = 1
        return -np.multiply(y_hot, np.log(p)).sum()

    # gets gradient of sigmoid
    def delta_sigmoid(self, X):
        a = self.sigmoid(X)
        return np.multiply(a, (1 - a))    

    # gets gradient of softmax and cross entropy
    def delta_xent(self, p, y):
        y_hot = np.zeros((y.shape[0], 250))
        for i, truth in enumerate(y):
            y_hot[i][truth] = 1
        return p - y_hot
    
    # turns data into one-hot representation
    def onehot(self, X):
        X_hot = []
        for i in range(X.shape[0]):
            X1 = np.zeros((250))
            X1[X[i][0]] = 1
            X2 = np.zeros((250))
            X2[X[i][1]] = 1
            X3 = np.zeros((250))
            X3[X[i][2]] = 1
            X_all = np.array([X1, X2, X3])
            X_hot.append(X_all)
        X_hot = np.array(X_hot)
        return X_hot
    
    # evaluates dataset
    def evaluate(self, X):
        h1 = X[:, 0, :] @ self.w[0]
        h2 = X[:, 1, :] @ self.w[0]
        h3 = X[:, 2, :] @ self.w[0]
        h_all = np.concatenate((h1, h2 ,h3), axis=1)
        g = h_all @ self.w[1] + self.b[0]
        f = self.sigmoid(g)
        e = f @ self.w[2] + self.b[1]
        p = self.softmax(e)
        pred = np.argmax(p, axis=1).reshape((-1, 1))
        return pred
    
    # calculates accuracy
    def accuracy(self, pred, y):
        acc = 0
        for i in range(y.shape[0]):
            if pred[i] == y[i]:
                acc += 1
        return acc/y.shape[0]