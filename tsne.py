import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from Network import Network
import pickle

# loads learned model parameter
d = pickle.load(open('model.pk', 'rb'))
w = d['w']
b = d['b']

# initializes model with learned parameters
model = Network(w, b)

# loads vocab data
vocab = np.load('data/data/vocab.npy')

# transforms one-hot representations into learned embeddings
embedded = np.identity(250) @ w[0]

# creates a TSNE model and trains it with embeddings
tsne_model = TSNE(random_state=0, perplexity=30, n_iter=1000)
tsne_data = tsne_model.fit_transform(embedded)

# creates a simple word plot
def word_plot(data=tsne_data, words=vocab):
    plt.scatter(data[:, 0], data[:, 1], 0)
    for i, text in enumerate(words):
        plt.annotate(text, (data[i, 0], data[i, 1]))
    plt.show()

# creates an interactive plot
def interactive_plot(data=tsne_data, words=vocab):
    fig,ax = plt.subplots()
    sc = plt.scatter(data[:, 0], data[:, 1])
    
    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    
    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
                               " ".join([words[n] for n in ind["ind"]]))
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)
    
    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()

word_plot()
interactive_plot()

# converts words to one-hot representations
def word_to_onehot(a, b, c, X=vocab):
    phrase = [np.where(X == a)[0][0], np.where(X == b)[0][0], np.where(X == c)[0][0]]
    onehot = np.zeros((1, 3, 250))
    onehot[0][0][phrase[0]] = 1
    onehot[0][1][phrase[1]] = 1
    onehot[0][2][phrase[2]] = 1
    return onehot

# predicts phrases
def predictor(s1, s2, s3, X=vocab, m=model):
    phrase = word_to_onehot(s1, s2, s3)
    pred = str(X[m.evaluate(phrase)[0][0]])
    print(s1, s2, s3, pred)

predictor('city', 'of', 'new')
predictor('life', 'in', 'the')
predictor('he', 'is', 'the')

# creates n-long strings by predicting the next word from initial three words
def word_guesser(s1, s2, s3, n, X=vocab):
    string = s1 + ' ' + s2 + ' ' + s3
    for i in range(n-3):
        phrase = word_to_onehot(s1, s2, s3)
        pred = str(vocab[model.evaluate(phrase)[0][0]])
        string += " " + pred
        s1 = s2
        s2 = s3
        s3 = pred
    print(string)

word_guesser('where', 'are', 'you', 10)
word_guesser('the', 'city', 'of', 10)
word_guesser('he', 'will', 'go', 10)
word_guesser('i', 'would', 'like', 10)