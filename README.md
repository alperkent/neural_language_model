# CMPE597 Project 1 - Neural language model using a multi-layer perceptron

# Description:
In this project, I created a neural language model that receives 3 consecutive words as the input and aims to predict the next word as output. The model uses a multi-layer perceptron that is implemented from scratch (only using Numpy functions) and it is trained using cross-entropy loss. 

The network consists of a 16 dimensional embedding layer, a 128 dimensional hidden layer and one output layer. The input consists of a sequence of 3 consecutive words, provided as integer valued indices representing a word in a 250-word dictionary. Each word is converted to a one-hot representation and fed to the embedding layer which is 250x16 dimensional. Hidden layer has sigmoid loss function and the output layer is a softmax over the 250 words in the dictionary. After the embedding layer, 3 word embeddings are concatenated and fed to the hidden layer.

![Network](https://user-images.githubusercontent.com/76096096/209042354-b58ecb11-8bc3-485f-9735-0046d8847902.png)

This is a Python 3 implementation using Numpy for necessary operations, Matplotlib for visualization, Pickle to save and load learned model parameters and scikit-learn for t-SNE functions.

# Files:
Network.py file implements forward and backward propagation from scratch with activations and other necessary functions.

main.py file loads the dataset, shuffles the training data, divides it into mini batches, trains the model and evaluates it on validation set during training, saving the model parameters to model.pk file and outputting cost, training accuracy, validation accuracy plots and final accuracy values.

eval.py file loads the learned model parameters and evaluates the model on test data, outputting test accuracy.

tsne.py file loads the learned model parameters, calculates 16 dimensional word embeddings, fits and plots t-SNE for the embeddings, outputting a word plot and an interactive plot with several word predictions from the model.

model.pk file contains learned model parameters in Pickle format.

# Instructions:
You can run the model after extracting the data files to "\data\data" folders under the same directory with model files. After this step, typing these commands in IPython should return the outputs:

In [1]: %cd "CURRENT DIRECTORY"

In [2]: %run main.py

In [3]: %run eval.py

In [4]: %run tsne.py

# Results:
**Hyperparameters:**
- Batch size = 256
- Learning rate = 0.01
- Epochs = 66

![Cost per epoch](/images/costs.png)

![Training accuracy per epoch](/images/training_acc.png)

Training accuracy: 43.36%

![Validation accuracy per epoch](/images/val_acc.png)

Validation accuracy: 31.14%

Test accuracy: 31.07%

2-D t-SNE word plot:

![2-D t-SNE word plot](/images/word_plot.png)

Clusters of related words:

1. should, could, can, would, will, may, might
2. my, your, his, their, our
3. have, has, had
4. been, be, was, were, is, are, ‘s
5. do, does, did
6. say, says, said
7. make, made
8. it, that, this
9. me, him, us, them
10. i, you, he, she, we, they
11. dr., ms.
12. now, then
13. where, when, department
14. what, who
15. because, and, if, but, or, ‘,’, ‘:’, ‘;’, ‘—', percent
16. few, more, much
17. since, street
18. program, so
19. five, officials
20. state, York
21. like, see

Looking at these examples, words in the same cluster usually have the same function in a sentence. Thus, they can stand in for one another in a sentence. Most of the clusters are from the same lexical category (pronouns, determiners, conjunctions, auxiliary verbs, etc.) Also, in some clusters, there seems to be some commonality with meaning too. However, words in some clusters (17, 18, 19) did not make a lot of sense.
I also prepared an interactive plot (by modifying the code from reference 5) that makes it a lot easier to examine the clusters. The interactive plot runs with a GUI, thus I cannot attach it here. However, you can run the tsne.py file in IPython to create the plot.

Some predictions:

- city of new → york
- life in the → world
- he is the → same

These predictions looked sensible. After seeing such predictions, I was curious if the model could achieve more. So, I wrote a function that creates n-word-long strings by continuously predicting the next word from preceding three words, just to see how well the network performs. But after a dozen trials, I realized that it can very seldom create meaningful phrases longer than 5-6 words, usually going into a loop of dots afterwards. Here are some examples:

1. where are you going to do ? of them .
2. the city of the country . . . the next
3. he will go to the way that i can do
4. i would like to think about it . those .

# Mathematical expressions:

**Forward propagation:**

![Forward propagation](https://user-images.githubusercontent.com/76096096/209042772-89267002-e186-4d72-a852-1783531189aa.png)

**Backward propagation:**

![Backward propagation](https://user-images.githubusercontent.com/76096096/209042811-9d66cd2b-8a33-4021-b066-1f8e9226a0cd.png)

# References:
1.	Neural networks from scratch in Python. (2021, April 27). Christian Dima. https://www.cristiandima.com/neural-networks-from-scratch-in-python/
2.	Clark, K. (2021, April 27). Computing Neural Network Gradients. Stanford CS224 Readings. https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf
3.	Derivative of Softmax loss function. (2021, April 27). Mathematics Stack Exchange. https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function/
4.	Code example of t-SNE: Dimensionality reduction Lecture 27@ Applied AI Course. (2021, April 27). YouTube. https://www.youtube.com/watch?v=eDGWcIt10d8
5.	Possible to make labels appear when hovering over a point in matplotlib?. (2021, April 27). Stack Overflow. https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib
