# CMPE597 Project 1 - Neural language model using a multi-layer perceptron

# Description:
This is a Python 3 implementation using Numpy for necessary operations, Matplotlib for visualization, Pickle to save and load learned model parameters and Sklearn for t-SNE functions.

# Files:
Network.py file implements forward and backward propagation with activations and other necessary functions.

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
