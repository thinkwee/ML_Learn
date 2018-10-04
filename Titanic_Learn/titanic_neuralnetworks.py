# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 12:42:25 2017

@author: thinkwee
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


class Config:
    nn_input_dim = 4  # input layer dimensionality
    nn_output_dim = 2  # output layer dimensionality
    # Gradient descent parameters (I picked these by hand)
    epsilon = 0.2  # learning rate for gradient descent
    reg_lambda = 0.05  # regularization strength


def PCA(trainX, trainy):
    estimator = PCA(n_components=2)
    X_pca = estimator.fit_transform(trainX)
    colors = {'red', 'blue'}
    for i in range(len(colors)):
        px = X_pca[:, 0][trainy == i]
        py = X_pca[:, 1][trainy == i]
        plt.scatter(px, py, colors[i])
    plt.legend(np.arange(0, 10).astype(str))
    plt.show()


def generate_data():
    vali = pd.read_csv(r"E:\Machine Learning\MLData\Titanic Machine Learning from Disaster\validation.csv")
    train = pd.read_csv(r"E:\Machine Learning\MLData\Titanic Machine Learning from Disaster\train.csv")
    train = train.dropna(subset=['Age', 'Embarked'], axis=0)
    vali = vali.dropna(subset=(['Age', 'Embarked']), axis=0)

    train.loc[train["Sex"] == "male", "Sex"] = 0
    train.loc[train["Sex"] == "female", "Sex"] = 1
    train.loc[train["Embarked"] == "S", "Embarked"] = 0
    train.loc[train["Embarked"] == "C", "Embarked"] = 1
    train.loc[train["Embarked"] == "Q", "Embarked"] = 2
    trainx = train.reindex(index=train.index[:], columns=['Age'] + ['Sex'] + ['Fare'] + ['Embarked'])

    vali.loc[vali["Sex"] == "male", "Sex"] = 0
    vali.loc[vali["Sex"] == "female", "Sex"] = 1
    vali.loc[vali["Embarked"] == "S", "Embarked"] = 0
    vali.loc[vali["Embarked"] == "C", "Embarked"] = 1
    vali.loc[vali["Embarked"] == "Q", "Embarked"] = 2
    valix = vali.reindex(index=vali.index[:], columns=['Age'] + ['Sex'] + ['Fare'] + ['Embarked'])

    valiy = vali.reindex(index=vali.index[:], columns=['Survived'])

    trainy = train.reindex(index=train.index[:], columns=['Survived'])

    return np.array(trainx).astype(int), np.array(trainy).ravel().astype(int), np.array(valix).astype(int), np.array(
        valiy).ravel().astype(int)


# Helper function to evaluate the total loss on the dataset
def calculate_loss(model, X, y):
    num_examples = len(X)  # training set size
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += Config.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1. / num_examples * data_loss


def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs


#    return np.argmax(probs, axis=1)


# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(X, y, nn_hdim, num_passes, print_loss):
    # Initialize the parameters to random values. We need to learn these.
    num_examples = len(X)
    np.random.seed(0)
    W1 = np.random.randn(Config.nn_input_dim, nn_hdim) / np.sqrt(Config.nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, Config.nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, Config.nn_output_dim))

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += Config.reg_lambda * W2
        dW1 += Config.reg_lambda * W1

        # Gradient descent parameter update
        W1 += -Config.epsilon * dW1
        b1 += -Config.epsilon * db1
        W2 += -Config.epsilon * dW2
        b2 += -Config.epsilon * db2

        # Assign new parameters to the model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" % (i, calculate_loss(model, X, y)))

    return model


def main():
    trainX, trainy, valiX, valiy = generate_data()
    sns.set(style="white",color_codes=True)
    sns.jointplot()

    # PCA(trainX, trainy)
    # total = len(valiy)
    # for i in range(3, 10):
    #     for j in range(3, 6):
    #         passes = i * 1000
    #         hdim = j
    #         model = build_model(trainX, trainy, hdim, passes, False)
    #         predicty = predict(model, valiX)
    #         count = 0
    #         for k in range(total):
    #             if (predicty[k][0] > 0.5 and valiy[k] == 1) or (predicty[k][0] <= 0.5 and valiy[k] == 0):
    #                 count += 1
    #         print("num_passes: %d  nn_hdim: %d  accuracy: %f \n" % (passes, hdim, count / total))


if __name__ == "__main__":
    main()
