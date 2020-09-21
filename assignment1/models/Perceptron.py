"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None # initialize in train()
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in Lecture 3.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        N, D = X_train.shape
        self.w = np.random.randn(self.n_class, D)

        for k in range(self.epochs):
            for i in range(N):
                resp = np.dot(self.w, X_train[i]) # a vector of response
                correct = resp[y_train[i]]
                for j in range(self.n_class):
                    if j == y_train[i]:
                        self.w[j] += self.lr * X_train[i]
                    if resp[j] > correct:
                        self.w[j] -= self.lr * X_train[i]
        # for t in self.w:
        #     print(t)
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        N, D = X_test.shape
        y_test = np.zeros(N)
        for i in range(N):
            y_test[i] = np.argmax(np.dot(self.w, X_test[i]))
        return y_test
