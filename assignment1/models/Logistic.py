"""Logistic regression model."""

import numpy as np
import math


class Logistic:
    def __init__(self, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.threshold = 0.5

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        ret = 1 / (1 + math.exp(-z))
        return ret

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        N, D = X_train.shape
        self.w = np.random.randn(1, D)

        for k in range(self.epochs):
            for i in range(N):
                if y_train[i] == 0:
                    yt = -1
                else:
                    yt = 1
                self.w += self.lr * \
                    self.sigmoid(-yt * np.dot(self.w,
                                              X_train[i])) * yt * X_train[i]
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
            resp = np.sign(np.dot(self.w, X_test[i]))
            if resp == -1:
                y_test[i] = 0
            else:
                y_test[i] = 1
        return y_test
