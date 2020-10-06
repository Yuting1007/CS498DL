"""Neural network model."""

from typing import Sequence

import numpy as np


class optimizer:
    def __init__(self, isadam, x):
        self.x = x
        self.isadam = isadam
        if isadam:
            self.beta_1 = 0.9
            self.beta_2 = 0.99
            self.epsilon = 1e-8
            self.t = 0
            self.m_t = np.zeros_like(x)
            self.v_t = np.zeros_like(x)

    def update(self, grad, lr = 1e-3):
        if self.isadam:
            self.t += 1
            self.m_t = self.beta_1 * self.m_t + (1.0 - self.beta_1) * grad
            self.v_t = self.beta_2 * self.v_t + (1.0 - self.beta_2) * (grad * grad)
            m_cap = self.m_t / (1.0 - (self.beta_1 ** self.t))
            v_cap = self.v_t / (1.0 - (self.beta_2 ** self.t))
            new_grad = m_cap/(np.sqrt(v_cap)+self.epsilon)
            # print(grad)
            # print(new_grad)
            # print("")
            self.x -= lr * new_grad
        else:
            self.x -= lr * grad


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices.

    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are the scores for
    each class."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        isadam = False,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:

        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)

        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: The number of classes C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers


        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = optimizer(isadam, np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1]))
            self.params["b" + str(i)] = optimizer(isadam, np.zeros(sizes[i]))

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.

        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias

        Returns:
            the output
        """

        return np.dot(X, W) + b

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).

        Parameters:
            X: the input data

        Returns:
            the output
        """
        return np.maximum(X, 0)

    def softmax(self, X: np.ndarray) -> np.ndarray:
        """The softmax function.

        Parameters:
            X: the input data

        Returns:
            the output
        """
        exp = np.exp(X - np.max(X))
        return exp / np.sum(exp)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the scores for each class for all of the data samples.

        Hint: this function is also used for prediction.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample

        Returns:
            Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of your network
        """
        self.outputs = {}
        self.outputs[0] = X
        # N, D = X.shape
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.softmax in here.
        cur_z = X
        for i in range(1, self.num_layers + 1):
            cur_weights = self.params["W" + str(i)].x
            cur_bias = self.params["b" + str(i)].x
            cur_z = self.linear(cur_weights, cur_z, cur_bias)
            self.outputs[i] = cur_z
            if i != self.num_layers:
                cur_z = self.relu(cur_z)


        ret = cur_z
        for i in range(cur_z.shape[0]):
            ret[i] = self.softmax(cur_z[i])
        return ret # a matrix of (N, C)? not sure

    def predict_batch(self, X: np.ndarray, y: np.ndarray):
        output = self.forward(X)
        output_label = np.argmax(output, axis=1)
        return output_label, (np.sum(output_label == y) / y.shape[0])

    def predict(self, X: np.ndarray, y: np.ndarray, batch_size=1000):
        preds = []
        labels = []
        for batch in range(X.shape[0] // batch_size):
            X_batch = X[batch * batch_size: (batch + 1) * batch_size, :]
            y_batch = y[batch * batch_size: (batch + 1) * batch_size]
            labels.extend(y_batch.tolist())
            output, _ = self.predict_batch(X_batch, y_batch)
            preds.extend(output.tolist())
        return preds, (np.sum(np.array(labels) == np.array(preds)) / len(preds))

    def backward(
        self, X: np.ndarray, y: np.ndarray, lr: float, reg: float = 0.0
    ) -> float:
        """Perform back-propagation and update the parameters using the
        gradients.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training sample
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            lr: Learning rate
            reg: Regularization strength

        Returns:
            Total loss for this batch of training samples
        """

        output = self.forward(X)
        output_label = np.argmax(output, axis=1)
        n_instance = output.shape[0]
        loss = np.sum(-np.log(output[np.arange(n_instance), y]))
        loss /= n_instance
        for i in range(1, self.num_layers + 1):
            loss += reg * np.sum(self.params["W" + str(i)].x * self.params["W" + str(i)].x)
        output[np.arange(n_instance), y] -= 1
        output /= n_instance

        for i in range(self.num_layers, 0, -1):
            grad_w = self.relu(self.outputs[i - 1].T).dot(output) + reg * 2 * self.params["W" + str(i)].x
            grad_b = output.sum(axis=0) + reg * 2 * self.params["b" + str(i)].x
            output = output.dot(self.params["W" + str(i)].x.T) * (self.outputs[i - 1] > 0)
            self.params["W" + str(i)].update(grad_w, lr)
            self.params["b" + str(i)].update(grad_b, lr)

        # TODO: implement me. You'll want to store the gradient of each layer
        # in self.gradients if you want to be able to debug your gradients
        # later. You can use the same keys as self.params. You can add
        # functions like self.linear_grad, self.relu_grad, and
        # self.softmax_grad if it helps organize your code.
        return output_label, loss
