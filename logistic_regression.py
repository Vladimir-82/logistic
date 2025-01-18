"""Logistic regression example."""

import numpy as np
from scipy import sparse


class LogisticRegression:
    def __init__(self):
        self.w = None
        self.loss_history = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        """Train this classifier using stochastic gradient descent.

        Inputs:
        - X: N x D array of training data. Each training point is a D-dimensional
             column.
        - y: 1-dimensional array of length N with labels 0-1, for 2 classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        # Add a column of ones to X for the bias sake.
        X = LogisticRegression.append_biases(X)
        num_train, dim = X.shape
        if self.w is None:
            # lazily initialize weights
            self.w = np.random.randn(dim) * 0.01

        # Run stochastic gradient descent to optimize W
        self.loss_history = []
        for it in range(num_iters):

            indices = np.random.choice(num_train, batch_size)
            X_batch = X[indices]
            y_batch = y[indices]

            # evaluate loss and gradient
            loss, gradW = self.loss(X_batch, y_batch, reg)
            self.loss_history.append(loss)
            # perform parameter update

            self.w -= learning_rate * gradW

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
        return self

    def predict_proba(self, X, append_bias=False):
        """Use the trained weights of this linear classifier to predict probabilities for
        data points.

        Inputs:
        - X: N x D array of data. Each row is a D-dimensional point.
        - append_bias: bool. Whether to append bias before predicting or not.

        Returns:
        - y_proba: Probabilities of classes for the data in X. y_pred is a 2-dimensional
          array with a shape (N, 2), and each row is a distribution of classes [prob_class_0, prob_class_1].
        """
        if append_bias:
            X = LogisticRegression.append_biases(X)
            scores = X.dot(self.w)
            return 1 / (1 + np.exp(-scores))

    def predict(self, X):
        """Use the ```predict_proba``` method to predict labels for data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        return np.argmax(self.predict_proba(X, append_bias=True), axis=1)  # added

    def loss(self, X_batch, y_batch, reg):
        """Logistic Regression loss function
        Inputs:
        - X: N x D array of data. Data are D-dimensional rows
        - y: 1-dimensional array of length N with labels 0-1, for 2 classes
        Returns:
        a tuple of:
        - loss as single float
        - gradient with respect to weights w; an array of same shape as w.
        """
        num_train = X_batch.shape[0]
        scores = X_batch.dot(self.w)
        y_proba = 1 / (1 + np.exp(-scores))

        loss = self.loss_calculate(y_batch, y_proba, reg)
        dw = self.gradient_calculate(X_batch, y_batch, reg, y_proba, num_train)

        return loss, dw

    def loss_calculate(self, y_batch, y_proba, reg):
        """Loss function calculation."""
        loss = -np.mean(y_batch * np.log(y_proba + 1e-15) + (1 - y_batch) * np.log(1 - y_proba + 1e-15))
        loss += reg * np.sum(self.w[1:] ** 2)
        return loss

    def gradient_calculate(self, X_batch, y_batch, reg, y_proba, num_train):
        """Gradient calculation."""
        dw = X_batch.T.dot(y_proba - y_batch) / num_train
        dw[1:] += reg * self.w[1:]
        return dw

    @staticmethod
    def append_biases(X):
        """Appends bias to each column of X."""
        return sparse.hstack((X, np.ones(X.shape[0])[:, np.newaxis])).tocsr()
