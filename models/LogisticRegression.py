import autograd.numpy as np
from autograd import grad
from utils.regularization import l1_reg, l2_reg
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


class LogisticRegression:
    """A logistic regression model."""

    def __init__(self, regularization=None):
        self.weights = None
        self.bias = None

        # For storing training progress
        self.train_errors = []
        self.val_errors = []
        self.train_accs = []
        self.val_accs = []

        # Regularization function
        self.reg = None
        if regularization is not None:
            if regularization == "L1":
                self.reg = l1_reg
            elif regularization == 'L2':
                self.reg = l2_reg
            else:
                raise NotImplementedError

    def _compose_params(self, weights=None, bias=None):
        """Concatenate weights and bias into a single parameters matrix."""
        if weights is None:
            weights = self.weights
            bias = self.bias.reshape(1, )
        return np.concatenate((weights, bias), axis=0)

    def _compose_inputs(self, X):
        """Concatenate inputs with a vector of ones to account for bias in params."""
        ones = np.ones((X.shape[0], 1))
        return np.concatenate((X, ones), axis=1)

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, X, params=None):
        if params is None:
            params = self._compose_params()
            X = self._compose_inputs(X)
        return self._sigmoid(np.dot(X, params))

    def _log_bernoulli(self, y, p):
        return np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    def _prediction_loss(self, params, X, y):
        return -self._log_bernoulli(y, self.predict(X, params))

    def _reg_loss(self, params, lambd):
        """Regularization loss. See utils.regularization.py."""
        if self.reg is None:
            return 0.0
        return self.reg(params, lambd)

    def loss(self, params, X, y, lambd):
        return self._prediction_loss(params, X, y) + self._reg_loss(params, lambd)

    def get_accuracy(self, X, y, params=None, threshold=0.5):
        if params is None:
            params = self._compose_params()
        preds = self.predict(X, params) >= threshold
        return np.sum(preds == y) / y.shape[0]

    def train(self, X, y, lambd=0.1, epochs=100, lr=0.001, threshold=0.5, val_X=None, val_y=None, init_scale=0.01):
        """
        Train the model using gradient descent.
        :param X: Training input variables
        :param y: Training targets
        :param lambd: Regularization parameter lambda
        :param epochs: Number of epochs to train for
        :param lr: Training learning rate
        :param threshold: Probability threshold used for positive classification
        :param val_X: Validation inputs
        :param val_y: Validation targets
        :param init_scale: Scale factor for randomly initializing weights

        :return: None
        """
        if self.weights is None:
            weights = np.random.randn(X.shape[1]) * init_scale
            bias = np.random.randn(1) * init_scale
        else:
            weights = self.weights
            bias = self.bias.reshape(1, )
        params = self._compose_params(weights, bias)

        X = self._compose_inputs(X)
        if val_X is not None:
            val_X = self._compose_inputs(val_X)

        # Training loop
        for i in tqdm(range(epochs)):
            params -= lr * grad(self.loss)(params, X, y, lambd)
            train_error = self.loss(params, X, y, lambd)
            train_acc = self.get_accuracy(X, y, params, threshold)

            self.train_errors.append(train_error)
            self.train_accs.append(train_acc)

            if val_X is not None:
                val_error = self.loss(params, val_X, val_y, lambd)
                val_acc = self.get_accuracy(val_X, val_y, params, threshold)

                self.val_errors.append(val_error)
                self.val_accs.append(val_acc)

        # Store final weights and bias
        self.weights = params[:-1]
        self.bias = params[-1]

        print("Final training loss: {}".format(train_error))
        print("Final training accuracy: {}".format(train_acc))
        if val_X is not None:
            print("\nFinal validation loss: {}".format(val_error))
            print("Final validation accuracy: {}".format(val_acc))

    def _plot_progress(self, type):
        if type == 'Loss':
            train = self.train_errors
            val = self.val_errors
        else:
            train = self.train_accs
            val = self.val_accs

        idx = [i for i in range(len(train))]
        plt.plot(idx, train, label='Training {}'.format(type))
        if val:
            plt.plot(idx, val, label='Validation {}'.format(type))
        plt.xlabel("Epoch")
        plt.ylabel(type)

        plt.title("Logistic Regression")
        plt.legend()
        plt.show()

    def plot_training_loss(self):
        """Plot the training and validation (if exists) errors during training."""
        self._plot_progress('Loss')

    def plot_training_acc(self):
        """Plot the training and validation (if exists) accuracies during training"""
        self._plot_progress('Accuracy')
