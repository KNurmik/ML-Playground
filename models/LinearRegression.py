import autograd.numpy as np
from autograd import grad
from utils.regularization import l1_reg, l2_reg
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


class LinearRegression:
    """
    Least squares linear regression model.

    When initializing, you can specify an optional regularization penalty by setting
    regularization = "L1" or "L2", resulting in lasso and ridge regression, respectively.
    """

    def __init__(self, regularization=None):
        self.weights = None
        self.bias = None
        self.train_errors = []
        self.val_errors = []

        # Regularization function
        self.reg = None
        if regularization is not None:
            if regularization == "L1":
                self.reg = l1_reg
            elif regularization == 'L2':
                self.reg = l2_reg
            else:
                raise NotImplementedError

    def predict(self, X, params=None):
        if params is None:
            params = np.concatenate((self.weights, self.bias.reshape(1,)), axis=0)
            ones = np.ones((X.shape[0], 1))
            X = np.concatenate((X, ones), axis=1)

        return np.dot(X, params)

    def prediction_loss(self, params, X, y):
        """Unregularized loss."""
        preds = self.predict(X, params)
        return np.mean(np.square(preds - y))

    def reg_loss(self, params, lambd):
        """Regularization loss. See utils.regularization.py."""
        if self.reg is None:
            return 0.0
        return self.reg(params, lambd)

    def loss(self, params, X, y, lambd):
        return self.prediction_loss(params, X, y) + self.reg_loss(params, lambd)

    def train(self, X, y, lambd=0.1, epochs=100, lr=0.001, val_X=None, val_y=None, init_scale=0.01):
        """
        Train the model using gradient descent.
        :param X: Training input variables
        :param y: Training targets
        :param lambd: Regularization parameter lambda
        :param epochs: Number of epochs to train for
        :param lr: Training learning rate
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
        params = np.concatenate((weights, bias), axis=0)

        # Add additional input variable of ones to match weight matrix including bias
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((X, ones), axis=1)
        if val_X is not None:
            ones = np.ones((val_X.shape[0], 1))
            val_X = np.concatenate((val_X, ones), axis=1)

        # Training loop
        for i in tqdm(range(epochs)):
            params -= lr * grad(self.loss)(params, X, y, lambd)
            train_error = self.loss(params, X, y, lambd)
            self.train_errors.append(train_error)

            if val_X is not None:
                val_error = self.loss(params, val_X, val_y, lambd)
                self.val_errors.append(val_error)

        # Store final weights and bias
        self.weights = params[:-1]
        self.bias = params[-1]

        print("Final training loss: {}".format(train_error))
        if val_X is not None:
            print("Final validation loss: {}".format(val_error))

    def plot_training(self):
        """Plot the training and validation (if exists) errors during training."""
        idx = [i for i in range(len(self.train_errors))]
        plt.plot(idx, self.train_errors, label='Training loss')
        if self.val_errors:
            plt.plot(idx, self.val_errors, label='Validation loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.title("Linear Regression Training Progress")
        plt.legend()
        plt.show()
