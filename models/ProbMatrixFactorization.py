import autograd.numpy as np
from autograd import grad
from utils.regularization import l1_reg, l2_reg
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


class ProbMatrixFactorization:
    """
    A probabilistic matrix factorization model as described in
    Salakhutdinov, R., Mnih, A. (2007). Probabilistic Matrix Factorization.
    NIPS'07: Proceedings of the 20th International Conference on Neural Information Processing Systems, pp. 1257–1264.

    To avoid complicated abstract explanations, the documentation speaks of the model as a movie rating prediction
    model, which outputs a user-movie rating matrix R = U @ V, where U is a matrix of user latent variables, and
    V is a matrix of movie latent variables.

    The model requires the following data input format:
    X is an n x 2 matrix where the first column corresponds to user IDs, and the second corresponds to movie IDs, i.e.
    every row means the given user rated the given movie.
    y is the corresponding n x 1 vector where every input contains the rating the user gave to the movie.

    The model assumes that the validation and testing sets contain the same userIDs and movieIDs as the training set.
    Furthermore, the IDs must be continuous integers starting from 1. That is, if there exists a user with ID 5, then
    there must also exist users with IDs 1, 2, 3, and 4.
    """

    def __init__(self, regularization=None):
        self.latent_u = None
        self.latent_v = None

        # For storing training progress
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

    def predict(self, X, latent_u=None, latent_v=None):
        if latent_u is None:
            latent_u = self.latent_u
            latent_v = self.latent_v
        u_idx, v_idx = (X[:, 0] - 1, X[:, 1] - 1)

        return np.sum(latent_u[u_idx] * latent_v[v_idx], axis=1)

    def _reg_loss(self, params, lambd):
        """Regularization loss. See utils.regularization.py."""
        if self.reg is None:
            return 0.0
        return self.reg(params, lambd)

    def _prediction_loss(self, latent_u, latent_v, X, y):
        preds = self.predict(X, latent_u, latent_v)
        return np.mean((preds - y)**2)

    def loss(self, params, X, y, lambd_u, lambd_v):
        latent_u, latent_v = params
        return self._prediction_loss(latent_u, latent_v, X, y) + \
               self._reg_loss(latent_u, lambd_u) + self._reg_loss(latent_v, lambd_v)

    def _update(self, params, X, y, lambd_u, lambd_v, lr):
        grad_u, grad_v = grad(self.loss)(params, X, y, lambd_u, lambd_v)
        latent_u, latent_v = params
        return latent_u - lr * grad_u, latent_v - lr * grad_v

    def train(self, X, y, latent_size=3,
              lambd_u=0.1, lambd_v=0.1, epochs=100, lr=0.1, val_X=None, val_y=None, init_scale=0.01):
        """
        Train the model using gradient descent.
        :param X: n x 2 matrix of user IDs and movie IDs
        :param y: n x 1 vector of ratings
        :param latent_size: size of latent representation vectors
        :param lambd_u: regularization parameter for user latent matrix
        :param lambd_v: regularization parameter for movie latent matrix
        :param epochs: number of epochs to train for
        :param lr: learning rate
        :param val_X: validation userIDs and movieIDs
        :param val_y: validation ratings
        :param init_scale: scaling parameter for initializing latent matrices

        :return: None
        """
        if self.latent_u is None:
            n_users = np.unique(X[:, 0]).shape[0]
            n_movies = np.unique(X[:, 1]).shape[0]

            latent_u = np.random.randn(n_users, latent_size) * init_scale
            latent_v = np.random.randn(n_movies, latent_size) * init_scale
        else:
            latent_u = self.latent_u
            latent_v = self.latent_v

        params = (latent_u, latent_v)

        # Training loop
        for i in tqdm(range(epochs)):
            params = self._update(params, X, y, lambd_u, lambd_v, lr)

            train_error = self.loss(params, X, y, lambd_u, lambd_v)
            self.train_errors.append(train_error)

            if val_X is not None:
                val_error = self.loss(params, val_X, val_y, lambd_u, lambd_v)
                self.val_errors.append(val_error)

        # Store final latent matrices
        self.latent_u, self.latent_v = params

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

        plt.title("Matrix Factorization Training Progress")
        plt.legend()
        plt.show()


