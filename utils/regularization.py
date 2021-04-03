import numpy as np


def l1_reg(params, lambd):
    """L1 regularization."""
    return lambd * np.sum(np.abs(params))


def l2_reg(params, lambd):
    """L2 regularization."""
    return lambd * np.sum(np.square(params))

