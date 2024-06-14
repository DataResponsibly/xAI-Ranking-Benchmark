import numpy as np


def atp_score(X=None):
    """
    If X is None, return the weights. Otherwise, return the score for each row in X.
    """
    weights = np.array([100, 100, 100, 100, 1, -1])

    if X is None:
        return weights

    if X.ndim == 1:
        X = np.array(X).reshape(1, -1)
    return (X * weights).sum(axis=1)