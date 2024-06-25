import numpy as np


def atp_score(X=None):
    """
    If X is None, return the weights. Otherwise, return the score for each row in X.
    Assumes X is preprocessed.
    """
    weights = np.array([100, 100, 100, 100, 100, -100])

    if X is None:
        return weights

    if X.ndim == 1:
        X = np.array(X).reshape(1, -1)
    return (X * weights).sum(axis=1)


def csrank_score(X=None):
    """
    If X is None, return the weights. Otherwise, return the score for each row in X.
    Assumes X is preprocessed.
    """
    weights = np.array([5, 12, 3, 7])

    if X is None:
        return weights

    # multiplier contains the maximum values in the original dataset
    multiplier = np.array([71.4, 12.6, 21.1, 13.8])

    return np.clip(
        (np.array(X)[:, :-1] * multiplier) ** weights + 1, a_min=1, a_max=np.inf
    ).prod(axis=1) ** (1 / weights.sum())


def higher_education_score(X=None):
    """
    If X is None, return the weights. Otherwise, return the score for each row in X.
    Assumes X is preprocessed.
    """
    weights = np.array([0.3, 0.3, 0.3, 0.025, 0.075])

    # multiplier contains the maximum values in the original dataset
    multiplier = 100

    if X is None:
        return weights

    return (np.array(X) * multiplier * weights).sum(axis=1)


def synthethic_score(X=None):
    """
    If X is None, return the weights. Otherwise, return the score for each row in X.
    Assumes X is preprocessed.
    """
    weights = np.array([0.25, 0.25, 0.25, 0.25])

    if X is None:
        return weights
    
    return (np.array(X) * weights).sum(axis=1)
