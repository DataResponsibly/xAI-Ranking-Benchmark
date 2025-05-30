import numpy as np


def atp_score(X=None):
    """
    Scorer for ATP Tennis data.
    If X is None, return the weights. Otherwise, return the score for each row in X.
    Assumes X is preprocessed.
    """
    weights = np.array([100, 100, 100, 100, 100, -100])

    if X is None:
        return weights

    if np.array(X).ndim == 1:
        X = np.array(X).reshape(1, -1)

    return (X * weights).sum(axis=1)


def csrank_score(X=None):
    """
    Scorer for CS Rankings data.
    If X is None, return the weights. Otherwise, return the score for each row in X.
    Assumes X is preprocessed.
    Scoring function is geometric mean of the adjusted counts per research area,
    with number of sub-areas as exponent.
    """
    weights = np.array([5, 12, 3, 7])

    if X is None:
        return weights

    # multiplier contains the maximum values in the original dataset
    multiplier = np.array([71.4, 12.6, 21.1, 13.8])

    if np.array(X).ndim == 1:
        X = np.array(X).reshape(1, -1)

    return np.clip(
        (np.array(X) * multiplier) ** weights + 1, a_min=1, a_max=np.inf
    ).prod(axis=1) ** (1 / weights.sum())


def higher_education_score(X=None):
    """
    Scorer for Times Higher Education data.
    If X is None, return the weights. Otherwise, return the score for each row in X.
    Assumes X is preprocessed.
    """
    weights = np.array([0.3, 0.3, 0.3, 0.025, 0.075])

    # multiplier contains the maximum values in the original dataset
    multiplier = 100

    if X is None:
        return weights

    if np.array(X).ndim == 1:
        X = np.array(X).reshape(1, -1)

    return (np.array(X) * multiplier * weights).sum(axis=1)


def synthetic_equal_score_3ftrs(X=None):
    """
    Scorer for synthetic data.
    If X is None, return the weights. Otherwise, return the score for each row in X.
    Assumes X is preprocessed.
    """
    weights = np.array([0.333, 0.333, 0.334])

    if X is None:
        return weights

    if np.array(X).ndim == 1:
        X = np.array(X).reshape(1, -1)

    return (X * weights).sum(axis=1)
