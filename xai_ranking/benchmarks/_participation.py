"""
Participation score, as proposed in:

    Abraham Gale and Amelie Marian. Explaining ranking functions. Proc. VLDB
    Endow., 14(4):640–652, 2020.

Code adapted from: https://github.com/yehudagale/metricComputer/tree/main/vldb_ranking
"""

from xai_ranking.utils import scores_to_ordering


def participation_score(X, ranks, top_k=10):
    """
    Computes the participation score for the top_k items.

    Parameters
    ----------
    X : pandas.DataFrame
        The input data.
    ranks : pandas.Series
        The ranks of the items.
    top_k : int, optional
        The number of top items to consider. Default is 10.

    Returns
    -------
    pandas.Series
        The participation score for each feature.
    """
    mask = ranks <= top_k
    X_top = X[mask]
    # thresh = score_function(X_top).min()

    pointwise_part = X_top.div(X_top.sum(axis=1), axis=0)
    return pointwise_part.mean(axis=0)


def weighted_participation_score(X, ranks, weights, top_k=10):
    """
    Computes the weighted participation score for the top_k items.

    Parameters
    ----------
    X : pandas.DataFrame
        The input data.
    ranks : pandas.Series
        The ranks of the items.
    weights : pandas.Series
        The weights for each item.
    top_k : int, optional
        The number of top items to consider. Default is 10.

    Returns
    -------
    pandas.Series
        The weighted participation score for each feature.
    """
    mask = ranks <= top_k
    X_top = X[mask].mul(weights)
    # thresh = score_function(X_top).min()

    pointwise_part = X_top.div(X_top.sum(axis=1), axis=0)
    return pointwise_part.mean(axis=0)


def participation_experiment(X, score_function, top_k=10, weights=None):
    """
    Runs the participation score experiment.

    Parameters
    ----------
    X : pandas.DataFrame
        The input data.
    score_function : callable
        The function to compute scores.
    top_k : int, optional
        The number of top items to consider. Default is 10.
    weights : pandas.Series, optional
        The weights for each item. Default is None.

    Returns
    -------
    pandas.Series
        The participation score or weighted participation score for each feature.
    """
    ranks = scores_to_ordering(score_function(X))
    if weights is not None:
        return weighted_participation_score(X, ranks, weights=weights, top_k=top_k)
    else:
        return participation_score(X, ranks, top_k=top_k)
