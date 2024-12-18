"""
Anahideh, H., & Mohabbati-Kalejahi, N. (2022). Local explanations of global
rankings: insights for competitive rankings. IEEE Access, 10, 30676-30693.

Original source code:
- https://github.com/anahideh/Ranking-Explanation/blob/main/ranking_code.ipynb
"""

import numpy as np
from xai_ranking.utils import scores_to_ordering
from .hre import (  # noqa
    feature_importance_DT,
    feature_importance_LR,
    feature_importance_OLS,
    feature_importance_PLS,
)


def hierarchical_ranking_explanation(
    X, score_function, model_type="OLS", s=5, *args, **kwargs
):
    """
    Parameters
    ----------
    X : pandas.DataFrame
        The input data for which explanations are to be generated.
    score_function : callable
        A function that takes the input data X and returns scores.
    model_type : str, optional
        The type of model to use for feature importance calculation.
        Can be one of "DT" (Decision Tree), "LR" (Logistic Regression),
        "OLS" (Ordinary Least Squares), or "PLS" (Partial Least Squares).
        Default is "OLS".
    s : int, optional
        A parameter for the feature importance function. Default is 5.
    *args : tuple
        Additional arguments to pass to the feature importance function.
    **kwargs : dict
        Additional keyword arguments to pass to the feature importance function.

    Returns
    -------
    numpy.ndarray
        An array of contributions for each observation in the input data.
    """
    # index = X.index
    X = X.copy().reset_index(drop=True)
    y = score_function(X)
    ranks = scores_to_ordering(y)

    func_name = f"feature_importance_{model_type}"
    feature_importance_func = eval(func_name)

    # TODO: Refactor appropriately (do we need this loop?)
    contributions = []
    for idx in range(X.shape[0]):
        obs_contr = feature_importance_func(X, y, ranks, idx, s)
        contributions.append(obs_contr)
    return np.array(contributions)


def hierarchical_ranking_batch_explanation(
    X,
    score_function,
    model_type="OLS",
    s=5,
    random_state=42,
    *args,
    **kwargs,
):
    """
    Parameters
    ----------
    X : pandas.DataFrame
        The input data for which explanations are to be generated.
    score_function : callable
        A function that takes the input data X and returns scores.
    model_type : str, optional
        The type of model to use for feature importance calculation.
        Can be one of "DT" (Decision Tree), "LR" (Logistic Regression),
        "OLS" (Ordinary Least Squares), or "PLS" (Partial Least Squares).
        Default is "OLS".
    s : int, optional
        A parameter for the feature importance function. Default is 5.
    random_state : int, optional
        The seed used by the random number generator. Default is 42.
    *args : tuple
        Additional arguments to pass to the feature importance function.
    **kwargs : dict
        Additional keyword arguments to pass to the feature importance function.

    Returns
    -------
    numpy.ndarray
        An array of contributions for each observation in the input data.
    """
    batch_size = (
        np.ceil(0.1 * len(X)).astype(int)
        if "batch_size" not in kwargs
        else kwargs["batch_size"]
    )
    batch_indices = np.random.RandomState(random_state).choice(X.index, batch_size)
    batch = X.loc[batch_indices].copy().reset_index(drop=True)
    batch_scores = score_function(batch)

    X = X.copy().reset_index(drop=True)
    y = score_function(X)

    func_name = f"feature_importance_{model_type}"
    feature_importance_func = eval(func_name)

    # TODO: Refactor appropriately (do we need this loop?)
    contributions = []
    for idx in range(X.shape[0]):
        cur_batch_scores = np.concatenate((np.array([y[idx]]), batch_scores), axis=0)
        ranks = scores_to_ordering(cur_batch_scores)
        obs_contr = feature_importance_func(
            np.concatenate((np.array([X.iloc[idx]]), batch), axis=0),
            cur_batch_scores,
            ranks,
            0,
            s,
        )
        contributions.append(obs_contr)
    return np.array(contributions)
