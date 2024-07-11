"""
Local Explanations of Global Rankings: Insights for Competitive Rankings

Hierarchical Ranking Explanation (HRE) framework.

Anahideh, H., & Mohabbati-Kalejahi, N. (2022). Local explanations of global
rankings: insights for competitive rankings. IEEE Access, 10, 30676-30693.

Original source code:
- https://github.com/anahideh/Ranking-Explanation/blob/main/ranking_code.ipynb
"""

import numpy as np
from sharp.utils import scores_to_ordering
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
    `model_type` can be one of "DT", "LR", "OLS", "PLS".
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
    batch_size=10,
    random_state=42,
    *args,
    **kwargs,
):
    """
    `model_type` can be one of "DT", "LR", "OLS", "PLS".
    """
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
