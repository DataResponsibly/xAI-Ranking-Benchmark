"""
Original source code:
- https://github.com/anahideh/Ranking-Explanation/blob/main/ranking_code.ipynb
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


def neighborhood_mask(ranks, idx, s):
    """
    Creates a neighborhood mask of size s around the given idx's rank.
    """
    max_rank = max(ranks)
    rank = ranks[idx]
    neighbors_ranks = np.arange(
        max(0, rank - s), min(rank + s + 1, max_rank + 1)
    ).tolist()
    mask = np.isin(ranks, neighbors_ranks)
    return mask


def feature_importance_OLS(X, y, ranks, idx, s):
    """
    Compute feature importance using Ordinary Least Squares.

    Notes
    -----
    - X is expected to be a pandas dataframe.
    """
    mask = neighborhood_mask(ranks, idx, s)
    model = sm.OLS(y[mask], X[mask])
    results = model.fit()
    return results.params


def _vip(x, model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_

    m, p = x.shape
    _, h = t.shape

    vips = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s).squeeze()

    return vips


def feature_importance_PLS(X, y, ranks, idx, s):
    """
    Compute feature importance using Partial Least Squares.
    """
    n_feat = X.shape[1]
    mask = neighborhood_mask(ranks, idx, s)
    pls = PLSRegression(n_components=n_feat)
    pls.fit(X[mask], y[mask])
    results = _vip(X[mask], pls)
    if hasattr(X, "columns"):
        return pd.Series(data=results, index=X.columns)
    else:
        return results


def feature_importance_DT(X, y, ranks, idx, s, max_depth=2):
    mask = neighborhood_mask(ranks, idx, s)
    dt = DecisionTreeRegressor(max_depth=max_depth)
    dt.fit(X[mask], y[mask])
    return pd.Series(data=dt.feature_importances_, index=dt.feature_names_in_)


def feature_importance_LR(X, y, ranks, idx, s):
    mask = neighborhood_mask(ranks, idx, s)
    lr = LinearRegression()
    lr.fit(X[mask], y[mask])

    #  Reproduce using linear algebra
    N = X[mask].shape[0]
    p = X[mask].shape[1] + 1  # plus one because LR adds an intercept term
    X_with_intercept = np.concatenate([np.ones((N, 1)), X[mask]], axis=1)
    # beta_hat = (
    #     np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    #     @ X_with_intercept.T
    #     @ y[mask].values
    # )
    # print(beta_hat)

    #  Compute standard errors of the parameter estimates
    y_hat = lr.predict(X[mask])
    residuals = np.array(y[mask]) - y_hat
    rss = residuals.T @ residuals
    sigma_squared_hat = rss / (N - p)
    var_beta_hat = (
        np.linalg.inv(X_with_intercept.T @ X_with_intercept) * sigma_squared_hat
    )
    sev = []
    for p_ in range(p):
        standard_error = var_beta_hat[p_, p_] ** 0.5
        sev.append(standard_error)

    t_statistic = lr.coef_ / np.array(sev[1:])
    return pd.Series(data=t_statistic, index=lr.feature_names_in_)
