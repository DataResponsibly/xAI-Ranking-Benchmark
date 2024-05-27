import numpy as np
from lime.lime_tabular import LimeTabularExplainer


def _scorer(X, score_function):
    scores = score_function(X)
    return np.array([1-scores, scores]).T


def lime_experiment(
    X, score_function, mode="classification"
):
    """
    `mode` can be one of `[classification, regression]`.
    """
    X_ = X.copy()
    X_["score"] = score_function(X)
    explainer = LimeTabularExplainer(
        X_,
        feature_names=X.columns,
        class_names=["score"],
        discretize_continuous=False,
        mode=mode
    )
    lime_values = explainer.explain_instance(
        X_, lambda X: _scorer(X, score_function)  # , num_features=5
    )
    return lime_values
