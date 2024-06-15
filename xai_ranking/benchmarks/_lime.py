from shap.explainers.other import LimeTabular


def lime_experiment(X, score_function, mode="regression"):
    """
    `mode` can be one of `[classification, regression]`.
    """
    X_ = X.copy()
    X_["score"] = score_function(X)
    explainer = LimeTabular(
        score_function,
        X,
        mode=mode,
    )
    lime_values = explainer.attributions(X)
    return lime_values
