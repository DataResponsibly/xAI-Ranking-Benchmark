import numpy.random
from shap.explainers.other import LimeTabular


def lime_experiment(X, score_function, mode="regression", **kwargs):
    """
    `mode` can be one of `[classification, regression]`.
    """
    explainer = LimeTabular(
        score_function,
        X,
        mode=mode,
        **kwargs,
    )
    lime_values = explainer.attributions(X)
    return lime_values


def lime_batch_experiment(X, score_function, mode="regression", batch_size=10, random_state=42):
    """
    `mode` can be one of `[classification, regression]`.
    """
    batch_indices = numpy.random.RandomState(random_state).choice(X.index, batch_size)
    batch = X[batch_indices]

    explainer = LimeTabular(
        score_function,
        batch,
        mode=mode,
    )
    lime_values = explainer.attributions(X)
    return lime_values
