import numpy as np
from shap import Explainer


def shap_experiment(X, score_function, **kwargs):
    """
    Calculates SHAP values.
    Parameters
    ----------
    X : pandas.DataFrame
        The input data for which SHAP values are to be computed.
    score_function : callable
        A function that takes the input data and returns the corresponding scores.
    **kwargs :
        Additional keyword arguments to be passed to the Explainer.

    Returns
    -------
    numpy.ndarray
        The SHAP values for the input data.
    """
    explainer = Explainer(score_function, masker=X)
    shap_values = explainer(X)
    return shap_values.values


def shap_batch_experiment(X, score_function, random_state=42, **kwargs):
    """
    Calculates SHAP values.
    Parameters
    ----------
    X : pandas.DataFrame
        The input data for which SHAP values are to be computed.
    score_function : callable
        A function that takes the input data and returns the corresponding scores.
    random_state : int, optional
        The seed used by the random number generator. Default is 42.
    **kwargs :
        Additional keyword arguments to be passed to the Explainer.

    Returns
    -------
    numpy.ndarray
        The SHAP values for the input data.
    """
    batch_size = (
        np.ceil(0.1 * len(X)).astype(int)
        if "batch_size" not in kwargs
        else kwargs["batch_size"]
    )
    batch_indices = np.random.RandomState(random_state).choice(X.index, batch_size)
    batch = X.loc[batch_indices]

    explainer = Explainer(score_function, masker=batch)
    shap_values = explainer(X)
    return shap_values.values
