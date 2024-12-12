import numpy as np
import numpy.random
from shap.explainers.other import LimeTabular


def lime_experiment(X, score_function, mode="regression", **kwargs):
    """
    Parameters
    ----------
    X : pandas.DataFrame
        The input data for which explanations are to be generated.
    score_function : callable
        The function used to score the data.
    mode : str, default="regression"
        The mode of the experiment. It can be either "classification" or "regression".
    **kwargs : dict
        Additional keyword arguments to be passed to the LIME explainer.

    Returns
    -------
    lime_values : array-like
        The LIME attributions for the input data `X`.
    """
    explainer = LimeTabular(
        score_function,
        X,
        mode=mode,
    )
    lime_values = explainer.attributions(X)
    return lime_values


def lime_batch_experiment(
    X, score_function, mode="regression", random_state=42, **kwargs
):
    """
    Parameters
    ----------
    X : pandas.DataFrame
        The input data for which explanations are to be generated.
    score_function : callable
        The function used to score the data.
    mode : str, default="regression"
        The mode of the experiment. It can be either "classification" or "regression".
    random_state : int, optional
        The seed used by the random number generator. Default is 42.
    **kwargs : dict
        Additional keyword arguments to be passed to the LIME explainer.

    Returns
    -------
    lime_values : array-like
        The LIME attributions for the input data `X`.
    """
    batch_size = (
        np.ceil(0.1 * len(X)).astype(int)
        if "batch_size" not in kwargs
        else kwargs["batch_size"]
    )
    batch_indices = numpy.random.RandomState(random_state).choice(X.index, batch_size)
    batch = X.loc[batch_indices]

    explainer = LimeTabular(
        score_function,
        batch,
        mode=mode,
    )
    lime_values = explainer.attributions(X)
    return lime_values
