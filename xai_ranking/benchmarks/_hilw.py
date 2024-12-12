"""
Methods were adapted from the following paper:
    Jun Yuan and Aritra Dasgupta. 2023. A Human-in-the-loop Workflow for Multi-Factorial
    Sensitivity Analysis of Algorithmic Rankers. In Proceedings of the Workshop on
    Human-In-the-Loop Data Analytics (HILDA '23). Association for Computing Machinery,
    New York, NY, USA, Article 5, 1–5. https://doi.org/10.1145/3597465.3605221
"""

import numpy as np
from xai_ranking.benchmarks.hilw import hilw_contributions, hilw_batch_contributions


def human_in_the_loop_experiment(
    X, score_function, upper_bound=1, lower_bound=None, *args, **kwargs
):
    """
    Parameters
    ----------
    X : pandas.DataFrame
        The input data for the experiment.
    score_function : callable
        The function used to score the input data.
    upper_bound : int, optional
        The upper bound for rank of the items (default is 1).
    lower_bound : int, optional
        The lower bound for rank of the items. If None, it defaults to
        the number of rows in X.
    *args : tuple
        Additional positional arguments to pass to the hilw_contributions function.
    **kwargs : dict
        Additional keyword arguments to pass to the hilw_contributions function.

    Returns
    -------
    pandas.Series
        The contributions of the features.
    """
    if lower_bound is None:
        lower_bound = X.shape[0]

    return hilw_contributions(
        X, score_function, upper_bound, lower_bound, **kwargs
    ).values


def human_in_the_loop_batch_experiment(
    X, score_function, upper_bound=1, lower_bound=None, random_state=42, *args, **kwargs
):
    """
    Parameters
    ----------
    X : pandas.DataFrame
        The input data for the experiment.
    score_function : callable
        The function used to score the input data.
    upper_bound : int, optional
        The upper bound for rank of the items (default is 1).
    lower_bound : int, optional
        The lower bound for rank of the items. If None, it defaults to
        the number of rows in X.
    random_state : int, optional
        The seed used by the random number generator. Default is 42.
    *args : tuple
        Additional positional arguments to pass to the hilw_contributions function.
    **kwargs : dict
        Additional keyword arguments to pass to the hilw_contributions function.

    Returns
    -------
    pandas.Series
        The contributions of the features.
    """
    batch_size = (
        np.ceil(0.1 * len(X)).astype(int)
        if "batch_size" not in kwargs
        else kwargs["batch_size"]
    )
    if lower_bound is None:
        lower_bound = X.shape[0]

    return hilw_batch_contributions(
        X, score_function, upper_bound, lower_bound, batch_size, random_state
    ).values
