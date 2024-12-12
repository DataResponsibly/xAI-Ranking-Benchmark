"""
Methods were used from the following paper:

    Pliatsika, V., Fonseca, J., Akhynko, K., Shevchenko, I., Stoyanovich, J. (2024).
    ShaRP: A Novel Feature Importance Framework for Ranking.
    https://doi.org/10.48550/arXiv.2401.16744

The code is available at:
    https://github.com/DataResponsibly/ShaRP

"""

import numpy as np
from sharp import ShaRP


def sharp_experiment(
    X,
    score_function,
    measure="shapley",
    verbose=0,
    n_jobs=-1,
    random_state=42,
    **kwargs
):
    """
    Conducts a ShaRP experiment to compute feature contributions.

    Parameters
    ----------
    X : pandas.DataFrame
        The input data for which the feature contributions are to be computed.
    score_function : callable
        The function used to score the input data.
    measure : str, default="shapley"
        The measure to use for computing feature contributions.
    verbose : int, default=0
        The verbosity level of the output.
    n_jobs : int, default=-1
        The number of jobs to run in parallel. -1 means using all processors.
    random_state : int, default=42
        The seed used by the random number generator.
    **kwargs : dict
        Additional keyword arguments to pass to the ShaRP class.

    Returns
    -------
    contributions : numpy.array
        The computed feature contributions for the input data.
    """
    qoi = "rank" if "qoi" not in kwargs else kwargs["qoi"]
    kwargs.pop("qoi")
    xai = ShaRP(
        qoi=qoi,
        target_function=score_function,
        measure=measure,
        random_state=random_state,
        verbose=verbose,
        n_jobs=n_jobs,
        **kwargs
    )

    xai.fit(X)
    contributions = xai.all(X)
    return contributions


def sharp_batch_experiment(
    X,
    score_function,
    measure="shapley",
    verbose=0,
    n_jobs=-1,
    random_state=42,
    **kwargs
):
    """
    Conducts a ShaRP experimentfor batches to compute feature contributions.

    Parameters
    ----------
    X : pandas.DataFrame
        The input data for which the feature contributions are to be computed.
    score_function : callable
        The function used to score the input data.
    measure : str, default="shapley"
        The measure to use for computing feature contributions.
    verbose : int, default=0
        The verbosity level of the output.
    n_jobs : int, default=-1
        The number of jobs to run in parallel. -1 means using all processors.
    random_state : int, default=42
        The seed used by the random number generator.
    **kwargs : dict
        Additional keyword arguments to pass to the ShaRP class.

    Returns
    -------
    contributions : numpy.array
        The computed feature contributions for the input data.
    """
    qoi = "rank" if "qoi" not in kwargs else kwargs["qoi"]
    kwargs.pop("qoi")
    batch_size = (
        np.ceil(0.1 * len(X)).astype(int)
        if "batch_size" not in kwargs
        else kwargs["batch_size"]
    )
    xai = ShaRP(
        qoi=qoi,
        target_function=score_function,
        measure=measure,
        random_state=random_state,
        verbose=verbose,
        n_jobs=n_jobs,
        **kwargs
    )

    batch_indices = np.random.RandomState(random_state).choice(X.index, batch_size)
    batch = X.loc[batch_indices]

    xai.fit(batch)
    contributions = xai.all(X) * (X.shape[0] / batch_size)
    return contributions
