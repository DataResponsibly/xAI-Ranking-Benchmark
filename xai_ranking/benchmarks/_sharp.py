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
    qoi = "rank" if "qoi" not in kwargs else kwargs["qoi"]
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
    qoi = "rank" if "qoi" not in kwargs else kwargs["qoi"]
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
