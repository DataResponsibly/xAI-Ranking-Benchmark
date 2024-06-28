import numpy
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
    sample_size = None if "sample_size" not in kwargs else kwargs["sample_size"]
    replace = False if "replace" not in kwargs else kwargs["replace"]
    xai = ShaRP(
        qoi=qoi,
        target_function=score_function,
        measure=measure,
        sample_size=sample_size,
        replace=replace,
        random_state=random_state,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    xai.fit(X)
    contributions = xai.all(X)
    return contributions


def sharp_batch_experiment(
    X, score_function, measure="shapley", verbose=0, n_jobs=-1, batch_size=10, random_state=42
):
    xai = ShaRP(
        qoi="rank",
        target_function=score_function,
        measure=measure,
        sample_size=None,
        replace=False,
        random_state=random_state,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    batch = numpy.random.RandomState(random_state).choice(X, batch_size)
    xai.fit(batch)
    contributions = xai.all(X)
    return contributions
