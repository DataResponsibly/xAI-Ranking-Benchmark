from sharp import ShaRP


def sharp_experiment(
    X, score_function, measure="shapley", verbose=0, n_jobs=-1, random_state=42
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

    xai.fit(X)
    contributions = xai.all(X)
    return contributions
