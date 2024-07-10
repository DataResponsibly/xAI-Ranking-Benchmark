from xai_ranking.benchmarks.hilw import hilw_contributions, hilw_batch_contributions


def human_in_the_loop_experiment(X, score_function, upper_bound=1, lower_bound=None, *args, **kwargs):
    if lower_bound is None:
        lower_bound = X.shape[0]

    return hilw_contributions(X, score_function, upper_bound, lower_bound).values


def human_in_the_loop_batch_experiment(
        X, score_function, upper_bound=1, lower_bound=None, batch_size=10, random_state=42, *args, **kwargs
):
    if lower_bound is None:
        lower_bound = X.shape[0]

    return hilw_batch_contributions(X, score_function, upper_bound, lower_bound, batch_size, random_state).values
