from xai_ranking.benchmarks.hilw import hilw_contributions


def human_in_the_loop(X, score_function, upper_bound=1, lower_bound=None):
    if lower_bound is None:
        lower_bound = X.shape[0]

    return hilw_contributions(X, score_function, upper_bound, lower_bound)
