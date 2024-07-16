import numpy as np
from sklearn.utils import check_random_state
from ._base import _MEASURES
from ._sensitivity import _pairwise_outcome_sensitivity


def bootstrapped_explanation_consistency(
    population_result, batch_results, measure="kendall", **kwargs
):
    batch_agreement = []
    for batch_exp in batch_results:
        mean_agreement = _MEASURES[measure](
            population_result, batch_exp, **kwargs
        ).mean()
        batch_agreement.append(mean_agreement)

    batch_agreement = np.array(batch_agreement)
    mean = batch_agreement.mean()
    sem = np.std(batch_agreement) / np.sqrt(batch_agreement.size)
    return mean, sem


def cross_method_explanation_consistency(
    results1, results2, measure="kendall", **kwargs
):
    res_ = _MEASURES[measure](results1, results2, **kwargs)
    mean = res_.mean()
    sem = np.std(res_) / np.sqrt(res_.size)
    return mean, sem


def _row_based_outcome_consistency(
    original_data,
    original_scores,
    score_func,
    contributions1,
    contributions2,
    row_idx,
    threshold,
    n_tests,
    stds,
    rng,
):
    return _pairwise_outcome_sensitivity(
        np.array(original_data)[row_idx],
        np.array(original_data)[row_idx],
        np.array(contributions1)[row_idx],
        np.array(contributions2)[row_idx],
        score_func,
        original_scores,
        threshold,
        n_tests,
        stds,
        rng,
    )


def cross_method_outcome_consistency(
    original_data,
    score_func,
    contributions1,
    contributions2,
    threshold=0.8,
    n_tests=10,
    std_multiplier=0.2,
    random_state=None,
):
    original_scores = score_func(original_data)
    stds = np.std(original_data, axis=0) * std_multiplier
    rng = check_random_state(random_state)

    consistencies = np.vectorize(
        lambda row_idx: _row_based_outcome_consistency(
            original_data,
            original_scores,
            score_func,
            contributions1,
            contributions2,
            row_idx,
            threshold,
            n_tests,
            stds,
            rng,
        )
    )(np.arange(len(original_data)))
    return np.mean(consistencies), np.std(consistencies) / np.sqrt(consistencies.size)
