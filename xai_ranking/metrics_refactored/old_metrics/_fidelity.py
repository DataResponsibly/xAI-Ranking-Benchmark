import numpy as np
from sklearn.utils import check_random_state
from sharp.utils import scores_to_ordering


def row_wise_fidelity(
    original_data, score_func, contributions, row_idx, threshold=0.8, n_tests=100, std_multiplier=0.2, random_state=None
):
    """
    Prediction gap fidelity as defined in https://arxiv.org/pdf/2205.07277

    NOTE: Instead of calculating the mean of the prediction gap, we calculate the 
    standard deviation. We do this because the noise added to the data has mean 0.
    """
    original_data_score = score_func(original_data)
    row_data = np.array(original_data)[row_idx]
    row_cont = np.array(contributions)[row_idx]
    row_rank = scores_to_ordering(original_data_score)[row_idx]

    # Find the most important features
    total_contribution = np.sum(np.abs(row_cont))
    order = np.argsort(np.abs(row_cont))
    original_order = np.argsort(order)
    cumulative_cont = np.cumsum(np.abs(row_cont)[order]) / total_contribution
    feature_mask = (cumulative_cont < 1-threshold)[original_order]

    # Apply perturbation to the most important features
    rng = check_random_state(random_state)
    stds = np.std(original_data, axis=0) * std_multiplier
    rows_perturbed = np.array(
        [row_data + rng.normal(loc=0, scale=stds) * feature_mask for _ in range(n_tests)]
    )

    # Compute the prediction gap fidelity
    # NOTE: Calculating std instead of mean (produced noise has mean 0)
    rows_perturbed_score = score_func(rows_perturbed)
    rows_perturbed_rank = np.array([
        scores_to_ordering(
            np.append(original_data_score, score)
        )[-1]
        for score in rows_perturbed_score
    ])
    score = (row_rank - rows_perturbed_rank).std()
    return score


def fidelity(
    original_data, score_func, contributions, threshold=0.8, n_tests=100, std_multiplier=0.2, aggregate_results=True, random_state=None
):
    sensitivities = np.vectorize(
        lambda row_idx: row_wise_fidelity(
            original_data, score_func, contributions, row_idx, threshold, n_tests, std_multiplier, random_state
        )
    )(
        np.arange(len(original_data))
    )
    if aggregate_results:
        return np.mean(sensitivities), np.std(sensitivities) / np.sqrt(sensitivities.size)
    else:
        return sensitivities


def compute_all_fidelity(
    original_data, results, threshold=0.8, n_tests=100, std_multiplier=0.2, aggregate_results=True, random_state=None
):
    datasets = list(results.keys())
    methods = list(results[datasets[0]].keys())
    methods = [method for method in methods if not method.startswith("BATCH_")]

    agg_mean = {}
    agg_err = {}
    for dataset in original_data:
        dataset_name = dataset["name"]
        score_func = dataset["scorer"]
        data_mean = {}
        data_err = {}
        for method in methods:
            try:  # TODO: REMOVE LATER; only for debugging
                contributions = results[dataset_name][method][0]
                data_mean[method], data_err[method] = fidelity(
                    dataset["data"][0],
                    score_func,
                    contributions,
                    threshold,
                    n_tests,
                    std_multiplier,
                    aggregate_results,
                    random_state
                )
            except:
                print(f"Error in dataset: {dataset_name}, with method: {method}")
        agg_mean[dataset_name] = data_mean
        agg_err[dataset_name] = data_err
    return agg_mean, agg_err
