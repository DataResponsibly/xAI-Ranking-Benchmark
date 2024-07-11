"""
Implementation of metrics for xAI performance analysis.
"""

from itertools import product
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sharp.utils import scores_to_ordering


def jaccard_similarity(a, b):
    intersection = len(list(set(a).intersection(b)))
    union = (len(set(a)) + len(set(b))) - intersection
    return float(intersection) / union


def row_wise_kendall(results1, results2):
    row_res1 = scores_to_ordering(results1, direction=1)
    row_res2 = scores_to_ordering(results2, direction=1)
    row_sensitivity = kendalltau(row_res1, row_res2).statistic
    return row_sensitivity


def row_wise_jaccard(results1, results2, n_features):
    row_res1 = scores_to_ordering(results1, direction=-1)
    row_res2 = scores_to_ordering(results2, direction=-1)
    top_idx1 = np.indices(row_res1.shape)[0, row_res1 <= n_features]
    top_idx2 = np.indices(row_res2.shape)[0, row_res2 <= n_features]
    row_sensitivity = jaccard_similarity(top_idx1, top_idx2)
    return row_sensitivity


def kendall_sensitivity(results1, results2):
    return results1.apply(
        lambda row: row_wise_kendall(row, results2.loc[row.name]), axis=1
    ).mean()


def jaccard_sensitivity(results1, results2, n_features=None):
    if n_features is None:
        n_features = results1.shape[1]
    return results1.apply(
        lambda row: row_wise_jaccard(row, results2.loc[row.name], n_features), axis=1
    ).mean()


def compute_all_sensitivity(results, n_features=None):
    """
    Compute the sensitivity of results for different datasets and methods.

    Parameters
    ----------
    results: dict
        A dictionary containing the results for different datasets and methods.

    n_features: int or None
        The number of top features to consider to compute Jaccard sensitivity.
        If None, all features are considered.

    Returns
    -------
    all_sensitivities: dict
        A dictionary containing the sensitivity values for each dataset and method.

    Notes
    -----
    - The sensitivity is computed by comparing the rankings of the results.
    - The method names starting with "BATCH_" are excluded from the computation.
    - The sensitivity is computed using Kendall's tau and Jaccard similarity.
    """
    datasets = list(results.keys())
    methods = list(results[datasets[0]].keys())
    methods = [method for method in methods if not method.startswith("BATCH_")]

    all_sensitivities = {}
    for dataset in datasets:
        data_sensitivity = {
            "kendall": pd.DataFrame(columns=methods, index=methods),
            "jaccard": pd.DataFrame(columns=methods, index=methods),
        }
        for method1, method2 in product(methods, methods):
            try:  # TODO: REMOVE LATER; only for debugging
                data_sensitivity["kendall"].loc[method1, method2] = kendall_sensitivity(
                    results[dataset][method1][0], results[dataset][method2][0]
                )
                data_sensitivity["jaccard"].loc[method1, method2] = jaccard_sensitivity(
                    results[dataset][method1][0],
                    results[dataset][method2][0],
                    n_features,
                )
            except:
                pass
        all_sensitivities[dataset] = data_sensitivity
    return all_sensitivities


def stability(population_experiments, batch_experiments, axis=None):
    """
    Averaged sum of the squared errors (SSE) between the population and the batch
    experiments.

    Parameters
    ----------
    axis : int, optional
        The axis to average the SSE over. Default is None, which averages over all axes.
        If axis=0, the SSE is averaged over the batches. If axis=1, the SSE is averaged
        over the features.
    """
    sse = []
    for batch_exp in batch_experiments:
        squared_diffs = (batch_exp - population_experiments) ** 2

        errors_mean = squared_diffs.sum(axis=0).to_frame().T
        sse.append(errors_mean)

    sse = pd.concat(sse) ** 0.5
    mean_sse = sse.mean(axis=axis)
    sem_sse = (
        sse.sem(axis=axis)
        if axis is not None
        else np.std(sse.values) / np.sqrt(sse.size)
    )
    # return sse
    return mean_sse, sem_sse


def compute_all_stability(results, axis=None):
    """
    Compute the stability of results for different datasets and methods.

    Parameters
    ----------
    results: dict
        A dictionary containing the results for different datasets and methods.

    axis: int or None
        The axis along which to compute the stability. If None, the stability is
        computed across all axes.

    Returns
    -------
    agg_mean: dict
        A dictionary containing the mean stability values for each dataset and method.

    agg_err: dict
        A dictionary containing the error values for each dataset and method.

    Notes
    -----
    - The stability is computed by comparing population experiments with batch
        experiments.
    - The method names starting with "BATCH_" are excluded from the computation.

    Example usage:
    ```
    results = {
        'dataset1': {
            'method1': [exp1],
            'method2': [exp2],
            'BATCH_method1': [batch_exp1, batch_exp2, batch_exp3],
            'BATCH_method2': [batch_exp4, batch_exp5, batch_exp6]
        },
        'dataset2': {
            'method1': [exp1],
            'method2': [exp2],
            'BATCH_method1': [batch_exp7, batch_exp8, batch_exp9],
            'BATCH_method2': [batch_exp10, batch_exp11, batch_exp12]
        }
    }

    agg_mean, agg_err = compute_all_stability(results, axis=0)
    ```
    """
    datasets = list(results.keys())
    methods = list(results[datasets[0]].keys())
    methods = [method for method in methods if not method.startswith("BATCH_")]

    agg_mean = {}
    agg_err = {}
    for dataset in datasets:
        data_mean = {}
        data_err = {}

        try:  # TODO: REMOVE LATER; only for debugging
            for method in methods:
                population_experiments = results[dataset][method]
                batch_experiments = results[dataset][f"BATCH_{method}"]

                res_ = stability(
                    population_experiments[0], batch_experiments, axis=axis
                )
                data_mean[method] = res_[0]
                data_err[method] = res_[1]
        except:
            pass

        agg_mean[dataset] = (
            pd.DataFrame(data_mean).T if axis is not None else pd.Series(data_mean)
        )
        agg_err[dataset] = (
            pd.DataFrame(data_err).T if axis is not None else pd.Series(data_err)
        )
    return agg_mean, agg_err


def fidelity():
    pass
