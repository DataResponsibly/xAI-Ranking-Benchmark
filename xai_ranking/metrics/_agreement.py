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


def kendall_agreement(results1, results2):
    return results1.reset_index(drop=True).apply(
        lambda row: row_wise_kendall(row, results2.iloc[row.name]), axis=1
    ).mean()


def jaccard_agreement(results1, results2, n_features=None):
    if n_features is None:
        n_features = results1.shape[1]
    return results1.reset_index(drop=True).apply(
        lambda row: row_wise_jaccard(row, results2.iloc[row.name], n_features), axis=1
    ).mean()


def compute_all_agreement(results, n_features=None):
    """
    Compute the agreement of results for different datasets and methods.

    Parameters
    ----------
    results: dict
        A dictionary containing the results for different datasets and methods.

    n_features: int or None
        The number of top features to consider to compute Jaccard agreement.
        If None, all features are considered.

    Returns
    -------
    all_sensitivities: dict
        A dictionary containing the agreement scores for each dataset and method.

    Notes
    -----
    - The agreement is computed by comparing the rankings of the results.
    - The method names starting with "BATCH_" are excluded from the computation.
    - The agreement is computed using Kendall's tau and Jaccard similarity.
    """
    datasets = list(results.keys())
    methods = list(results[datasets[0]].keys())
    methods = [method for method in methods if not method.startswith("BATCH_")]

    all_agreements = {}
    for dataset in datasets:
        data_agreement = {
            "kendall": pd.DataFrame(columns=methods, index=methods),
            "jaccard": pd.DataFrame(columns=methods, index=methods),
        }
        for method1, method2 in product(methods, methods):
            try:  # TODO: REMOVE LATER; only for debugging
                data_agreement["kendall"].loc[method1, method2] = kendall_agreement(
                    results[dataset][method1][0], results[dataset][method2][0]
                )
                data_agreement["jaccard"].loc[method1, method2] = jaccard_agreement(
                    results[dataset][method1][0],
                    results[dataset][method2][0],
                    n_features,
                )
            except:
                print(f"Error in dataset: {dataset}, with methods: {method1}, {method2}")

        all_agreements[dataset] = data_agreement
    return all_agreements
