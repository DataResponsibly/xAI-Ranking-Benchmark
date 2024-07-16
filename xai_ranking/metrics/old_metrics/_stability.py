import numpy as np
import pandas as pd


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
            print(f"Error in dataset: {dataset}, with method: {method}")

        agg_mean[dataset] = (
            pd.DataFrame(data_mean).T if axis is not None else pd.Series(data_mean)
        )
        agg_err[dataset] = (
            pd.DataFrame(data_err).T if axis is not None else pd.Series(data_err)
        )
    return agg_mean, agg_err
