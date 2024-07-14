import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from xai_ranking.metrics._agreement import row_wise_kendall
from sharp.utils import scores_to_ordering


def row_wise_sensitivity(original_data, contributions, row_idx, rankings, n_neighbors, agg_type="mean"):
    row_data = np.array(original_data)[row_idx]
    row_cont = np.array(contributions)[row_idx]
    row_rank = np.array(rankings)[row_idx]
    min_ranking = max(0, row_rank - n_neighbors)
    max_ranking = min(row_rank + n_neighbors, max(rankings))

    # Select neighbors that are close ranking-wise
    mask = (
        (rankings >= min_ranking) & (rankings <= max_ranking) & (rankings != row_rank)
    )
    data_neighbors = np.array(original_data)[mask]
    cont_neighbors = np.array(contributions)[mask]

    # Select neighbors that are close distance-wise
    distances = np.apply_along_axis(
        lambda row: euclidean(row, row_data), 1, data_neighbors
    )
    neighbors_idx = np.argpartition(distances, -n_neighbors)[-n_neighbors:]
    data_neighbors = data_neighbors[neighbors_idx]
    cont_neighbors = cont_neighbors[neighbors_idx]

    # Compute Kendall tau distance between the target point and its neighbors
    distances = np.apply_along_axis(
        lambda row: row_wise_kendall(row, row_cont), 1, cont_neighbors
    )

    if agg_type == "max":
        return np.max(distances)
    elif agg_type == "mean":
        return np.mean(distances)
    else:
        raise ValueError(f"Unknown aggregation type: {agg_type}")


def sensitivity(original_data, contributions, rankings, n_neighbors, agg_type="mean"):
    sensitivities = np.vectorize(
        lambda row_idx: row_wise_sensitivity(
            original_data, contributions, row_idx, rankings, n_neighbors, agg_type
        )
    )(
        np.arange(len(original_data))
    )
    return np.mean(sensitivities), np.std(sensitivities) / np.sqrt(sensitivities.size)


def compute_all_sensitivity(original_data, results, n_neighbors, agg_type="mean"):
    datasets = list(results.keys())
    methods = list(results[datasets[0]].keys())
    methods = [method for method in methods if not method.startswith("BATCH_")]

    agg_mean = {}
    agg_err = {}
    for dataset in original_data:
        dataset_name = dataset["name"]
        data_mean = {}
        data_err = {}

        for method in methods:
            try:  # TODO: REMOVE LATER; only for debugging

                scorer = dataset["scorer"]
                rankings = scores_to_ordering(scorer(dataset["data"][0]))

                res_ = sensitivity(
                    original_data=dataset["data"][0], 
                    contributions=results[dataset_name][method][0], 
                    rankings=rankings,
                    n_neighbors=n_neighbors,
                    agg_type=agg_type
                )

                data_mean[method] = res_[0]
                data_err[method] = res_[1]
            except:
                print(f"Error in dataset: {dataset_name}, with method: {method}")

        agg_mean[dataset_name] = pd.Series(data_mean)
        agg_err[dataset_name] = pd.Series(data_err)
    return agg_mean, agg_err