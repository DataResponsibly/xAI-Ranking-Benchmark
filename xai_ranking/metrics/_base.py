import numpy as np
from scipy.stats import kendalltau
from scipy.spatial.distance import euclidean
from sharp.utils import scores_to_ordering


def _find_neighbors(original_data, rankings, contributions, row_idx, n_neighbors):
    row_data = np.array(original_data)[row_idx]
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
    return data_neighbors, cont_neighbors


def _get_importance_mask(row_cont, threshold):
    if threshold >= 1:
        row_res = scores_to_ordering(row_cont, direction=-1)
        mask = row_res <= threshold
    else:
        total_contribution = np.sum(np.abs(row_cont))
        order = np.argsort(np.abs(row_cont))
        original_order = np.argsort(order)
        cumulative_cont = np.cumsum(np.abs(row_cont)[order]) / total_contribution
        mask = (cumulative_cont < 1 - threshold)[original_order]

    return mask


def jaccard_similarity(a, b):
    intersection = len(list(set(a).intersection(b)))
    union = (len(set(a)) + len(set(b))) - intersection
    return float(intersection) / union


def row_wise_kendall(results1, results2):
    """
    Calculate the row-wise Kendall's tau correlation coefficient between two
    sets of contributions.

    Parameters
    ----------
    results1 : array-like
        The first set of contributions.
    results2 : array-like
        The second set of contributions.

    Returns
    -------
    float
        The row-wise Kendall's tau correlation coefficient.

    Notes
    -----
    The row-wise Kendall's tau correlation coefficient measures the similarity
    between two sets of rankings. It takes into account ties and is robust to
    outliers.

    Examples
    --------
    >>> results1 = [1, 2, 3, 4]
    >>> results2 = [4, 3, 2, 1]
    >>> row_wise_kendall(results1, results2)
    -1.0

    >>> results1 = [1, 2, 3, 4]
    >>> results2 = [1, 2, 3, 4]
    >>> row_wise_kendall(results1, results2)
    1.0
    """
    row_res1 = scores_to_ordering(results1, direction=1)
    row_res2 = scores_to_ordering(results2, direction=1)
    row_sensitivity = kendalltau(row_res1, row_res2).statistic
    return row_sensitivity


def row_wise_jaccard(results1, results2, n_features):
    """
    Calculate the row-wise Jaccard similarity between two sets of results.

    Parameters
    ----------
    results1 : numpy.ndarray
        The first set of results. It should be a 2-dimensional array with shape
        (n_samples, n_features).
    results2 : numpy.ndarray
        The second set of results. It should be a 2-dimensional array with shape
        (n_samples, n_features).
    n_features : int, float or None, default=0.8
        The number of top features to consider. If None, all features are
        considered. If an integer value is provided, only the top n_features
        features are considered. If n_features < 1, the most
        important features are determined based on their contribution to the
        total score (as a percentage of the total contribution in absolute
        values).

    Returns
    -------
    float
        The row-wise Jaccard similarity between the two sets of results.

    Notes
    -----
    The row-wise Jaccard similarity is calculated by first converting the
    results into rankings using the `scores_to_ordering` function. Then, the top
    n_features features are selected based on the rankings. Finally, the Jaccard
    similarity is calculated between the selected features for each row.

    If n_features is less than 1, the most important features are determined
    based on their contribution to the total score.  The cumulative contribution
    of each feature is calculated and the features are selected until the
    cumulative contribution exceeds 1 - n_features.

    Examples
    --------
    >>> results1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    >>> results2 = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
    >>> n_features = 2
    >>> row_wise_jaccard(results1, results2, n_features)
    """
    if n_features is None:
        n_features = results1.shape[1]

    mask1 = _get_importance_mask(results1, n_features)
    mask2 = _get_importance_mask(results2, n_features)

    top_idx1 = np.indices(results1.shape)[0, mask1]
    top_idx2 = np.indices(results1.shape)[0, mask2]
    row_similarity = jaccard_similarity(top_idx1, top_idx2)
    return row_similarity


def row_wise_euclidean(results1, results2):
    total1 = np.sum(np.abs(results1))
    total2 = np.sum(np.abs(results2))
    return euclidean(results1 / total1, results2 / total2)


def euclidean_agreement(results1, results2):
    """
    Calculate the Euclidean agreement between two sets of contributions across a
    dataset. Results are normalized, 0 means most similar and 1 means most
    dis-similar.

    Parameters
    ----------
    results1 : pandas.DataFrame
        The first set of contributions results.
    results2 : pandas.DataFrame
        The second set of contributions results.

    Returns
    -------
    pandas.Series
        A pandas Series containing the Euclidean agreement values for each pair
        of contributions vectors in `results1` and `results2`. The values are
        normalized between 0 and 1, where 0 means most similar and 1 means most
        dissimilar.

    Notes
    -----
    The Euclidean agreement is calculated by comparing each pair of contributions
    vectors in `results1` and `results2` using the Euclidean distance.
    """
    return results1.reset_index(drop=True).apply(
        lambda row: row_wise_euclidean(row, results2.iloc[row.name]), axis=1
    )


def kendall_agreement(results1, results2):
    """
    Calculate the Kendall agreement between two sets of contributions across a
    dataset. Results are normalized, 0 means most similar and 1 means most
    dis-similar.

    Parameters
    ----------
    results1 : pandas.DataFrame
        The first set of contributions results.
    results2 : pandas.DataFrame
        The second set of contributions results.

    Returns
    -------
    pandas.Series
        A pandas Series containing the Kendall agreement values for each pair
        of contributions vectors in `results1` and `results2`. The values are
        normalized between 0 and 1, where 0 means most similar and 1 means most
        dissimilar.

    Notes
    -----
    The Kendall agreement is calculated by comparing each pair of contributions
    vectors in `results1` and `results2` using the Kendall's tau correlation
    coefficient. The agreement is then averaged across all pairs of rankings.
    """
    return results1.reset_index(drop=True).apply(
        lambda row: (1 - row_wise_kendall(row, results2.iloc[row.name])) / 2, axis=1
    )


def jaccard_agreement(results1, results2, n_features=0.8):
    """
    Calculate the Jaccard similarity between two sets of results. Results are
    normalized, 0 means most similar and 1 means most dis-similar.

    Parameters
    ----------
    results1 : pandas.DataFrame
        The first set of results.
    results2 : pandas.DataFrame
        The second set of results.
    n_features : int, float or None, default=0.8
        The number of top features to consider. If None, all features are
        considered. If an integer value is provided, only the top n_features
        features are considered. If n_features < 1, the most
        important features are determined based on their contribution to the
        total score (as a percentage of the total contribution in absolute
        values).

    Returns
    -------
    pandas.Series
        The Jaccard agreement between each pair of results.

    Notes
    -----
    The Jaccard agreement is a measure of similarity between two sets of
    results. It is calculated as the average Jaccard similarity coefficient
    between each pair of results.
    """
    if n_features is None:
        n_features = results1.shape[1]

    return results1.reset_index(drop=True).apply(
        lambda row: 1 - row_wise_jaccard(row, results2.iloc[row.name], n_features),
        axis=1,
    )


_ROW_WISE_MEASURES = {
    "euclidean": row_wise_euclidean,
    "jaccard": row_wise_jaccard,
    "kendall": row_wise_kendall,
}

_MEASURES = {
    "euclidean": euclidean_agreement,
    "jaccard": jaccard_agreement,
    "kendall": kendall_agreement,
}
