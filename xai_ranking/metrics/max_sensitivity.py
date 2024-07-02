import sys

sys.path.append("..")

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.stats import kendalltau
from xai_ranking.benchmarks import (
    human_in_the_loop,
    hierarchical_ranking_explanation,
    lime_experiment,
    shap_experiment,
    sharp_experiment,
    participation_experiment,
)
from xai_ranking.datasets import (
    fetch_atp_data,
    fetch_csrank_data,
    fetch_higher_education_data,
    fetch_movers_data
)
from xai_ranking.scorers import (
    atp_score,
    csrank_score,
    higher_education_score,
)
from xai_ranking.preprocessing import (
    preprocess_atp_data,
    preprocess_csrank_data,
    preprocess_higher_education_data,
)
from xai_ranking.metrics import kendall_tau

RNG_SEED = 42


def compute_sensitivity(df, target_idx, radius, explanator, score_function, distance_func):
    """
    Computes max sensitivity of explanation method
    for the point of interest based on points in its radius
    """
    target_point = df.iloc[target_idx].values
    target_point_idx = df.iloc[target_idx].name
    euclidean_distances = df.apply(lambda row: euclidean(target_point, row.values), axis=1)

    neighbors = df[euclidean_distances <= radius]
    # exclude the target point
    neighbors = neighbors[neighbors.index != target_point_idx]
    
    contributions = explanator(df, score_function)
    target_point_contri = contributions[target_idx]
    neighbors_indices = df.index.get_indexer(neighbors.index)
    neighbors_contributions = contributions[neighbors_indices]

    distances = np.array([distance_func(target_point_contri, contrib).statistic for contrib in neighbors_contributions])
    max_explanation_distance = np.max(distances)

    return max_explanation_distance


if __name__ == "__main__":
    target_idx = 2
    radius = 0.05
    data, _, _ = preprocess_atp_data(fetch_atp_data().head(5))
    print(compute_sensitivity(data, target_idx, radius, human_in_the_loop, atp_score, kendalltau))
