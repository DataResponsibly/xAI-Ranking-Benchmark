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


def compute_sensitivity(features_df, contri_df, target_idx, num_neighbors, score_function, distance_func):
    """
    Computes max sensitivity of explanation method
    for the point of interest based on points in its radius
    """
    target_point = features_df.iloc[target_idx].values
    target_point_idx = features_df.iloc[target_idx].name   # actual index in df

    # Define the range around the target point
    start_idx = max(0, target_idx - 2*num_neighbors)
    end_idx = min(len(features_df), target_idx + 2*num_neighbors + 1)  # +1 to include the target point itself

    features_df_subset = features_df.iloc[start_idx:end_idx]

    euclidean_distances = features_df_subset.apply(lambda row: euclidean(target_point, row.values), axis=1)
    neighbors_indices = euclidean_distances.nsmallest(num_neighbors+1).index     # actual indexes in df
    neighbors_indices = neighbors_indices[neighbors_indices != target_point_idx]

    neighbors_contributions = contri_df.loc[neighbors_indices].values
    target_point_contri = contri_df.loc[target_point_idx].values

    distances = np.array([distance_func(target_point_contri, contrib).statistic for contrib in neighbors_contributions])
    max_explanation_distance = np.max(distances)

    return max_explanation_distance


if __name__ == "__main__":
    target_idx = 2
    num_neighbors = 2
    features_data, _, _ = preprocess_atp_data(fetch_atp_data().head(10))
    contri_data = pd.read_csv("../../notebooks/results/_contributions_ATP_HIL.csv", index_col="player_name")
    print(compute_sensitivity(features_data, contri_data, target_idx, num_neighbors, atp_score, kendalltau))
