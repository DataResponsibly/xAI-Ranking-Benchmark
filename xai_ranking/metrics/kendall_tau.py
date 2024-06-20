import sys

sys.path.append("..")

from scipy.stats import kendalltau
import numpy as np
import pandas as pd
from xai_ranking.benchmarks import (
    human_in_the_loop,
    hierarchical_ranking_explanation,
    lime_experiment,
    shap_experiment,
    sharp_experiment,
    participation_experiment,
)
from xai_ranking.preprocessing import (
    preprocess_atp_data,
    preprocess_csrank_data,
    preprocess_higher_education_data,
)
from xai_ranking.datasets import (
    fetch_atp_data,
    fetch_csrank_data,
    fetch_higher_education_data,
    fetch_movers_data,
)
from xai_ranking.scorers import (
    atp_score,
    csrank_score,
    higher_education_score,
)

DATASETS = [
    {
        "name": "ATP",
        "data": fetch_atp_data().head(5),
        "preprocess": preprocess_atp_data,
        "scorer": atp_score,
    },
    {
        "name": "CSRank",
        "data": fetch_csrank_data().head(5),
        "preprocess": preprocess_csrank_data,
        "scorer": csrank_score,
    },
    {
        "name": "Higher Education",
        "data": fetch_higher_education_data(year=2021).head(5),
        "preprocess": preprocess_higher_education_data,
        "scorer": higher_education_score,
    },
]

RNG_SEED = 42


def compare_methods(method1, method2):
    """
    Compares two methods passed as function arguments
    with Kendall tau on all datasets for each data point
    """
    results = {}
    for dataset in DATASETS:
        results[dataset["name"]] = []
        preprocess_func = dataset["preprocess"]
        score_func = dataset["scorer"]
        X, ranks, scores = preprocess_func(dataset["data"])

        contributions1 = method1(X, score_func)
        contributions2 = method2(X, score_func)
        
        for data_point_idx, _ in enumerate(contributions1):
            statistic_result = kendalltau(contributions1[data_point_idx], contributions2[data_point_idx])
            results[dataset["name"]].append(statistic_result.statistic)

    return results


if __name__ == "__main__":
    print(compare_methods(human_in_the_loop, shap_experiment))
