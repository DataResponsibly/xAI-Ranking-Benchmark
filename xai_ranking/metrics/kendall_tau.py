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
        "data": fetch_atp_data().head(20),
        "preprocess": preprocess_atp_data,
        "scorer": atp_score,
    },
    {
        "name": "CSRank",
        "data": fetch_csrank_data().head(20),
        "preprocess": preprocess_csrank_data,
        "scorer": csrank_score,
    },
    {
        "name": "Higher Education",
        "data": fetch_higher_education_data(year=2021).head(20),
        "preprocess": preprocess_higher_education_data,
        "scorer": higher_education_score,
    },
]

RNG_SEED = 42


def compare_methods(results_df1, results_df2):
    """
    Compares two methods passed as function arguments
    with Kendall tau on all datasets for each data point
    """
    results = []        
    for row in results_df1.iterrows():
        data_point_idx = row[0]
        statistic_result = kendalltau(results_df1.iloc[data_point_idx].values[1:], results_df2.iloc[data_point_idx].values[1:])
        results.append(statistic_result.statistic)
    return results


if __name__ == "__main__":
    import pandas as pd
    hilw_results = pd.read_csv("../../notebooks/results/_contributions_ATP_HIL.csv")
    sharp_results = pd.read_csv("../../notebooks/results/_contributions_ATP_ShaRP.csv")
    print(len(compare_methods(hilw_results, sharp_results)))
