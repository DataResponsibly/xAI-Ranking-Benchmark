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

datasets = [
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
xai_methods = [
    {"name": "LIME", "experiment": lime_experiment},
    {"name": "SHAP", "experiment": shap_experiment},
    {"name": "ShaRP", "experiment": sharp_experiment},
    # {"name": "Participation", "experiment": participation_experiment},
    {"name": "HRE", "experiment": hierarchical_ranking_explanation},
    {"name": "HIL", "experiment": human_in_the_loop},
]

RNG_SEED = 42


def compare_():
    results = {}
    for dataset in datasets:
        results[dataset["name"]] = {}
        for xai_method in xai_methods:
            experiment_func = xai_method["experiment"]
            preprocess_func = dataset["preprocess"]
            score_func = dataset["scorer"]
            X, ranks, scores = preprocess_func(dataset["data"])
            contributions = experiment_func(X, score_func)
            results[dataset["name"]][xai_method["name"]] = contributions
            # with open(f"_contributions_{dataset['name']}_{xai_method['name']}.npy", "wb") as f:
            #     np.save(f, contributions)


if __name__ == "__main__":
    atp_data = fetch_atp_data(sheet_name='Serve 2022')
    print(atp_data.columns)
    # score_function = {"serve__pct_1st_serve": 0.5, "serve__pct_2nd_serve_points_won": 0.5}
    # compare_sharp_hilw(atp_data, score_function)
