import os
import sys

sys.path.append("..")

from itertools import product
from lightgbm import LGBMRanker
from sklearn.utils import check_random_state
from xai_ranking.benchmarks import (
    human_in_the_loop_experiment,
    hierarchical_ranking_explanation,
    lime_experiment,
    shap_experiment,
    sharp_experiment,
    # rank_lime_experiment,
    # participation_experiment,
)
from xai_ranking.preprocessing import (
    preprocess_atp_data,
    preprocess_csrank_data,
    preprocess_higher_education_data,
    preprocess_movers_data,
    preprocess_synthetic_data,
)
from xai_ranking.datasets import (
    fetch_atp_data,
    fetch_csrank_data,
    fetch_higher_education_data,
    fetch_movers_data,
    fetch_synthetic_data,
)
from xai_ranking.scorers import (
    atp_score,
    csrank_score,
    higher_education_score,
    synthetic_equal_score_3ftrs,
)
from xai_ranking.metrics import (
    explanation_sensitivity,
    outcome_sensitivity,
    bootstrapped_explanation_consistency,
    cross_method_explanation_consistency,
    cross_method_outcome_consistency,
    outcome_fidelity,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from mlresearch.utils import check_random_states, set_matplotlib_style
from sharp.utils import scores_to_ordering

RNG_SEED = 42

# Set up ranker for the moving company dataset:
X, ranks, score = preprocess_movers_data(fetch_movers_data(test=False))
qids_train = X.index.value_counts().to_numpy()

model = LGBMRanker(
    objective="lambdarank", label_gain=list(range(max(ranks) + 1)), verbose=-1
)
model.fit(
    X=X,
    y=ranks,
    group=qids_train,
)

datasets = [
    {
        "name": "ATP",
        "data": preprocess_atp_data(fetch_atp_data()),
        "scorer": atp_score,
    },
    {
        "name": "CSRank",
        "data": preprocess_csrank_data(fetch_csrank_data()),
        "scorer": csrank_score,
    },
    {
        "name": "Higher Education",
        "data": preprocess_higher_education_data(
            fetch_higher_education_data(year=2020)
        ),
        "scorer": higher_education_score,
    },
    # {
    #     "name": "Moving Company",
    #     "data": preprocess_movers_data(fetch_movers_data(test=True)),
    #     "scorer": model.predict,
    # },
    {
        "name": "Synthetic_0",
        "data": preprocess_synthetic_data(
            fetch_synthetic_data(synth_dt_version=0, item_num=2000)
        ),
        "scorer": synthetic_equal_score_3ftrs,
    },
    {
        "name": "Synthetic_1",
        "data": preprocess_synthetic_data(
            fetch_synthetic_data(synth_dt_version=1, item_num=2000)
        ),
        "scorer": synthetic_equal_score_3ftrs,
    },
    {
        "name": "Synthetic_2",
        "data": preprocess_synthetic_data(
            fetch_synthetic_data(synth_dt_version=2, item_num=2000)
        ),
        "scorer": synthetic_equal_score_3ftrs,
    },
]
xai_methods = [
    {
        "iterations": 1,
        "name": "LIME",
        "experiment": lime_experiment,
        "kwargs": {"mode": "regression"},  # classification, regression
    },
    {
        "iterations": 1,
        "name": "SHAP",
        "experiment": shap_experiment,
        "kwargs": {},
    },
    {
        "iterations": 1,
        "name": "ShaRP_RANK",
        "experiment": sharp_experiment,
        "kwargs": {
            "qoi": "rank",
            "verbose": True,
            "sample_size": None,
            "measure": "shapley",
            "n_jobs": -1,
            "replace": False,
        },
    },
    {
        "iterations": 1,
        "name": "ShaRP_SCORE",
        "experiment": sharp_experiment,
        "kwargs": {
            "qoi": "rank_score",
            "verbose": True,
            "sample_size": None,
            "measure": "shapley",
            "n_jobs": -1,
            "replace": False,
        },
    },
    # {
    #     "iterations": 1,
    #     "name": "ShaRP_TOPK",
    #     "experiment": sharp_experiment,
    #     "kwargs": {
    #         "qoi": "top_k",
    #         "verbose": True,
    #         "sample_size": None,
    #         "measure": "shapley",
    #         "n_jobs": -1,
    #         "replace": True,
    #     },
    # },
    {
        "iterations": 1,
        "name": "HRE_DT",
        "experiment": hierarchical_ranking_explanation,
        "kwargs": {"model_type": "DT", "s": 10},  # DT, LR, OLS, PLS
    },
    {
        "iterations": 1,
        "name": "HRE_LR",
        "experiment": hierarchical_ranking_explanation,
        "kwargs": {"model_type": "LR", "s": 10},  # DT, LR, OLS, PLS
    },
    {
        "iterations": 1,
        "name": "HRE_OLS",
        "experiment": hierarchical_ranking_explanation,
        "kwargs": {"model_type": "OLS", "s": 10},  # DT, LR, OLS, PLS
    },
    {
        "iterations": 1,
        "name": "HRE_PLS",
        "experiment": hierarchical_ranking_explanation,
        "kwargs": {"model_type": "PLS", "s": 10},  # DT, LR, OLS, PLS
    },
    # {
    #     "iterations": 1,
    #     "name": "HIL_Shapley",
    #     "experiment": human_in_the_loop_experiment,
    #     "kwargs": {"upper_bound": 1, "lower_bound": None, "method_type": "shapley"},
    # },
    {
        "iterations": 1,
        "name": "HIL_Standardized-Shapley",
        "experiment": human_in_the_loop_experiment,
        "kwargs": {
            "upper_bound": 1,
            "lower_bound": None,
            "method_type": "standardized shapley",
        },
    },
    {
        "iterations": 1,
        "name": "HIL_Rank-Shapley",
        "experiment": human_in_the_loop_experiment,
        "kwargs": {
            "upper_bound": 1,
            "lower_bound": None,
            "method_type": "rank-relevance shapley",
        },
    },
    # {
    #     "iterations": 1,
    #     "name": "RankLIME",
    #     "experiment": rank_lime_experiment,
    #     "kwargs": {
    #         "explanation_size": 3,
    #         "rank_similarity_coefficient": lambda x, y: kendalltau(x, y)[0],
    #         "individual_masking": True,
    #         "use_entry": 0,
    #         "use_pandas_where": False,
    #     },
    # },
    # {"iterations": 1, "name": "Participation", "experiment": participation_experiment},
]

total_states = sum(map(lambda x: x["iterations"], xai_methods)) * len(datasets)
random_states = (x for x in check_random_states(RNG_SEED, total_states))

# Uncomment to run full experiment

results = {}
for dataset in datasets:
    results[dataset["name"]] = {}
    for xai_method in xai_methods:
        results[dataset["name"]][xai_method["name"]] = []

        experiment_func = xai_method["experiment"]
        score_func = dataset["scorer"]

        X, ranks, scores = dataset["data"]

        for iteration_idx in range(xai_method["iterations"]):
            random_state = next(random_states)
            result_fname = (
                f"notebooks/results/contributions/_contributions_{dataset['name']}_"
                f"{xai_method['name']}_{iteration_idx}.csv"
            )

            if os.path.exists(result_fname):
                print(f"{result_fname} exists, skipping")
                continue
            if (
                xai_method["name"] in ("HRE_LR", "HRE_PLS")
                and dataset["name"] == "Moving Company"
            ):
                # dataset has binary categorical data
                # specified methods cannot handle such data
                continue
            if (
                xai_method["name"].startswith("HIL")
                and dataset["name"] == "Moving Company"
            ):
                # method requires weights, that are absent fot L2R
                continue

            kwargs = {} if "kwargs" not in xai_method else xai_method["kwargs"]
            if dataset["name"] == "Moving Company" and xai_method["name"].startswith(
                "ShaRP"
            ):
                kwargs["sample_size"] = 150

            contributions = experiment_func(
                X, score_func, random_state=random_state, **kwargs
            )

            results[dataset["name"]][xai_method["name"]].append(contributions)
            result_df = pd.DataFrame(contributions, columns=X.columns, index=X.index)
            result_df.to_csv(result_fname)