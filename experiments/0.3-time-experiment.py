import os
import sys

sys.path.append("..")

import tqdm
import time
from copy import deepcopy
import numpy as np
import pandas as pd

import itertools
import random

from sklearn.utils import check_random_state
from lightgbm import LGBMRanker
from sharp import ShaRP
from sharp.utils import scores_to_ordering
from xai_ranking.preprocessing import preprocess_higher_education_data
from xai_ranking.scorers import higher_education_score
from mlresearch.utils import check_random_states

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

RNG_SEED = 42
N_RUNS = 10

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

random_states = check_random_states(RNG_SEED, N_RUNS)

datasets = [
    {
        "name": "Higher Education",
        "data": preprocess_higher_education_data(
            fetch_higher_education_data(year=2020)
        ),
        "scorer": higher_education_score,
        "n_observations": 100,
    },
    {
        "name": "ATP",
        "data": preprocess_atp_data(fetch_atp_data()),
        "scorer": atp_score,
        "n_observations": 86,
    },
    {
        "name": "CSRank",
        "data": preprocess_csrank_data(fetch_csrank_data()),
        "scorer": csrank_score,
        "n_observations": 100,
    },
    # {
    #     "name": "Moving Company",
    #     "data": preprocess_movers_data(fetch_movers_data(test=True)),
    #     "scorer": model.predict,
    #     "n_observations": 100,
    # },
    {
        "name": "Synthetic_0",
        "data": preprocess_synthetic_data(
            fetch_synthetic_data(synth_dt_version=0, item_num=2000)
        ),
        "scorer": synthetic_equal_score_3ftrs,
        "n_observations": 100,
    },
    {
        "name": "Synthetic_1",
        "data": preprocess_synthetic_data(
            fetch_synthetic_data(synth_dt_version=1, item_num=2000)
        ),
        "scorer": synthetic_equal_score_3ftrs,
        "n_observations": 100,
    },
    {
        "name": "Synthetic_2",
        "data": preprocess_synthetic_data(
            fetch_synthetic_data(synth_dt_version=2, item_num=2000)
        ),
        "scorer": synthetic_equal_score_3ftrs,
        "n_observations": 100,
    },
]

approaches = ["rank", "rank_score", "pairwise-rank", "pairwise-rank_score"]

default_kwargs = {
    "measure": "shapley",
    "sample_size": None,
    "coalition_size": None,
    "replace": False,
    "n_jobs": 14,
}
# parameters_to_change = {
#     "coalition_size": [i for i in range(1, 7)],
#     "sample_size": [20, 50, 100, 250] + list(range(500, 2000, 500)),
#     "n_jobs": [1, 2, 4, 8, 16, 32, 48],
# }

parameters_to_change = {
    "coalition_size": [i for i in range(1, 7)],
    "sample_size": [20, 50, 100, 250] + list(range(500, 2000, 500)),
}

result_cols = (
    [
        "dataset",
        "n_observations",
        "approach",
        "parameter",
        "parameter_value",
        "avg_time",
    ]
    + [f"time_{i}" for i in range(N_RUNS)]
    + [f"agreement_kendall_{i}" for i in range(N_RUNS)]
    + [f"agreement_jaccard2_{i}" for i in range(N_RUNS)]
    + [f"agreement_euclidean_{i}" for i in range(N_RUNS)]
    + [f"fidelity_{i}" for i in range(N_RUNS)]
)

for dataset in datasets:
    result_df = []
    # Set up basic settings
    X = dataset["data"][0]

    # Get scores and ranks
    scorer = dataset["scorer"]
    scores = np.array(scorer(dataset["data"][0]))

    # Remove items that tie
    res = [idx for idx, val in enumerate(scores) if val in scores[:idx]]
    X = X.drop([X.index[i] for i in res])
    # Rescore, get rank
    scores = np.array(scorer(X))
    ranking = scores_to_ordering(scores)

    # Set experiment size if we deleted too many items
    dataset["n_observations"] = (
        dataset["n_observations"]
        if X.shape[0] > dataset["n_observations"]
        else X.shape[0]
    )

    rng = check_random_state(RNG_SEED)

    # rank and score indexes
    sam_idx = rng.choice(
        np.indices((X.shape[0],)).squeeze(),
        size=dataset["n_observations"],
        replace=False,
    )

    # pairwise pairs
    combos = list(itertools.combinations(np.indices((X.shape[0],)).squeeze(), 2))
    pairs_indexes = rng.choice(
        len(combos),
        size=dataset["n_observations"],
        replace=False,
    )
    pairs_sample = [combos[i] for i in pairs_indexes]
    pairs = [
        (pair[0], pair[1]) if np.random.choice([0, 1]) else (pair[1], pair[0])
        for pair in pairs_sample
    ]

    for approach in approaches:
        iteration_qoi = approach
        if approach.startswith("pairwise"):
            iteration_qoi = approach.split("-")[1]
            approach = "pairwise"
        print(
            "----------------",
            dataset["name"],
            "|",
            approach,
            "|",
            iteration_qoi,
            "----------------",
        )

        times = []
        kendall_cons = []
        jaccard_cons = []
        euclidean_cons = []
        fidelity = []

        print("Exact computation")
        for i in tqdm.tqdm(range(N_RUNS)):
            start = time.time()
            if approach != "pairwise":
                baseline_sharp = ShaRP(
                    qoi=iteration_qoi,
                    target_function=dataset["scorer"],
                    random_state=random_states[i],
                    **default_kwargs,
                )
                baseline_sharp.fit(X)
                sam_idx1 = sam_idx
                baseline_contr = baseline_sharp.all(X.values[sam_idx1])
            else:
                baseline_sharp = ShaRP(
                    qoi=iteration_qoi,
                    target_function=dataset["scorer"],
                    random_state=random_states[i],
                    **default_kwargs,
                )
                baseline_sharp.fit(X)
                baseline_pairwise = []
                sam_idx1 = [i[0] for i in pairs]
                sam_idx2 = [i[1] for i in pairs]
                for idx1, idx2 in pairs:
                    baseline_pairwise.append(
                        baseline_sharp.pairwise(X.values[idx1], X.values[idx2])
                    )
                baseline_contr = np.array(baseline_pairwise)

            end = time.time()

            baseline_contr = pd.DataFrame(
                baseline_contr, columns=X.columns, index=X.index.values[sam_idx1]
            )
            # Save metrics
            times.append(end - start)
            kendall_cons.append(np.nan)
            jaccard_cons.append(np.nan)
            euclidean_cons.append(np.nan)

            target = scores if approach == "rank_score" else ranking
            avg_target = target.mean()
            if approach != "pairwise":
                res_ = outcome_fidelity(
                    baseline_contr,
                    target[sam_idx1],
                    avg_target,
                    target_max=X.shape[0] if approach == "rank" else target.max(),
                    rank=approach == "rank",
                )
            else:
                res_ = outcome_fidelity(
                    baseline_contr,
                    target[sam_idx1],
                    avg_target,
                    target_max=X.shape[0] if approach == "rank" else target.max(),
                    target_pairs=target[sam_idx2],
                    rank=True,
                )

            fidelity.append(res_)

        exact_results_row = (
            [
                dataset["name"],
                dataset["n_observations"],
                approach + "_" + iteration_qoi,
                np.nan,
                np.nan,
                np.mean(times),
            ]
            + times
            + kendall_cons
            + jaccard_cons
            + euclidean_cons
            + fidelity
        )
        result_df.append(exact_results_row)
        print("Finished computing exact results")
        ############################################################################################

        for parameter, parameter_values in parameters_to_change.items():
            print(f"Alternating parameter: {parameter}")
            default_value = deepcopy(
                default_kwargs[parameter] if parameter in default_kwargs else None
            )

            if parameter == "coalition_size":
                parameter_values = [
                    val for val in parameter_values if X.shape[-1] > val
                ]
            if parameter == "sample_size":
                parameter_values = [
                    val for val in parameter_values if X.shape[0] >= val
                ] + [X.shape[0]]

            if approach == "pairwise" and parameter == "sample_size":
                continue

            for parameter_value in tqdm.tqdm(parameter_values):

                default_kwargs[parameter] = parameter_value

                times = []
                kendall_cons = []
                jaccard_cons = []
                euclidean_cons = []
                fidelity = []

                print(f"Parameter {parameter}, value {parameter_value}")
                for i in tqdm.tqdm(range(N_RUNS)):
                    start = time.time()
                    if approach != "pairwise":
                        sharp = ShaRP(
                            qoi=iteration_qoi,
                            target_function=dataset["scorer"],
                            random_state=random_states[i],
                            **default_kwargs,
                        )
                        sharp.fit(X)
                        sam_idx1 = sam_idx
                        contr = sharp.all(X.values[sam_idx1])
                    else:
                        sharp = ShaRP(
                            qoi=iteration_qoi,
                            target_function=dataset["scorer"],
                            random_state=random_states[i],
                            **default_kwargs,
                        )
                        sharp.fit(X)
                        pairwise = []
                        sam_idx1 = [i[0] for i in pairs]
                        sam_idx2 = [i[1] for i in pairs]
                        for idx1, idx2 in pairs:
                            pairwise.append(
                                sharp.pairwise(X.values[idx1], X.values[idx2])
                            )
                        contr = np.array(pairwise)

                    end = time.time()

                    contr = pd.DataFrame(
                        contr, columns=X.columns, index=np.array(X.index)[sam_idx1]
                    )

                    # Save metrics
                    times.append(end - start)
                    # Kendall consistency
                    kendall_cons.append(
                        cross_method_explanation_consistency(
                            contr, baseline_contr, measure="kendall"
                        )[0]
                    )
                    # Jaccard consistency
                    jaccard_cons.append(
                        cross_method_explanation_consistency(
                            contr, baseline_contr, measure="jaccard", n_features=2
                        )[0]
                    )
                    # Eulidean consistency
                    euclidean_cons.append(
                        cross_method_explanation_consistency(
                            contr,
                            baseline_contr,
                            measure="euclidean",
                            normalization=True,
                        )[0]
                    )
                    # Iniatialize normalizer
                    target = scores if approach == "rank_score" else ranking
                    avg_target = target.mean()
                    max_target = X.shape[0] if approach == "rank" else target.max()
                    # Fidelity
                    if approach != "pairwise":
                        res_ = outcome_fidelity(
                            contr,
                            target[sam_idx1],
                            avg_target,
                            target_max=max_target,
                            rank=approach == "rank",
                        )
                    else:
                        res_ = outcome_fidelity(
                            contr,
                            target[sam_idx1],
                            avg_target,
                            target_max=max_target,
                            target_pairs=target[sam_idx2],
                            rank=True,
                        )

                    fidelity.append(res_)

                results_row = (
                    [
                        dataset["name"],
                        dataset["n_observations"],
                        approach + "_" + iteration_qoi,
                        parameter,
                        parameter_value,
                        np.mean(times),
                    ]
                    + times
                    + kendall_cons
                    + jaccard_cons
                    + euclidean_cons
                    + fidelity
                )
                result_df.append(results_row)
                print(f"Stored results for {parameter} | {parameter_value}")

            default_kwargs[parameter] = default_value

    results = pd.DataFrame(result_df, columns=result_cols)
    results.to_csv("notebooks/results/time/time-experiment-" + dataset["name"] + ".csv")

results = pd.DataFrame(result_df, columns=result_cols)
results

metric = "exp_cons_kendall"
col_mask = results.columns.str.startswith(metric)
results[f"avg_{metric}"] = results.iloc[:, col_mask].mean(1)
col_mask = results.columns == f"avg_{metric}"
col_mask[:6] = True
results.iloc[:, col_mask]
