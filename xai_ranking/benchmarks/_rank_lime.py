import numpy as np

from xai_ranking.benchmarks.rank_lime import RankingLIME


def rank_lime_experiment(X, score_function, **kwargs):
    xai = RankingLIME(background_data=np.array(X), original_model=score_function, **kwargs)
    attributions = xai.get_doc_wise_attribution(np.array(X))
    result = np.empty((attributions[-1][0] + 1, attributions[-1][1]))
    for attribution in attributions:
        result[attribution[0], attribution[1] - 1] = attribution[-1]
    return result
