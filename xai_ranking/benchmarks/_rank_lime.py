import numpy as np

from xai_ranking.benchmarks.rank_lime import RankingLIME


def rank_lime_experiment(X, score_function, **kwargs):
    """
    Parameters
    ----------
    X : array-like
        The input data for which the attributions are to be computed.
    score_function : callable
        The model or function used to score the input data.
    **kwargs : dict
        Additional keyword arguments to be passed to the RankingLIME constructor.

    Returns
    -------
    numpy.ndarray
        A 2D array where each element represents the attribution score for
        a specific feature in a specific document.
    """
    xai = RankingLIME(
        background_data=np.array(X), original_model=score_function, **kwargs
    )
    attributions = xai.get_doc_wise_attribution(np.array(X))
    result = np.empty((attributions[-1][0] + 1, attributions[-1][1]))
    for attribution in attributions:
        result[attribution[0], attribution[1] - 1] = attribution[-1]
    return result
