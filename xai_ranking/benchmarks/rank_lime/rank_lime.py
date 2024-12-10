import numpy as np
from xai_ranking.utils import scores_to_ordering
from scipy.stats import kendalltau
from functools import partial
import lime
import random


def new_model_predict_lime(
    array,
    original_model_predict,
    query_features,
    similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
):
    # Determine ranking for current query
    pred = original_model_predict(query_features)
    og_rank = scores_to_ordering(pred)

    # Adjust feature vectors: Add perturbation array to all documents
    adjusted_features = np.array([[a + doc for doc in query_features] for a in array])

    scores = []
    for features_background_sample in adjusted_features:
        # Determine ranking for adjusted document feature vectors
        new_pred = original_model_predict(features_background_sample)
        new_rank = scores_to_ordering(new_pred)
        scores.append(similarity_coefficient(og_rank, new_rank))

    return np.array(scores)


def new_model_predict_lime_individual_masking(
    array,
    original_model_predict,
    query_features,
    similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
):
    # Determine ranking for current query
    pred = original_model_predict(query_features)
    og_rank = scores_to_ordering(pred)

    # Adjust feature vectors: Add perturbation array to all documents
    adjusted_features = [a.reshape(query_features.shape) for a in array]

    scores = []
    for features_background_sample in adjusted_features:
        # Determine ranking for adjusted document feature vectors
        new_pred = original_model_predict(features_background_sample)
        new_rank = scores_to_ordering(new_pred)
        scores.append(similarity_coefficient(og_rank, new_rank))

    return np.array(scores)


class RankingLIME:
    def __init__(
        self,
        background_data,
        original_model,
        explanation_size=3,
        name="",
        rank_similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
        individual_masking=True,
        use_entry=0,
        use_pandas_where=False,
        **kwargs
    ):
        self.background_data = background_data
        self.original_model = original_model
        self.explanation_size = explanation_size
        self.use_pandas_where = use_pandas_where

        self.name = name

        self.feature_shape = np.shape(background_data[0])
        self.num_features = len(background_data[0])
        self.individual_masking = individual_masking
        self.use_entry = use_entry

        self.explainer = self.get_explainer()
        self.rank_similarity_coefficient = rank_similarity_coefficient
        if not individual_masking:
            self.new_model_predict = partial(
                new_model_predict_lime,
                original_model_predict=original_model,
                similarity_coefficient=rank_similarity_coefficient,
            )
        else:
            self.new_model_predict = partial(
                new_model_predict_lime_individual_masking,
                original_model_predict=original_model,
                similarity_coefficient=rank_similarity_coefficient,
            )
        self.feature_attribution_explanation = None
        self.feature_selection_explanation = None

    def get_explainer(self, num_candidates=None):
        if self.individual_masking:
            background_data = np.array(
                [
                    np.array(
                        random.choices(self.background_data, k=num_candidates)
                    ).flatten()
                    for _ in range(len(self.background_data))
                ]
            )

            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                background_data,
                feature_names=[i for i in range(0, self.num_features * num_candidates)],
                mode="regression",
            )
        else:
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.background_data,
                feature_names=[i for i in range(0, self.num_features)],
                mode="regression",
            )
        return lime_explainer

    def get_doc_wise_attribution(self, query_features, query_id=""):
        if self.individual_masking:
            new_model_predict = partial(
                self.new_model_predict, query_features=query_features
            )
            query_features_flattened = query_features.flatten()
            explainer = self.get_explainer(num_candidates=len(query_features))
            exp = explainer.explain_instance(
                query_features_flattened,
                new_model_predict,
                num_features=len(query_features_flattened),
            )
            exp = exp.local_exp[self.use_entry]
            exp = sorted(exp, key=lambda x: x[0])
            exp = [e[1] for e in exp]
            exp = np.array(exp).reshape(query_features.shape)
            exp = [
                (doc_num, feature + 1, value)
                for doc_num, doc in enumerate(exp)
                for feature, value in enumerate(doc)
            ]
        else:
            zero_vector = np.array(np.full(self.feature_shape, 0))

            new_model_predict = partial(
                self.new_model_predict, query_features=query_features
            )

            exp = self.explainer.explain_instance(
                zero_vector, new_model_predict, num_features=self.num_features
            )
            exp = exp.local_exp[self.use_entry]
            exp = [
                (feature + 1, value) for (feature, value) in exp
            ]  # We start counting features at 1

        return exp
