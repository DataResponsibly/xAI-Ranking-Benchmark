# Based on:
# https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Census%20income%20classification%20with%20scikit-learn.html

from shap import Explainer


def shap_experiment(X, score_function, **kwargs):
    explainer = Explainer(score_function, masker=X, **kwargs)
    shap_values = explainer(X)
    return shap_values.values
