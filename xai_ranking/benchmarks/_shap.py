# Based on:
# https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Census%20income%20classification%20with%20scikit-learn.html
import numpy
from shap import Explainer


def shap_experiment(X, score_function, **kwargs):
    explainer = Explainer(score_function, masker=X, **kwargs)
    shap_values = explainer(X)
    return shap_values.values


def shap_batch_experiment(X, score_function, batch_size=10, random_state=42):
    batch = numpy.random.RandomState(random_state).choice(X, batch_size)
    explainer = Explainer(score_function, masker=batch)
    shap_values = explainer(X)
    return shap_values.values
