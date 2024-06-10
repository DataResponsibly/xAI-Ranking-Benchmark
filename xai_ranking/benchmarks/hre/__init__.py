"""
All of the code in this directory is based on the code from the paper:

    Anahideh, H., & Mohabbati-Kalejahi, N. (2022). Local explanations of global
    rankings: insights for competitive rankings. IEEE Access, 10, 30676-30693.

The original code/implementation can be found at
https://github.com/anahideh/Ranking-Explanation/blob/main/ranking_code.ipynb

The code in this directory is a modified version of the original code to ensure
it is dataset-agnostic.
"""

from .anahideh import (
    feature_importance_DT,
    feature_importance_LR,
    feature_importance_OLS,
    feature_importance_PLS,
)

__all__ = [
    "feature_importance_DT",
    "feature_importance_LR",
    "feature_importance_OLS",
    "feature_importance_PLS",
]
