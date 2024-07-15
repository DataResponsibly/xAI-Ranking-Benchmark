"""
Implementation of metrics for xAI performance analysis.
"""

from ._sensitivity import sensitivity, compute_all_sensitivity
from ._stability import stability, compute_all_stability
from ._agreement import kendall_agreement, jaccard_agreement, compute_all_agreement
from ._fidelity import fidelity, compute_all_fidelity


__all__ = [
    "sensitivity",
    "compute_all_sensitivity",
    "stability",
    "compute_all_stability",
    "kendall_agreement",
    "jaccard_agreement",
    "compute_all_agreement",
    "fidelity",
    "compute_all_fidelity"
]
