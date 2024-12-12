import numpy as np


# Reviewed
def outcome_fidelity(
    contributions, target, avg_target, target_max=1, target_pairs=None, rank=True
):
    """
    Calculates the outcome fidelity of contributions to a target, e.g.,
    how well the explanation matches the model prediction being explained.

    Returns
    -------
    avg_est_err : float
        The average estimation error.
    """
    if target_pairs is None:
        if rank:
            avg_est_err = (
                1
                - np.mean(np.abs(target - (avg_target - contributions.sum(axis=1))))
                / target_max
            )
        else:
            avg_est_err = np.mean(
                1
                - np.abs(target - (avg_target + contributions.sum(axis=1))) / target_max
            )
    else:
        if rank:
            better_than = target < target_pairs
        else:
            better_than = target > target_pairs

        est_better_than = contributions.sum(axis=1) > 0
        avg_est_err = (better_than == est_better_than).mean()
    return avg_est_err
