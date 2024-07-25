import numpy as np


def outcome_fidelity(
    contributions, target, avg_target, dataset_size=1, target_pairs=None, rank=True
):
    if target_pairs is None:
        if rank:
            avg_est_err = (
                1
                - np.mean(np.abs(target - (avg_target - contributions.sum(axis=1))))
                / dataset_size
            )
        else:
            avg_est_err = np.mean(
                np.abs(target - (avg_target + contributions.sum(axis=1)))
            )
    else:
        if rank:
            better_than = target < target_pairs
        else:
            better_than = target > target_pairs

        est_better_than = contributions.sum(axis=1) > 0
        avg_est_err = (better_than == est_better_than).mean()
    return avg_est_err
