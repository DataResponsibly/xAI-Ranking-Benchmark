import sys

sys.path.append("..")

from scipy.stats import kendalltau
import pandas as pd

RNG_SEED = 42


def kendall_tau(results_df1, results_df2):
    """
    Compares two methods results passed as function arguments
    with Kendall tau for each data point
    """
    results = []
    for row in results_df1.iterrows():
        data_point_idx = row[0]
        statistic_result = kendalltau(
            results_df1.iloc[data_point_idx].values[1:],
            results_df2.iloc[data_point_idx].values[1:],
        )
        results.append(statistic_result.statistic)
    return results


if __name__ == "__main__":
    hilw_results = pd.read_csv("../../notebooks/results/_contributions_ATP_HIL.csv")
    sharp_results = pd.read_csv("../../notebooks/results/_contributions_ATP_ShaRP.csv")
    print(len(kendall_tau(hilw_results, sharp_results)))
