import pandas as pd
import numpy as np
import math
from os.path import dirname, abspath, join
from pathlib import Path


def fetch_synthetic_data(synth_dt_version=2, item_num=1000):
    # Feature names
    column_names = ["n1", "n2", "n3"]
    
    # Check if files exist, if not we will make them
    filepath = join(dirname(abspath(__file__)), "files", f"Synthetic_{synth_dt_version}_{item_num}.txt")

    if Path(filepath).is_file():
        df = pd.read_csv(
            filepath,
            delimiter=",",
            names=column_names,
            dtype=np.float64,
        )
    else:
        # Make index names
        ind = range(0, item_num)
    
        # Make features based on synthetic data version passed
        if synth_dt_version == 0:
            # All features are independent
            means = [0.5, 0.5, 0.5]
            var = [0.1, 0.1, 0.1]
            corr = -0.8
            covs = [[var[0], 0, 0], [0, var[1], 0], [0, 0, var[2]]]
            features = np.random.multivariate_normal(means, covs, item_num)
        elif synth_dt_version == 1:
            # Features 1 & 2 are negatively correlated
            means = [0.5, 0.5, 0.5]
            var = [0.1, 0.1, 0.1]
            corr = -0.8
            cov1_2 = math.sqrt(var[0]) * math.sqrt(var[1]) * corr
            covs = [[var[0], cov1_2, 0], [cov1_2, var[1], 0], [0, 0, var[2]]]
            features = np.random.multivariate_normal(means, covs, item_num)  
        elif synth_dt_version == 2:
            # Features 1 & 2 are negatively correlated
            # Feature 1 & 3 are positively correlated
            # Features 2 & 3 are negatively correlated
            means = [0.5, 0.5, 0.5]
            var = [0.1, 0.1, 0.1]
            corr = [-0.8, 0.6, -0.2]
            cov1_2 = math.sqrt(var[0]) * math.sqrt(var[1]) * corr[0]
            cov1_3 = math.sqrt(var[0]) * math.sqrt(var[2]) * corr[1]
            cov2_3 = math.sqrt(var[1]) * math.sqrt(var[2]) * corr[2]
            covs = [
                [var[0], cov1_2, cov1_3],
                [cov1_2, var[1], cov2_3],
                [cov1_3, cov2_3, var[2]],
            ]
            features = np.random.multivariate_normal(means, covs, item_num)
        else:
            return None
    
        # Make dataframe
        df = pd.DataFrame(features, columns=column_names, index=ind)
    
        # Normalize data
        for series_name, series in df.items():
            df[series_name] = (series - series.min()) / (series.max() - series.min())

        # Write to file
        df.to_csv(filepath, index=False, header=False)

    return df