import os
import pandas as pd


def merge_by_year(folder_path, save_merged):
    """
    Gets path to folder with csv files, merges them by year,
    and writes to destination folder called save_merged
    """
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # dictionary with year as key, list of datasets about that year as value
    dfs_by_year = {}
    for csv_file in csv_files:
        year = csv_file.split('_')[-1][:4]
        df = pd.read_csv(os.path.join(folder_path, csv_file))
        dfs_by_year.setdefault(year, []).append(df)

    merged_dfs = {}
    for year, dataframes in dfs_by_year.items():
        merged_df = dataframes[0]
        if len(dataframes) > 1:
            for i, df in enumerate(dataframes[1:]):
                merged_df = pd.merge(merged_df, df, on='standing_player2', how='outer')

        merged_dfs[year] = merged_df

    for year, df in merged_dfs.items():
        df.to_csv(save_merged + f'merged_{year}.csv')



if __name__=="__main__":
    folder_path = '../../data/interim'
    save_merged = '../../data/processed/'
    merge_by_year(folder_path, save_merged)
