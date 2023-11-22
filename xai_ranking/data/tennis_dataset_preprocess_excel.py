"Preprocesses ATP_info.xlsx - works with each tab and saves to csv"

import pandas as pd
import urllib.request
import os


def download_data(path):
    if not os.path.exists(path):
        os.makedirs(path)

    file_names = ['3.1_ATP_info.xlsx', 'heights_weights.csv']
    file_names += [f'final_rankings_{year}.csv' for year in range(2020, 2024)]

    for file in file_names:
        urllib.request.urlretrieve(f'https://zenodo.org/record/10108667/files/{file}', os.path.join(path, file))

    print("Files were successfully downloaded!")


def process(df, sheet_name):
    """
    This function replaces redundant characters in column names, 
    splits players' rank and names which were in one column in some datasets,
    and for all column names except `standing_player2` adds dataframe name to them 
    """
    df.columns = df.columns.str.replace(r'[^a-zA-Z0-9%_ ]', '', regex=True)
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('%', 'pct')
    df.columns = df.columns.str.replace('/', '_per_')
    df.columns = df.columns.str.replace('.', '')
    df.columns = df.columns.str.replace('__', '_')
    df.columns = df.columns.str.replace('Percentage', 'pct')
    df.columns = df.columns.str.lower()

    def split_mixed_values(x):
        if isinstance(x, str) and any(c.isalpha() for c in x) and any(c.isdigit() for c in x):
            numbers, names = x.split()[0], x.split()[1:]
            names = ' '.join([name for name in names])
            return pd.Series([names, numbers], index=['Names', 'Numbers'])
        return pd.Series([x, None], index=['Names', 'Numbers'])

    for col in df.columns:
        df_split = df[col].apply(split_mixed_values)
        if df_split['Numbers'].notna().any():
            df[['standing_player2', 'standing_player']] = df_split
            df = df.drop(columns=[col])
            break

    adj_columns = []
    df_name = sheet_name.split(' ')[:-1]
    df_name = '_'.join(i for i in df_name).lower()
    for col in df.columns:
        if 'standing_player2' in col:
            adj_columns.append('standing_player2')
        else:
            adj_columns.append(df_name + '__' + col)

    df.columns = adj_columns

    return df


def read_xls(excel_file='../../data/external/ATP_data/3.1_ATP_info.xlsx', path_save_csv='../../data/interim/'):
    """
    Reads `excel file` with data about tennis players, cleans the data,
    and saves each worksheet tab to csv file with path `path_save_csv`
    """
    xls = pd.ExcelFile(excel_file)

    for sheet_name in xls.sheet_names:
        df_raw = xls.parse(sheet_name)
        df = process(df_raw, sheet_name)
        csv_file = path_save_csv + f'{sheet_name.replace(" ", "_").lower()}.csv'
        df.to_csv(csv_file, index=False)



if __name__=="__main__":
    download_data(path='../../data/external/')

    path_xls_file = '../../data/external/ATP_data/3.1_ATP_info.xlsx'
    path_save_csv = '../../data/interim/'

    read_xls(path_xls_file, path_save_csv)
