# License: MIT
# Authors: - Joao Fonseca <jpfonseca@novaims.unl.pt>
#          - Kateryna Akhynko <kateryna.akhynko@ucu.edu.ua>

from urllib.parse import urljoin
import numpy as np
import pandas as pd


BASE_URL = "https://zenodo.org/record/10245175/files/"


def fetch_atp_data(
    file="3.1_ATP_info.xlsx", sheet_name="Serve 2022", add_heights_weights=False
):
    """
    Loads ATP data into memory stored by the link in the Notes section. It is possible
    to load only one file at a time. If the file is an Excel file, the sheet name must be
    specified.
    Coefficients for "Serve 2022" scoring function were inferred from the data:
    [100, 100, 100, 100, 1, -1]

    Parameters
    ----------
    file : str, optional
        The name of the file to load data from. Default is "3.1_ATP_info.xlsx".
    sheet_name : str, optional
        The name of the sheet to load data from if the file is an Excel file. Default is "Serve 2022".
    add_heights_weights : bool, optional
        If True, additional height and weight information will be added to the data. Default is False.
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the loaded data, indexed by player name.
    Raises
    ------
    ValueError
        If the specified sheet name is not found in the Excel file.
    TypeError
        If the file extension is not recognized.
    Notes
    -----
    https://zenodo.org/records/10245175 : link to the data source.
    """

    if file.endswith(".xlsx"):
        xl = pd.ExcelFile(urljoin(BASE_URL, file))
        all_sheets = xl.sheet_names

        if sheet_name is None or sheet_name not in all_sheets:
            raise ValueError(
                f"Sheet name `{sheet_name}` not found. Select one of {all_sheets}."
            )

        df = xl.parse(sheet_name)

    elif file.endswith(".csv"):
        df = pd.read_csv(urljoin(BASE_URL, file))
    else:
        ext = file.split(".")[-1]
        raise TypeError(f"unrecognized file extension `.{ext}`.")

    df = preprocess(df, sheet_name)

    if add_heights_weights:
        df_info = pd.read_csv(urljoin(BASE_URL, "heights_weights.csv"))
        df_info.rename(columns={"standing_player2": "player_name"}, inplace=True)

        df_info.set_index(
            df_info["player_name"].apply(lambda x: x.lower().strip().replace(" ", "")),
            inplace=True,
        )
        df_info.drop(columns="player_name", inplace=True)
        df.set_index(
            df["player_name"].apply(lambda x: x.lower().strip().replace(" ", "")),
            inplace=True,
        )

        df = df.join(df_info)

        # Find values not added with join (related to typos and name ordering)
        df["__last_name"] = df["player_name"].apply(lambda x: x.split(" ")[-1].lower())
        idx_nan = df.index[df.isna().any(axis=1)]
        for attr in ["height_cm", "weight_kg"]:
            df.loc[idx_nan, attr] = df.loc[idx_nan].apply(
                lambda row: _parse_missing_height_weights(row, df_info, attr), axis=1
            )
        df.drop(columns="__last_name", inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df.set_index("player_name")


def _parse_missing_height_weights(row, df_info, attr):
    value = df_info.loc[df_info.index.str.contains(row["__last_name"]), attr].values
    if len(value) == 1:
        return float(value[0])
    else:
        return np.nan


def preprocess(df, sheet_name):
    """
    This function replaces redundant characters in column names,
    splits players' rank and names which were in one column in some datasets,
    and for all column names except `standing_player2` adds dataframe name to them.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to preprocess.
    sheet_name : str
        The name of the sheet from which the dataframe was read.

    Returns
    -------
    pandas.DataFrame
        The preprocessed dataframe with cleaned and adjusted column names.
    """
    df.columns = df.columns.str.replace(r"[^a-zA-Z0-9%_ ]", "", regex=True)
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.replace("%", "pct")
    df.columns = df.columns.str.replace("/", "_per_")
    df.columns = df.columns.str.replace(".", "")
    df.columns = df.columns.str.replace("__", "_")
    df.columns = df.columns.str.replace("Percentage", "pct")
    df.columns = df.columns.str.lower()

    def split_mixed_values(x):
        if (
            isinstance(x, str)
            and any(c.isalpha() for c in x)
            and any(c.isdigit() for c in x)
        ):
            numbers, names = x.split()[0], x.split()[1:]
            names = " ".join([name for name in names])
            return pd.Series([names, numbers], index=["Names", "Numbers"])
        return pd.Series([x, None], index=["Names", "Numbers"])

    for col in df.columns:
        df_split = df[col].apply(split_mixed_values)
        if df_split["Numbers"].notna().any():
            df[["standing_player2", "standing_player"]] = df_split
            df = df.drop(columns=[col])
            break

    adj_columns = []
    df_name = sheet_name.split(" ")[:-1]
    df_name = "_".join(i for i in df_name).lower()
    for col in df.columns:
        if "standing_player2" in col:
            adj_columns.append("player_name")
        else:
            adj_columns.append(df_name + "__" + col)

    df.columns = adj_columns

    return df
