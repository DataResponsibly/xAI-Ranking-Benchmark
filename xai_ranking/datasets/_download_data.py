import pandas as pd


CSRANKS_URL = "https://zenodo.org/records/11234896/files/csrankings_raw.csv"
TIMES_URL = "https://zenodo.org/records/11235321/files/times2-revised.csv"


def fetch_csrank_data():
    """
    Fetches and processes the CSRankings data.
    Reads the data from a CSV file located at the CSRANKS_URL.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the processed CSRankings data with "Institution" as the index.
    """
    return (
        pd.read_csv(CSRANKS_URL)
        .drop(columns="Unnamed: 0")
        .rename(columns={"Count": "Rank"})
    ).set_index("Institution")


def fetch_higher_education_data(year=None):
    """
    Fetches and processes Times Higher Education data.
    Reads the data from a CSV file located at the TIMES_URL.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the processed Times Higher Education data
        with "university_name" as the index.
    """
    df = pd.read_csv(TIMES_URL)

    if year is not None:
        df = df[df["year"] == year].copy()

    return df.set_index("university_name")
