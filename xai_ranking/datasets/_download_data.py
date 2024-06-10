import pandas as pd


CSRANKS_URL = "https://zenodo.org/records/11234896/files/csrankings_raw.csv"
TIMES_URL = "https://zenodo.org/records/11235321/files/times2-revised.csv"


def fetch_csrank_data():
    return (
        pd.read_csv(CSRANKS_URL)
        .drop(columns="Unnamed: 0")
        .rename(columns={"Count": "Rank"})
    )


def fetch_higher_education_data(year=None):
    df = pd.read_csv(TIMES_URL)

    if year is not None:
        df = df[df["year"] == year].copy()

    return df
