from sharp.utils import scores_to_ordering
from xai_ranking.scorers import synthetic_equal_score_3ftrs


def preprocess_atp_data(df):
    """
    Preprocess ATP data.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing ATP data.

    Returns
    -------
    X : pandas.DataFrame
        The preprocessed dataframe
    ranks : pandas.Series
        The series containing ranks for each player.
    scores : pandas.Series
        The series containing scores for each player.
    """
    X = df.drop(columns=["serve__standing_player", "serve__rating"])
    pct_cols = X.columns.str.contains("pct")
    X.iloc[:, ~pct_cols] = X.iloc[:, ~pct_cols] / 100
    ranks = df.serve__standing_player
    scores = df.serve__rating
    return X, ranks, scores


def preprocess_csrank_data(df):
    """
    Preprocess CS Rankings data.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing CS Rankings data.

    Returns
    -------
    X : pandas.DataFrame
        The preprocessed dataframe
    ranks : pandas.Series
        The series containing ranks for each department.
    scores : pandas.Series
        The series containing scores for each department.
    """
    X = df.drop(columns=["Rank", "Score"])
    X = X.iloc[:, X.columns.str.contains("Count")]
    # X["Faculty"] = df["Faculty"]
    X = X / X.max()
    ranks = df.Rank
    scores = df.Score
    return X, ranks, scores


def preprocess_higher_education_data(df):
    """
    Preprocess Times Higher Education data.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing Times Higher Education data.

    Returns
    -------
    X : pandas.DataFrame
        The preprocessed dataframe
    ranks : pandas.Series
        The series containing ranks for each university.
    scores : pandas.Series
        The series containing scores for each university.
    """
    X = df.drop(columns=["world_rank", "total_score", "country", "year"])
    X = X / 100
    ranks = df.world_rank
    scores = df.total_score
    return X, ranks, scores


def preprocess_movers_data(df):
    """
    Preprocess fictional data on applications for a moving company.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing the data.

    Returns
    -------
    X : pandas.DataFrame
        The preprocessed dataframe
    ranks : pandas.Series
        The series containing ranks for each row.
    scores : pandas.Series
        The series containing scores for each row.
    """
    X = df.drop(columns=["relevance", "qualification_score"])
    ranks = scores_to_ordering(df["qualification_score"])
    scores = df["qualification_score"]

    data = X.weight_lifting_ability
    X.weight_lifting_ability = (data - data.min()) / (data.max() - data.min())
    return X, ranks, scores


def preprocess_synthetic_data(df):
    """
    Preprocess synthetic data.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing synthetic data.

    Returns
    -------
    X : pandas.DataFrame
        The preprocessed dataframe
    ranks : pandas.Series
        The series containing ranks for each row.
    scores : pandas.Series
        The series containing scores for each row.
    """
    X = df
    scores = synthetic_equal_score_3ftrs(X)
    ranks = scores_to_ordering(scores)
    return X, ranks, scores
