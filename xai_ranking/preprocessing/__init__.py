from sharp.utils import scores_to_ordering
from xai_ranking.scorers import synthetic_equal_score_3ftrs


def preprocess_atp_data(df):
    X = df.drop(columns=["serve__standing_player", "serve__rating"])
    pct_cols = X.columns.str.contains("pct")
    X.iloc[:, ~pct_cols] = X.iloc[:, ~pct_cols] / 100
    ranks = df.serve__standing_player
    scores = df.serve__rating
    return X, ranks, scores


def preprocess_csrank_data(df):
    X = df.drop(columns=["Rank", "Score"])
    X = X.iloc[:, X.columns.str.contains("Count")]
    X["Faculty"] = df["Faculty"]
    X = X / X.max()
    ranks = df.Rank
    scores = df.Score
    return X, ranks, scores


def preprocess_higher_education_data(df):
    X = df.drop(columns=["world_rank", "total_score", "country", "year"])
    X = X / 100
    ranks = df.world_rank
    scores = df.total_score
    return X, ranks, scores


def preprocess_movers_data(df):
    X = df.drop(columns=["relevance", "qualification_score"])
    ranks = scores_to_ordering(df["qualification_score"])
    scores = df["qualification_score"]

    data = X.weight_lifting_ability
    X.weight_lifting_ability = (data - data.min()) / (data.max() - data.min())
    return X, ranks, scores

def preprocess_synthetic_data(df):
    X = df
    scores = synthetic_equal_score_3ftrs(X)
    ranks = scores_to_ordering(scores)
    return X, ranks, scores
