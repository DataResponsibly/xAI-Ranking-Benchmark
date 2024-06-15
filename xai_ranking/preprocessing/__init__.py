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
