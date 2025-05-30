"""
Moving company dataset (aka "movers" data) from Yang, Ke, Joshua R. Loftus, and
Julia Stoyanovich. "Causal intersectionality for fair ranking." arXiv preprint
arXiv:2006.08688 (2020).
"""

from os.path import dirname, abspath, join
import pandas as pd


def fetch_movers_data(test=False, fair="biased"):
    """
    Fetches a dataset with fictional info on applications for a moving company
    taken from:

        Yang, Ke, Joshua R. Loftus, and Julia Stoyanovich. "Causal
        intersectionality for fair ranking." arXiv preprint arXiv:2006.08688
        (2020).

    The "Y" feature shown in the paper is renamed to "qualification_score" and
    the "X" feature is renamed to "weight_lifting_ability".

    The target variable is "qualification_score" or "relevance". Other variables
    are encoded as follows:
    - Gender: 0 -> Male, 1 -> Female
    - Race: 0 -> White, 1 -> Black

    Parameters
    ----------
    test : bool, optional
        If True, fetches the test set. Otherwise, fetches the training set.
        Default is False.

    fair : str, optional
        If "fair", fetches the debiased version of the dataset. If "partial", fetches
        the partially debiased version (gender is debiased, race is not). If "biased",
        fetches the biased dataset.

    Returns
    -------
    pd.DataFrame
        The processed moving company data.
    """
    split = "test" if test else "train"
    if fair == "biased":
        filename = f"R10_{split}_ranklib.txt"
    elif fair == "partial":
        filename = f"fair_count__bias__R10_{split}_ranklib.txt"
    elif fair == "fair":
        filename = f"fair_count__fair_count__R10_{split}_ranklib.txt"

    filepath = join(dirname(abspath(__file__)), "files", filename)
    df = pd.read_csv(
        filepath,
        delimiter=" ",
        names=["relevance", "qid", "gender", "race", "X", "Y", "meta"],
    )
    df.drop(columns=["meta"], inplace=True)

    df["qid"] = df["qid"].str.replace("qid:", "")
    df.set_index("qid", inplace=True)

    df["gender"] = df["gender"].str.replace("1:", "")
    df["race"] = df["race"].str.replace("2:", "")
    df["X"] = df["X"].str.replace("3:", "")
    df["Y"] = df["Y"].str.replace("4:", "")
    df = df.astype({"gender": int, "race": int, "X": float, "Y": float})
    df.rename(
        columns={
            "Y": "qualification_score",
            "X": "weight_lifting_ability",
            # "meta": "metadata",
        },
        inplace=True,
    )
    return df
