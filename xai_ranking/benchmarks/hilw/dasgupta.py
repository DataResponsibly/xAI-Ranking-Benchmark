"""
Module with all methods from Dasgupta paper
"""

import pandas as pd

# import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sharp.utils import scores_to_ordering


def hilw_contributions(df, score_function, upper_bound, lower_bound):
    """
    Based on Dasgupta's original implementation.

    input: df_all, weight
    hilw contributions for the entire population (no groupings, no batches).
    """

    df = df.copy()

    features = df.columns

    df["rank"] = scores_to_ordering(score_function(df))

    # dff = pd.DataFrame()
    # grouped = df.groupby(group_feature)

    # print(features)
    avg_attributes = dict()
    for attr in features:
        avg_attributes[attr + "_avg"] = df.loc[:, attr].mean()
        df[attr + "_contri"] = df[attr] - avg_attributes[attr + "_avg"]

    # use topN to subset the data
    df = df.query(f"{upper_bound} <= rank <= {lower_bound}")
    # grouped = dff.groupby(group_feature)

    contri_attributes = [x for x in df.columns if str(x).endswith("_contri")]

    # df_mean_contri = pd.DataFrame(index=contri_attributes)

    mean_contri = [df[attr].abs().tolist() for attr in contri_attributes]
    df_mean_contri = pd.DataFrame(data=mean_contri, index=contri_attributes).T

    # if num_batches == 1:
    #     df_mean_contri = transform_df(df_mean_contri)
    #     df_mean_contri_privileged = transform_df(df_mean_contri_privileged)
    #     df_mean_contri_protected = transform_df(df_mean_contri_protected)

    return df_mean_contri


def transform_df(df):
    new_columns = pd.DataFrame(df.iloc[:, 0].tolist())
    new_columns.columns = range(1, len(new_columns.columns) + 1)
    new_columns = new_columns.set_index(df.index)
    return new_columns


def shapley_values(
    d, weights, upper_bound, lower_bound, features, num_batches, group_feature="_N"
):
    """
    input: df_all, weight
    """
    df = d.copy()
    dff = pd.DataFrame()
    grouped = df.groupby(group_feature)

    for n, group in grouped:
        # group[[x for x in features]] = MinMaxScaler().fit_transform(group[[x for x in features]])
        avg_attributes = dict()
        for attr in features:
            avg_attributes[attr + "_avg"] = group.loc[:, attr].mean()
            group[attr + "_contri"] = weights[attr] * (
                group[attr] - avg_attributes[attr + "_avg"]
            )

            dff = pd.concat([dff, group], axis=0)

    ### use topN to subset the data
    dff = dff.query(f"{upper_bound} <= rank <= {lower_bound}")
    grouped = dff.groupby(group_feature)

    contri_attributes = [x for x in dff.columns if "contri" in x]

    df_mean_contri = pd.DataFrame(index=contri_attributes)
    df_mean_contri_privileged = pd.DataFrame(index=contri_attributes)
    df_mean_contri_protected = pd.DataFrame(index=contri_attributes)

    for n, group in grouped:
        if num_batches == 1:
            mean_contri = [group[attr].abs().tolist() for attr in contri_attributes]
            privileged_mean_contri = [
                group.query('Group == "privileged"')[attr].abs().tolist()
                for attr in contri_attributes
            ]
            protected_mean_contri = [
                group.query('Group == "protected"')[attr].abs().tolist()
                for attr in contri_attributes
            ]
        else:
            mean_contri = [group[attr].abs().mean() for attr in contri_attributes]
            privileged_mean_contri = [
                group.query('Group == "privileged"')[attr].abs().mean()
                for attr in contri_attributes
            ]
            protected_mean_contri = [
                group.query('Group == "protected"')[attr].abs().mean()
                for attr in contri_attributes
            ]

        df_mean_contri[n] = mean_contri
        df_mean_contri_privileged[n] = privileged_mean_contri
        df_mean_contri_protected[n] = protected_mean_contri

    if num_batches == 1:
        df_mean_contri = transform_df(df_mean_contri)
        df_mean_contri_privileged = transform_df(df_mean_contri_privileged)
        df_mean_contri_protected = transform_df(df_mean_contri_protected)

    return df_mean_contri, df_mean_contri_privileged, df_mean_contri_protected


### standardized shapley values
def competing_powers(
    d, weights, upper_bound, lower_bound, features, num_batches, group_feature="_N"
):
    """
    input: df_all, weights
    """
    df = d.copy()
    dff = pd.DataFrame()
    grouped = df.groupby(group_feature)

    for n, group in grouped:
        # group[['Attribute 1', 'Attribute 2']] = MinMaxScaler().fit_transform(group[['Attribute 1', 'Attribute 2']])
        score_sum = group.loc[:, "score"].sum()

        for attr in features:
            group[attr + "_contri"] = weights[attr] * group[attr] / score_sum
            dff = pd.concat([dff, group], axis=0)

    ### use topN to subset the data
    dff = dff.query(f"{upper_bound} <= rank <= {lower_bound}")
    grouped = dff.groupby(group_feature)

    contri_attributes = [x for x in dff.columns if "contri" in x]

    df_mean_contri = pd.DataFrame(index=contri_attributes)
    df_mean_contri_privileged = pd.DataFrame(index=contri_attributes)
    df_mean_contri_protected = pd.DataFrame(index=contri_attributes)

    for n, group in grouped:
        if num_batches == 1:
            mean_contri = [group[attr].abs().tolist() for attr in contri_attributes]
            privileged_mean_contri = [
                group.query('Group == "privileged"')[attr].abs().tolist()
                for attr in contri_attributes
            ]
            protected_mean_contri = [
                group.query('Group == "protected"')[attr].abs().tolist()
                for attr in contri_attributes
            ]
        else:
            mean_contri = [group[attr].abs().mean() for attr in contri_attributes]
            privileged_mean_contri = [
                group.query('Group == "privileged"')[attr].abs().mean()
                for attr in contri_attributes
            ]
            protected_mean_contri = [
                group.query('Group == "protected"')[attr].abs().mean()
                for attr in contri_attributes
            ]

        df_mean_contri[n] = mean_contri
        df_mean_contri_privileged[n] = privileged_mean_contri
        df_mean_contri_protected[n] = protected_mean_contri

    if num_batches == 1:
        df_mean_contri = transform_df(df_mean_contri)
        df_mean_contri_privileged = transform_df(df_mean_contri_privileged)
        df_mean_contri_protected = transform_df(df_mean_contri_protected)

    return df_mean_contri, df_mean_contri_privileged, df_mean_contri_protected


### Rank-relevance Shapley values
def competing_powers2(
    d,
    weights,
    upper_bound,
    lower_bound,
    features,
    num_batches,
    exponential=1,
    group_feature="_N",
):
    """
    input: df_all, weights
    adjust the competing power according to the rank distirbution, and exponential
    """
    df = d.copy()
    dff = pd.DataFrame()
    grouped = df.groupby(group_feature)

    for n, group in grouped:
        # group[['Attribute 1', 'Attribute 2']] = MinMaxScaler().fit_transform(group[['Attribute 1', 'Attribute 2']])
        rank_sum = group.loc[:, "rank"].sum()
        rank_max = group.loc[:, "rank"].max()

        # calculate the attention of the item based on reverse of the rank over the rank_sum, with optional exponential magnifier par
        group["attention"] = (1 - group["rank"] / rank_max) ** exponential
        group[["attention"]] = MinMaxScaler().fit_transform(
            group[["attention"]]
        )  # scale the attention back to 0 to 1

        ## the raw payout is the score_std
        group["score_std"] = sum([weights[attr] * group[attr] for attr in features])
        for attr in features:
            group[attr + "_contri"] = (
                weights[attr] * group[attr] * group["attention"] / group["score_std"]
            )
            dff = pd.concat([dff, group], axis=0)

    ### use topN to subset the data
    dff = dff.query(f"{upper_bound} <= rank <= {lower_bound}")
    grouped = dff.groupby(group_feature)

    contri_attributes = [x for x in dff.columns if "contri" in x]

    df_mean_contri = pd.DataFrame(index=contri_attributes)
    df_mean_contri_privileged = pd.DataFrame(index=contri_attributes)
    df_mean_contri_protected = pd.DataFrame(index=contri_attributes)

    for n, group in grouped:
        if num_batches == 1:
            mean_contri = [group[attr].abs().tolist() for attr in contri_attributes]
            privileged_mean_contri = [
                group.query('Group == "privileged"')[attr].abs().tolist()
                for attr in contri_attributes
            ]
            protected_mean_contri = [
                group.query('Group == "protected"')[attr].abs().tolist()
                for attr in contri_attributes
            ]
        else:
            mean_contri = [group[attr].abs().mean() for attr in contri_attributes]
            privileged_mean_contri = [
                group.query('Group == "privileged"')[attr].abs().mean()
                for attr in contri_attributes
            ]
            protected_mean_contri = [
                group.query('Group == "protected"')[attr].abs().mean()
                for attr in contri_attributes
            ]

        df_mean_contri[n] = mean_contri
        df_mean_contri_privileged[n] = privileged_mean_contri
        df_mean_contri_protected[n] = protected_mean_contri

    if num_batches == 1:
        df_mean_contri = transform_df(df_mean_contri)
        df_mean_contri_privileged = transform_df(df_mean_contri_privileged)
        df_mean_contri_protected = transform_df(df_mean_contri_protected)

    return df_mean_contri, df_mean_contri_privileged, df_mean_contri_protected


# box plots
# def box_plot_competing_power3(df1, df2, df3, df_names):
#     dfs = [df1, df2, df3]
#     fig = go.Figure()
#
#     marker_colors = ["#FF851B", "#1b23ff", "#00FF00", "#FF00FF", "#FFFF00", "#00FFFF"]
#
#     for idx, df in enumerate(dfs):
#         df = df.T.copy()
#         # df.columns = [f'Attribute {i+1}' for i in range(len(df.columns))]
#         # df = pd.melt(df, id_vars=None, value_vars=[f'Attribute {i+1}' for i in range(len(df.columns))])
#         df = pd.melt(df, id_vars=None, value_vars=df.columns)
#         df = df.rename(columns={"variable": "average contribution"})
#
#         y = df["average contribution"].values
#
#         fig.add_trace(
#             go.Box(
#                 y=df["value"].values,
#                 x=y,
#                 name=df_names[idx],
#                 boxpoints="all",
#                 jitter=0.5,
#                 whiskerwidth=0.2,
#                 marker_color=marker_colors[idx % len(marker_colors)],
#                 marker_size=2,
#                 line_width=1,
#             )
#         )
#
#     fig.update_layout(
#         xaxis=dict(title="average contribution", zeroline=False),
#         boxmode="group",
#     )
#
#     return fig
