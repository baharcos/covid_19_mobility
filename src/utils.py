def create_moving_average(data, varlist, grouping_var, kind="backward", time=7):
    """Generate moving average variable and add to the data frame

    Args:
        data (pandas.DataFrame): variables of varlist, index must contain grouping_var
        varlist (list): variables for which moving average should be calculated
        grouping_var ([type]): [description]
        kind (str): forward or backward. Defaults to "backward".
        time (int): time span for moving average. Defaults to 7.

    Returns:
        pandas.DataFrame: Input dataframe with additional moving average variables
    """

    suffix = "_avg_" + str(time) + "d"
    varlist_moving_avg = list(map(lambda x: x + suffix, varlist))

    out = data

    if kind == "backward":

        out[varlist_moving_avg] = (
            out[varlist]
            .apply(lambda x: x.groupby(level=grouping_var).rolling(time).mean(), axis=0)
            .reset_index(level=0, drop=True)
            .to_numpy()
        )

    if kind == "forward":

        out[varlist_moving_avg] = (
            out[varlist]
            .apply(
                lambda x: x[::-1]
                .groupby(level=grouping_var)
                .rolling(time)
                .mean()[::-1],
                axis=0,
            )
            .sort_index(0)
            .reset_index(level=0, drop=True)
            .to_numpy()
        )
    out = out.sort_index()
    return out
