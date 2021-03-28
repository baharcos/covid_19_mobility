from datetime import datetime


def create_date(data, date_name="date"):
    """Generates and adds date variables: datetime, day, week, weekend, month and year

    Args:
        data (pandas.DataFrame): must contain a date column
        date_name (str): Defaults to "date".

    Returns:
        pandas.DataFrame: Input dataframe with additional date variables
    """

    out = data.rename(columns={date_name: "date_str"})
    out["date"] = list(map(lambda x: datetime.strptime(x, "%Y-%m-%d"), out["date_str"]))

    out["weekday"] = list(map(lambda x: x.weekday(), out["date"]))
    out["weekday"] = out["weekday"].replace(
        {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    )

    out["day"] = list(map(lambda x: x.day, out["date"]))
    out["week"] = list(map(lambda x: x.week, out["date"]))
    out["weekend"] = list(
        map(lambda x: int(x), (out["weekday"] == "Sat") | (out["weekday"] == "Sun"))
    )
    out["month"] = list(map(lambda x: x.month, out["date"]))
    out["year"] = list(map(lambda x: x.year, out["date"]))

    return out


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
