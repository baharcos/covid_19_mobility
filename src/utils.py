from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


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

def mobility_plot(data_set, var_list_moving_avg, titles, colors, group_var, fig_width=20, fig_height=40):
    """Creating multiple plots in one figure for a given data frame, titles and colors have to be defined
    before using the function

    Args:
        data_set (pandas.DataFrame): 
        var_list_moving_avg ([type]): which variables the figures should be created for
        fig_width (int, optional): Define figure width. Defaults to 20.
        fig_height (int, optional): Define figure height. Defaults to 40.
    """
    num_plots = len(titles)
    fig, ax = plt.subplots(num_plots, 1, figsize=(fig_width, fig_height))

    for i in range(num_plots):
        sns.set_palette(colors)
        sns.lineplot(
            x=data_set.index,
            y=var_list_moving_avg[i],
            data=data_set,
            hue=group_var,
            ax=ax[i],
        )
        ax[i].axhline(0, color="black", alpha=0.8)
        ax[i].set_title(titles[i])
        ylim_min = data_set.loc[:, var_list_moving_avg[i]].min() - 10
        ylim_max = data_set.loc[:, var_list_moving_avg[i]].max() + 10
        ax[i].set_ylim(ylim_min, ylim_max)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["top"].set_visible(False)