"""
This task produces two different kinds of plots:\n
1. Overall German mobility with respect to categories "Retail and Recreation", "Grocery
and Pharmacy", "Workplaces", "Parks", "Transit Stations" and "Residential"\n
2. Mobility in different forms of division of Germany: "City States", "Non-City States",
"City vs Territorial", "Former BRD vs DDR states" and "North-East-South-West comparison"
\n
"""
import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD


# Function for plots
titles = [
    "Retail and Recreation (7d-average)",
    "Grocery and Pharmacy (7d-average)",
    "Workplaces (7d-average)",
    "Parks (7d-average)",
    "Residential (7d-average)",
    "Transit Stations (7d-average)",
]
varlist_moving_avg = [
    "retail_and_recreation_avg_7d",
    "grocery_and_pharmacy_avg_7d",
    "workplaces_avg_7d",
    "parks_avg_7d",
    "residential_avg_7d",
    "transit_stations_avg_7d",
]


def mobility_plot(
    data_set, var_list_moving_avg, path, fig_width=20, fig_height=40, group_var="state"
):
    """
    Produces figure for mobility based on variables provided to the function

    Input:
        data_set (df): dataset containing
        var_list_moving_avg (list): list containing categories to observe
        fig_width (int): Width of figure (default is 20)
        fig_height (int): Height of figure (default is 40)
        group_var (string): Variable to be grouped on
        path (path-like): path where to save the produces figure

    Output:
        figure (matplotlib object): produced figure
    """

    num_plots = len(var_list_moving_avg)
    fig, ax = plt.subplots(num_plots, 1, figsize=(fig_width, fig_height))

    for i in range(num_plots):
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

        plt.savefig(path)


@pytask.mark.depends_on(BLD / "data" / "eu_composed_data_country_level.pkl")
@pytask.mark.produces(
    BLD / "figures" / "German_Mobility" / "plot_overall_german_mobility.png"
)
def task_plot_german_mobility(depends_on, produces):

    # Load EU data and keep German data only
    eu_country_level_data = pd.read_pickle(depends_on)
    eu_country_level_data = eu_country_level_data.set_index(["country", "date"])
    germany_country_level_data = eu_country_level_data.loc["Germany"]

    fig, ax = plt.subplots(figsize=(15, 8))

    sns.lineplot(
        data=germany_country_level_data[
            [
                "retail_and_recreation_avg_7d",
                "grocery_and_pharmacy_avg_7d",
                "parks_avg_7d",
                "transit_stations_avg_7d",
                "workplaces_avg_7d",
                "residential_avg_7d",
            ]
        ]
    )

    ax.axhline(0, color="black", alpha=0.8)
    ax.axvspan(
        pd.to_datetime("2020-03-22"),
        pd.to_datetime("2020-06-15"),
        alpha=0.2,
        color="lightgrey",
    )
    ax.axvspan(
        pd.to_datetime("2020-11-02"),
        pd.to_datetime("2021-03-01"),
        alpha=0.2,
        color="lightgrey",
    )
    ax.axvspan(
        pd.to_datetime("2020-12-16"),
        pd.to_datetime("2021-03-01"),
        alpha=0.4,
        color="lightgrey",
    )

    plt.savefig(produces)


products = {
    "city_state": BLD / "figures" / "German_Mobility" / "city_state.png",
    "non_city_state": BLD / "figures" / "German_Mobility" / "non_city_state.png",
    "city_vs_territorial_state": BLD
    / "figures"
    / "German_Mobility"
    / "city_vs_territorial_state.png",
    "former_brd_vs_ddr": BLD / "figures" / "German_Mobility" / "former_brd_vs_ddr.png",
    "four_regions": BLD / "figures" / "German_Mobility" / "four_regions.png",
}


@pytask.mark.depends_on(BLD / "data" / "german_states_data.pkl")
@pytask.mark.produces(products)
def task_plot_german_states_mobility(depends_on, produces):
    # Load EU data and keep German data only
    germany_state_level = pd.read_pickle(depends_on)
    germany_state_level = germany_state_level.reset_index(0)

    # City states comparison (unaggregated)
    mobility_plot(
        data_set=germany_state_level.loc[
            germany_state_level["city_noncity"] == "city state"
        ],
        var_list_moving_avg=varlist_moving_avg,
        fig_width=20,
        fig_height=40,
        path=produces["city_state"],
    )

    # Non city_state comparison (unaggregated)
    mobility_plot(
        data_set=germany_state_level.loc[
            germany_state_level["city_noncity"] == "territorial state"
        ],
        var_list_moving_avg=varlist_moving_avg,
        fig_width=20,
        fig_height=40,
        path=produces["non_city_state"],
    )

    # City versus territorial state comparison (aggegregated)
    mobility_plot(
        data_set=germany_state_level.groupby(["date", "city_noncity"])
        .mean()
        .reset_index(1),
        var_list_moving_avg=varlist_moving_avg,
        fig_width=20,
        fig_height=40,
        group_var="city_noncity",
        path=produces["city_vs_territorial_state"],
    )

    # Former BRD and DDR comparison (aggegregated)
    mobility_plot(
        data_set=germany_state_level.groupby(["date", "brd_ddr"]).mean().reset_index(1),
        var_list_moving_avg=varlist_moving_avg,
        fig_width=20,
        fig_height=40,
        group_var="brd_ddr",
        path=produces["former_brd_vs_ddr"],
    )

    # Four regions of Germany
    mobility_plot(
        data_set=germany_state_level.groupby(["date", "four_regions"])
        .mean()
        .reset_index(1),
        var_list_moving_avg=varlist_moving_avg,
        fig_width=20,
        fig_height=40,
        group_var="four_regions",
        path=produces["four_regions"],
    )
