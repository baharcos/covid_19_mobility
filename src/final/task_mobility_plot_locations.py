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
from estimagic.visualization.colors import get_colors
from utils import mobility_plot

from src.config import BLD


# Function for plots
colors = get_colors("categorical", 12)

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


de_products = {
    "city_vs_territorial_state": BLD
    / "figures"
    / "German_Mobility"
    / "city_vs_territorial_state.png",
    "former_brd_vs_ddr": BLD / "figures" / "German_Mobility" / "former_brd_vs_ddr.png",
    "four_regions": BLD / "figures" / "German_Mobility" / "four_regions.png",
}


@pytask.mark.depends_on(BLD / "data" / "german_states_data.pkl")
@pytask.mark.produces(de_products)
def task_plot_german_states_mobility(depends_on, produces):
    # Load EU data and keep German data only
    germany_state_level = pd.read_pickle(depends_on)
    germany_state_level = germany_state_level.reset_index(0)


    # City versus territorial state comparison (aggregated)
    mobility_plot(
        data_set=germany_state_level.groupby(["date", "city_noncity"])
        .mean()
        .reset_index(1),
        var_list_moving_avg=varlist_moving_avg,
        titles=titles,
        colors=colors,
        fig_width=20,
        fig_height=40,
        group_var="city_noncity",
    )
    plt.savefig(produces["city_vs_territorial_state"])

    # Former BRD and DDR comparison (aggregated)
    mobility_plot(
        data_set=germany_state_level.groupby(["date", "brd_ddr"]).mean().reset_index(1),
        var_list_moving_avg=varlist_moving_avg,
        titles=titles,
        colors=colors,
        fig_width=20,
        fig_height=40,
        group_var="brd_ddr",
    )
    plt.savefig(produces["former_brd_vs_ddr"])

    # Four regions of Germany
    mobility_plot(
        data_set=germany_state_level.groupby(["date", "four_regions"])
        .mean()
        .reset_index(1),
        var_list_moving_avg=varlist_moving_avg,
        titles=titles,
        colors=colors,
        fig_width=20,
        fig_height=40,
        group_var="four_regions",
    )
    plt.savefig(produces["four_regions"])

eu_products = {
    "small": BLD
    / "figures"
    / "European_Mobility"
    / "plot_small_eu_countries_mobility.png",
    "large": BLD
    / "figures"
    / "European_Mobility"
    / "plot_large_eu_countries_mobility.png",
}


@pytask.mark.depends_on(BLD / "data" / "eu_composed_data_country_level.pkl")
@pytask.mark.produces(eu_products)
def task_plot_european_countries(depends_on, produces):
    # Load in data
    eu_complete_data = pd.read_pickle(depends_on)
    eu_complete_data = eu_complete_data.set_index(["country", "date"])

    small = ["Germany", "Netherlands", "Austria", "Sweden", "Denmark"]
    large = ["Germany", "France", "United Kingdom", "Italy", "Spain"]

    mobility_plot(
        data_set=eu_complete_data.loc[small].reset_index(0), 
        var_list_moving_avg=varlist_moving_avg, 
        titles=titles, 
        colors=colors, 
        group_var="country",
    )
    plt.savefig(produces["small"])

    mobility_plot(
        data_set=eu_complete_data.loc[large].reset_index(0), 
        var_list_moving_avg=varlist_moving_avg, 
        titles=titles, 
        colors=colors, 
        group_var="country",
    )
    plt.savefig(produces["large"])