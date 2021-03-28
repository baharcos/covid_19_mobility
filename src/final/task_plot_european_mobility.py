"""
This task compares overall German mobility with mobility from larger european countries
(France, United Kingdom, Italy, Spain) and smaller european countries
(Netherlands, Austria, Sweden, Denmark)
"""
import matplotlib.pyplot as plt
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD


def plot_european_countries(countries, path):
    """
    Plots mobility on country level for different countries

    Input:
        countries (list): list containing countries whose mobility should be comared
        path (path-like): path where to save the produced figure

    Output:
        figure (matplotlib object): figure showing mobility in different countries
    """

    # Plotting infection numbers in small countries
    # fig, ax = plt.subplots(figsize=(15,8))

    # sns.lineplot(x=countries.index, y="new_cases_avg_7d",data=countries,hue="country")

    # ax.axhline(0,color="black",alpha=0.8)

    # Compare Germany, France, UK, Italy and Spain for different dimensions
    fig, ax = plt.subplots(3, 1, figsize=(15, 24))

    sns.lineplot(
        x=countries.index,
        y="retail_and_recreation_avg_7d",
        data=countries,
        hue="country",
        ax=ax[0],
    )
    sns.lineplot(
        x=countries.index,
        y="grocery_and_pharmacy_avg_7d",
        data=countries,
        hue="country",
        ax=ax[1],
    )
    sns.lineplot(
        x=countries.index,
        y="workplaces_avg_7d",
        data=countries,
        hue="country",
        ax=ax[2],
    )

    for i in range(3):
        ax[i].axhline(0, color="black", alpha=0.8)

    ax[0].set_title("Retail and Recreation (7d-average)")
    ax[0].set_ylim(-100, 30)

    ax[1].set_title("Grocery and Pharmacy (7d-average)")
    ax[1].set_ylim(-70, 30)

    ax[2].set_title("Workplaces (7d-average)")
    ax[2].set_ylim(-80, 10)

    return plt.savefig(path)


products = {
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
@pytask.mark.produces(products)
def task_plot_european_countries(depends_on, produces):
    # Load in data
    eu_complete_data = pd.read_pickle(depends_on)
    eu_complete_data = eu_complete_data.set_index(["country", "date"])

    small = ["Germany", "Netherlands", "Austria", "Sweden", "Denmark"]
    large = ["Germany", "France", "United Kingdom", "Italy", "Spain"]

    plot_european_countries(
        countries=eu_complete_data.loc[small].reset_index(0), path=produces["small"]
    )
    plot_european_countries(
        countries=eu_complete_data.loc[large].reset_index(0), path=produces["large"]
    )
