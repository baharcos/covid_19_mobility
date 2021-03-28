"""Generate mobility plots

"""
import matplotlib.pyplot as plt
import seaborn as sns
from estimagic.visualization.colors import get_colors

# from src.config import BLD

colors = get_colors("categorical", 12)
titles = [
    "Retail and Recreation (7d-average)",
    "Grocery and Pharmacy (7d-average)",
    "Workplaces (7d-average)",
    "Parks (7d-average)",
    "Residential (7d-average)",
    "Transit Stations (7d-average)",
]
small = ["Germany", "Netherlands", "Austria", "Sweden", "Denmark"]
large = ["Germany", "France", "United Kingdom", "Italy", "Spain"]


def mobility_plot(data_set, var_list_moving_avg, fig_width=20, fig_height=40):
    """[summary]

    Args:
        data_set ([type]): [description]
        var_list_moving_avg ([type]): [description]
        fig_width (int, optional): [description]. Defaults to 20.
        fig_height (int, optional): [description]. Defaults to 40.
    """
    num_plots = len(var_list_moving_avg)
    fig, ax = plt.subplots(num_plots, 1, figsize=(fig_width, fig_height))

    for i in range(num_plots):
        sns.set_palette(colors)
        sns.lineplot(
            x=data_set.index,
            y=var_list_moving_avg[i],
            data=data_set,
            hue="state",
            ax=ax[i],
        )
        ax[i].axhline(0, color="black", alpha=0.8)
        ax[i].set_title(titles[i])
        ylim_min = data_set.loc[:, var_list_moving_avg[i]].min() - 10
        ylim_max = data_set.loc[:, var_list_moving_avg[i]].max() + 10
        ax[i].set_ylim(ylim_min, ylim_max)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["top"].set_visible(False)


# @pytask.mark.depends_on(BLD/"data"/"eu_composed_data_country_level.pkl")
# mobility_plot(data_set=eu_complete_data.loc[small].reset_index(0)
