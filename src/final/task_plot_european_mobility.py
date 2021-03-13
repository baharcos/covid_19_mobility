import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytask
import seaborn as sns

from datetime import datetime
from src.config import BLD
from src.config import SRC


def plot_european_countries(countries, path):

    # # Plotting infection numbers in small countries
    # fig, ax = plt.subplots(figsize=(15,8))

    # sns.lineplot(x=countries.index, y="new_cases_avg_7d",data=countries,hue="country")

    # ax.axhline(0,color="black",alpha=0.8)


    ### Compare Germany, France, UK, Italy and Spain for different dimensions

    fig, ax = plt.subplots(3, 1, figsize=(15,24))

    sns.lineplot(x=countries.index, y="retail_and_recreation_avg_7d",data=countries,hue="country",ax=ax[0])
    sns.lineplot(x=countries.index, y="grocery_and_pharmacy_avg_7d",data=countries,hue="country",ax=ax[1])
    sns.lineplot(x=countries.index, y="workplaces_avg_7d",data=countries,hue="country",ax=ax[2])

    for i in range(3):
        ax[i].axhline(0,color="black",alpha=0.8)

    ax[0].set_title("Retail and Recreation (7d-average)")
    ax[0].set_ylim(-100,30)

    ax[1].set_title("Grocery and Pharmacy (7d-average)")
    ax[1].set_ylim(-70,30)

    ax[2].set_title("Workplaces (7d-average)")
    ax[2].set_ylim(-80,10)

    plt.savefig(path)


specification = (
    (
        BLD / "figures" / f"plot_{countries}_mobility.png"
    ) 
    for countries in ["small_countries", "large_countries"]
)

@pytask.mark.depends_on(BLD/"data"/"eu_complete_data.csv")
@pytask.mark.parametrize("produces", specification)
def task_plot_european_countries(depends_on, produces):
    #Load in data
    eu_complete_data = pd.read_csv(depends_on)
    eu_complete_data = eu_complete_data.set_index(["country", "date"])

    small_countries = eu_complete_data.loc[["Germany","Netherlands","Austria","Sweden","Denmark"]]
    large_countries = eu_complete_data.loc[["Germany","France","United Kingdom","Italy","Spain"]]
    
    for countries in [small_countries, large_countries]:
        countries = countries.reset_index(0)
        plot_european_countries(countries, produces)
