"""Clean and format the previously downloaded data sets for the analysis.

"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC

# from utils import create_date
# from utils import create_moving_average


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


# Take european countries list from google data
european_countries = np.array(
    [
        "Austria",
        "Bosnia and Herzegovina",
        "Belgium",
        "Bulgaria",
        "Belarus",
        "Switzerland",
        "Czechia",
        "Germany",
        "Denmark",
        "Spain",
        "Finland",
        "France",
        "United Kingdom",
        "Georgia",
        "Greece",
        "Croatia",
        "Hungary",
        "Ireland",
        "Italy",
        "Liechtenstein",
        "Lithuania",
        "Luxembourg",
        "Latvia",
        "Moldova",
        "North Macedonia",
        "Malta",
        "Netherlands",
        "Norway",
        "Poland",
        "Portugal",
        "Romania",
        "Serbia",
        "Russia",
        "Sweden",
        "Slovenia",
        "Slovakia",
        "Turkey",
        "Ukraine",
    ],
    dtype=object,
)

# Several Divisions of Germany
list_city_states = ["Berlin", "Bremen", "Hamburg"]
list_non_city_states = [
    "Baden-Württemberg",
    "Bavaria",
    "Brandenburg",
    "Hessen",
    "Lower Saxony",
    "Mecklenburg-Vorpommern",
    "North Rhine-Westphalia",
    "Rhineland-Palatinate",
    "Saarland",
    "Saxony",
    "Saxony-Anhalt",
    "Schleswig-Holstein",
    "Thuringia",
]
list_former_brd = [
    "Baden-Württemberg",
    "Bavaria",
    "Bremen",
    "Hamburg",
    "Hessen",
    "Lower Saxony",
    "North Rhine-Westphalia",
    "Rhineland-Palatinate",
    "Saarland",
    "Schleswig-Holstein",
]
list_former_ddr = [
    "Brandenburg",
    "Mecklenburg-Vorpommern",
    "Saxony",
    "Saxony-Anhalt",
    "Thuringia",
]
list_west_germany = [
    "Hessen",
    "North Rhine-Westphalia",
    "Rhineland-Palatinate",
    "Saarland",
]
list_south_germany = ["Baden-Württemberg", "Bavaria"]
list_north_germany = ["Bremen", "Hamburg", "Lower Saxony", "Schleswig-Holstein"]
list_east_germany = [
    "Brandenburg",
    "Berlin",
    "Mecklenburg-Vorpommern",
    "Saxony",
    "Saxony-Anhalt",
    "Thuringia",
]


# def prepare_mobility_data(data):
#     """
#     Add docstring here

#     """

#     # Keep only european countries in the dataset
#     eu_data = data.query("country_region in @european_countries")

#     # Replace NaN with "country" in "sub_region_1" column
#     eu_data["sub_region_1"] = eu_data["sub_region_1"].replace(np.nan, "country")

#     # Drop census_fips_code (contains no information)
#     eu_data.drop("census_fips_code", axis=1)

#     # Rename variables
#     eu_data.columns = map(
#         lambda x: x.replace("_percent_change_from_baseline", ""), eu_data.columns
#     )
#     eu_data.rename(columns={"country_region": "country"}, inplace=True)

#     # Drop census_fips_code because contains only NaN's
#     eu_data = eu_data.drop(["census_fips_code"], axis=1)

#     # Change datatypes of some columns to string
#     eu_data[
#         ["sub_region_1", "sub_region_2", "metro_area", "iso_3166_2_code", "place_id"]
#     ] = eu_data[
#         ["sub_region_1", "sub_region_2", "metro_area", "iso_3166_2_code", "place_id"]
#     ].astype(
#         str
#     )

#     # Create date variables
#     eu_data = create_date(eu_data)

#     # Use MultiIndex for better overview
#     eu_data = eu_data.set_index(["country", "date"])
#     eu_data = eu_data.drop("date_str", axis=1)

#     # create dataset for state-level comparison
#     germany_state_level = eu_data.loc["Germany"]
#     germany_state_level = germany_state_level.drop(
#         ["place_id", "metro_area", "sub_region_2", "country_region_code"], axis=1
#     )
#     germany_state_level = germany_state_level.rename(columns={"sub_region_1": "state"})

#     germany_state_level = germany_state_level.reset_index()
#     germany_state_level = germany_state_level.set_index(["state", "date"])

#     germany_state_level.loc[list_city_states, "city_noncity"] = "city state"
#     germany_state_level.loc[list_non_city_states, "city_noncity"] = "territorial state"
#     germany_state_level.loc[list_former_brd, "brd_ddr"] = "former BRD"
#     germany_state_level.loc[list_former_ddr, "brd_ddr"] = "former DDR"
#     germany_state_level.loc[list_west_germany, "four_regions"] = "West"
#     germany_state_level.loc[list_south_germany, "four_regions"] = "South"
#     germany_state_level.loc[list_north_germany, "four_regions"] = "North"
#     germany_state_level.loc[list_east_germany, "four_regions"] = "East"

#     germany_state_level = germany_state_level.drop("country")

#     germany_state_level = create_moving_average(
#         germany_state_level,
#         [
#             "retail_and_recreation",
#             "grocery_and_pharmacy",
#             "parks",
#             "transit_stations",
#             "workplaces",
#             "residential",
#         ],
#         "state",
#         kind="forward",
#     )
#     germany_state_level.to_pickle(produces["german_states"])

#     # Create dataset for comparison between different european countries
#     eu_country_level_data = eu_data[eu_data["sub_region_1"] == "country"]
#     eu_country_level_data = eu_country_level_data.loc[
#         eu_country_level_data["metro_area"] == "nan",
#     ]
#     eu_country_level_data = eu_country_level_data.drop(
#         ["sub_region_1", "sub_region_2", "metro_area", "iso_3166_2_code"], axis=1
#     )

#     # Create moving average
#     eu_country_level_data = create_moving_average(
#         eu_country_level_data,
#         [
#             "retail_and_recreation",
#             "grocery_and_pharmacy",
#             "parks",
#             "transit_stations",
#             "workplaces",
#             "residential",
#         ],
#         "country",
#         kind="forward",
#     )

#     # Load in infection numbers
#     eu_infect_numbers = pd.read_pickle(depends_on["infection"])
#     # eu_infect_numbers = eu_infect_numbers.set_index(["country", "date"])

#     # Join the two datasets
#     eu_composed_data_country_level = eu_country_level_data.join(eu_infect_numbers)
#     eu_composed_data_country_level = eu_composed_data_country_level.reset_index()
#     eu_composed_data_country_level = eu_composed_data_country_level.drop(
#         "Unnamed: 0", axis=1
#     )

#     return eu_data


def prepare_owid_infection_data(data):
    """
    Add docstring here

    """
    # Keep only european countries which are in the Google data
    # eu_infection_data = data.query("location in @european_countries")

    # Rename location to country
    out = data.rename(columns={"location": "country"})

    # Take only columns we need
    out = out.loc[:, ("country", "date", "total_cases", "new_cases")]

    # Create date variables
    out = create_date(out)

    # Use MultiIndex for better overview
    out = out.set_index(["country", "date"])

    # Generate 7-day simple moving average
    out = create_moving_average(out, ["new_cases"], "country", kind="forward", time=7)

    return out


def prepare_stringency_data(data):
    """
    Add docstring here

    """
    out = create_date(data, "date")
    out["date"] = out["date"].apply(lambda x: x.date())
    out = out.set_index(["country", "date"])
    out = out.sort_index()

    out = create_moving_average(
        out,
        ["stringency_index"],
        grouping_var="country",
        kind="forward",
        time=7,
    )

    return out


########################################################################################

# @pytask.mark.depends_on(
#     {
#         "google": SRC / "original_data" / "google_data.csv",
#         "infection": BLD / "data" / "infection_data.pkl",
#     }
# )
# @pytask.mark.produces(
#     {
#         "german_states": BLD / "data" / "german_states_data.pkl",
#         "eu_country_level": BLD / "data" / "eu_composed_data_country_level.pkl",
#     }
# )
# def task_prepare_mobility_data(depends_on, produces):
#     google_mobility_data = pd.read_csv(depends_on["google"])
#     google_mobility_data_clean = prepare_mobility_data(google_mobility_data)
#     google_mobility_data_clean.to_pickle(produces["eu_country_level"])


@pytask.mark.depends_on(SRC / "original_data" / "owid_data.csv")
@pytask.mark.produces(
    {
        "eu_infection_data": BLD / "data" / "infection_data.pkl",
        "germany_infection_data": BLD / "data" / "germany_infection_data.pkl",
    }
)
def task_prepare_owid_infection_data(depends_on, produces):
    global_infection_data = pd.read_csv(depends_on)
    global_infection_data = prepare_owid_infection_data(data=global_infection_data)

    eu_infection_data = global_infection_data.loc[european_countries]
    germany_infection_data = global_infection_data.loc["Germany"]

    eu_infection_data.to_pickle(produces["eu_infection_data"])
    germany_infection_data.to_pickle(produces["germany_infection_data"])


@pytask.mark.depends_on(SRC / "original_data" / "stringency_index_data.csv")
@pytask.mark.produces(
    {
        "global_stringency_data": BLD / "data" / "global_stringency_data.pkl",
        "germany_stringency_data": BLD / "data" / "germany_stringency_data.pkl",
    }
)
def task_prepare_stringency_data(depends_on, produces):
    global_stringency_data = pd.read_csv(depends_on)
    global_stringency_data = prepare_stringency_data(global_stringency_data)
    germany_stringency_data = global_stringency_data.loc["Germany"].drop(
        "country_code", axis=1
    )

    global_stringency_data.to_pickle(produces["global_stringency_data"])
    germany_stringency_data.to_pickle(produces["germany_stringency_data"])
