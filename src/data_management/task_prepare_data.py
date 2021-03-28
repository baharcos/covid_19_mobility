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


def prepare_eu_mobility_data(data):
    """
    Add docstring here
    """

    out = data.rename(columns={"country_region": "country"})
    out = out.set_index("country")
    out = out.loc[european_countries]

    # Replace NaN with "country" in "sub_region_1" column
    out["sub_region_1"] = out["sub_region_1"].replace(np.nan, "country")

    # Rename variables
    out.columns = map(
        lambda x: x.replace("_percent_change_from_baseline", ""), out.columns
    )
    out.rename(columns={"country_region": "country"}, inplace=True)

    # Drop census_fips_code because contains only NaN's
    out = out.drop(["census_fips_code"], axis=1)

    # Change datatypes of some columns to string
    out[
        ["sub_region_1", "sub_region_2", "metro_area", "iso_3166_2_code", "place_id"]
    ] = out[
        ["sub_region_1", "sub_region_2", "metro_area", "iso_3166_2_code", "place_id"]
    ].astype(
        str
    )

    # Create date variables
    out = create_date(out)

    # Use MultiIndex for better overview
    out = out.set_index("date", append=True)
    out = out.drop(
        [
            "Unnamed: 0",
            "date_str",
            "iso_3166_2_code",
            "metro_area",
            "sub_region_2",
            "place_id",
        ],
        axis=1,
    )
    out = out.rename(columns={"sub_region_1": "state"})

    return out


def prepare_mobility_germany_state_data(data):
    """
    Add docstring here.
    """

    out = data.reset_index()
    out = out.set_index("state")

    out.loc[list_city_states, "city_noncity"] = "city state"
    out.loc[list_non_city_states, "city_noncity"] = "territorial state"
    out.loc[list_former_brd, "brd_ddr"] = "former BRD"
    out.loc[list_former_ddr, "brd_ddr"] = "former DDR"
    out.loc[list_west_germany, "four_regions"] = "West"
    out.loc[list_south_germany, "four_regions"] = "South"
    out.loc[list_north_germany, "four_regions"] = "North"
    out.loc[list_east_germany, "four_regions"] = "East"

    out = create_moving_average(
        out,
        [
            "retail_and_recreation",
            "grocery_and_pharmacy",
            "parks",
            "transit_stations",
            "workplaces",
            "residential",
        ],
        "state",
        kind="forward",
    )

    out = out.drop("country_region_code", axis=1)
    out = out.set_index("date", append=True)

    return out

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


@pytask.mark.depends_on(SRC / "original_data" / "google_data.csv")
@pytask.mark.produces(
    {
        "mobility_germany_state_level": BLD
        / "data"
        / "mobility_germany_states_data.pkl",
        "mobility_germany_country_level": BLD
        / "data"
        / "mobility_germany_country_data.pkl",
        "mobility_eu_country_level": BLD / "data" / "mobility_eu_mobility_data.pkl",
    }
)
def task_prepare_mobility_data(depends_on, produces):
    google_mobility_data = pd.read_csv(depends_on)
    eu_mobility_data = prepare_eu_mobility_data(google_mobility_data)

    germany_state_level_mobility_data = eu_mobility_data.loc["Germany"]
    germany_state_level_mobility_data = prepare_mobility_germany_state_data(
        germany_state_level_mobility_data
    )
    germany_country_level_mobility_data = germany_state_level_mobility_data.loc[
        "country"
    ]

    eu_mobility_data = create_moving_average(
        eu_mobility_data,
        [
            "retail_and_recreation",
            "grocery_and_pharmacy",
            "parks",
            "transit_stations",
            "workplaces",
            "residential",
        ],
        "country",
        kind="forward",
    )
    eu_mobility_data.to_pickle(produces["mobility_eu_country_level"])
    germany_country_level_mobility_data.to_pickle(
        produces["mobility_germany_country_level"]
    )
    germany_country_level_mobility_data.to_pickle(
        produces["mobility_germany_state_level"]
    )


@pytask.mark.depends_on(SRC / "original_data" / "owid_data.csv")
@pytask.mark.produces(
    {
        "eu_infection_data": BLD / "data" / "eu_infection_data.pkl",
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


@pytask.mark.depends_on(
    {
        "mobility_eu_country_level": BLD
        / "data"
        / "mobility_eu_mobility_data.pkl",
        "eu_infection_data": BLD / "data" / "eu_infection_data.pkl",
    }
)
@pytask.mark.produces(BLD / "data" / "eu_composed_data_country_level.pkl")
def task_prepare_eu_compounded_data(depends_on, produces):
    eu_mobility_data = pd.read_pickle(depends_on["mobility_eu_country_level"])
    eu_infection_data = pd.read_pickle(depends_on["eu_infection_data"])

    eu_composed_data = pd.merge(
        eu_mobility_data, eu_infection_data, left_index=True, right_index=True
    )
    eu_composed_data.to_pickle(produces)
