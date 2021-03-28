"""
This task creates the data necessary for regression of lockdown fatigue.

"""
from datetime import datetime
from datetime import timedelta

import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC


def prepare_regression_data(
    data_composed, stringency_data, dates_lockdowns, first_last_day=None
):
    """
    Creates dataframe with all necessary variables (especially time variables) for
    regression

    Input:
    data_composed (df): dataframe containing mobility and infection data
    stringency_data (df): dataframe containing stringency data
    dates_lockdowns (dict): dictionary with lockdowns as keys and their start and end
    dates as values stored in a list
    first_last_day (list): list containing two dates (strings in "YYYY-MM-DD" format)
    for first/last day in regression sample

    Output:
    regression_data (df): dataframe which contains all variables necessary for
    regression

    """

    # Prepare datasets for merge
    germany_composed_country_level = data_composed.loc[
        data_composed["country"] == "Germany",
    ]
    germany_composed_country_level = germany_composed_country_level.set_index("date")

    stringency_data = stringency_data.reset_index(level=0)
    stringency_data["date"] = stringency_data["date"].apply(lambda x: x.date())
    stringency_data = stringency_data.set_index("date")

    # Merge the two datasets
    regression_data = pd.merge(
        germany_composed_country_level,
        stringency_data,
        left_index=True,
        right_index=True,
    )

    # Change the index to datetime format
    regression_data.index = list(
        map(lambda x: pd.to_datetime(x).date(), regression_data.index.values)
    )

    # Drop unnecessary variables
    regression_data = regression_data.drop(
        ["country", "country_region_code", "place_id"], axis=1
    )

    # Create necessary time variables

    lockdown_names = [*dates_lockdowns]

    for lockdown in lockdown_names:

        lockdown_duration_name = lockdown + "_duration"
        lockdown_7days_moving_average_name = lockdown + "_7days_moving_average"
        lockdown_7days_moving_average_duration_name = (
            lockdown_7days_moving_average_name + "_duration"
        )

        regression_data[lockdown] = (
            (
                regression_data.index
                >= datetime.strptime(dates_lockdowns[lockdown][0], "%Y-%m-%d").date()
            )
            & (
                regression_data.index
                <= datetime.strptime(dates_lockdowns[lockdown][1], "%Y-%m-%d").date()
            )
        ).astype(int)
        regression_data[lockdown_7days_moving_average_name] = (
            (
                regression_data.index
                >= datetime.strptime(dates_lockdowns[lockdown][0], "%Y-%m-%d").date()
            )
            & (
                regression_data.index
                <= datetime.strptime(dates_lockdowns[lockdown][1], "%Y-%m-%d").date()
                - timedelta(7)
            )
        ).astype(int)

        regression_data[lockdown_duration_name] = 0
        regression_data.loc[regression_data[lockdown] == 1, lockdown_duration_name] = (
            regression_data.loc[regression_data[lockdown] == 1].reset_index().index + 1
        )
        regression_data[lockdown_7days_moving_average_duration_name] = 0
        regression_data.loc[
            regression_data[lockdown_7days_moving_average_name] == 1,
            lockdown_7days_moving_average_duration_name,
        ] = (
            regression_data.loc[
                regression_data[lockdown_7days_moving_average_name] == 1
            ]
            .reset_index()
            .index
            + 1
        )

    if first_last_day is not None:
        regression_data = regression_data.loc[
            regression_data.index
            >= datetime.strptime(first_last_day[0], "%Y-%m-%d").date(),
        ]
        regression_data = regression_data.loc[
            regression_data.index
            <= datetime.strptime(first_last_day[1], "%Y-%m-%d").date(),
        ]

    return regression_data


@pytask.mark.depends_on(
    {
        "eu_composed_data_country_level": BLD
        / "data"
        / "eu_composed_data_country_level.pkl",
        "stringency_data": BLD / "data" / "german_stringency_data.pkl",
        "dates_lockdowns": SRC / "model_specs" / "time_lockdowns.pkl",
    }
)
@pytask.mark.produces(BLD / "data" / "regression_data.pkl")
def task_create_regression_data(depends_on, produces):
    eu_composed_country_level = pd.read_pickle(
        depends_on["eu_composed_data_country_level"]
    )
    stringency_data = pd.read_pickle(depends_on["stringency_data"])
    dates_lockdowns = pd.read_pickle(depends_on["dates_lockdowns"])
    regression_data = prepare_regression_data(
        data_composed=eu_composed_country_level,
        stringency_data=stringency_data,
        dates_lockdowns=dates_lockdowns,
        first_last_day=["2020-02-15", "2021-02-22"],
    )
    regression_data.to_pickle(produces)
    # regression_data.to_csv(produces)
