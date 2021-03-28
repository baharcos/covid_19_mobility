"""
This task runs the regressions to identify lockdown fatigue.

"""
import pickle
from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
import pytask
import statsmodels.formula.api as smf
from ordered_set import OrderedSet
from stargazer.stargazer import Stargazer

from src.config import BLD
from src.config import SRC

# Dummy for circumventing pre-commit hook issues
dummy = np.mean([1, 2])


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


def ols_regression_formatted(
    data, specifications, as_latex=False, covariates_names=None, covariates_order=None
):

    """
    Creates formatted tables for different dependent variables and specifications

    Input:
    data (df): Dataframe containing all necessary variables for OLS regression
    specifications (dictionary): dependent variables as keys and list of specifications
    as values
    as_latex (bool): specify whether Output as table or Latex code
    covariate_names (dict): dictionary with covariate names as in "data" as keys and new
    covariate names as values

    Output:
    list_of_tables (list of stargazer tables): list of formatted tables

    """

    # Create dictionary which connects dependent variables with formatted tables
    dict_regression_tables = {}

    # Generate regressions
    for depvar in specifications.keys():

        regression_list = []
        specification_list = specifications[depvar]
        list_all_covariates = []

        for specification in specification_list:

            estimation_equation = depvar + " ~ " + specification
            regression = smf.ols(data=data, formula=estimation_equation).fit()
            regression_list.append(regression)

            # Create set of all variables for this dependent variable
            list_all_covariates = list(
                set(list_all_covariates + regression.params.index.values.tolist())
            )

        # Format table with stargazer
        formatted_table = Stargazer(regression_list)

        # No dimension of freedoms and blank dependent variable
        formatted_table.show_degrees_of_freedom(False)
        formatted_table.dependent_variable_name("")

        # Optional: Change order of covariates
        if covariates_order is not None:

            covariates_order_depvar = list(
                OrderedSet(covariates_order).intersection(list_all_covariates)
            )
            list_remaining_covariates = list(
                OrderedSet(list_all_covariates).difference(
                    OrderedSet(covariates_order_depvar)
                )
            )
            covariates_sorted = list(
                OrderedSet(covariates_order_depvar).union(list_remaining_covariates)
            )
            covariates_sorted.remove("Intercept")
            covariates_sorted = covariates_sorted + ["Intercept"]

            formatted_table.covariate_order(covariates_sorted)

        # Optional: Change name of covariates
        if covariates_names is not None:

            formatted_table.rename_covariates(covariates_names)

        # Add table or latex code to dictionary
        if as_latex is True:

            dict_regression_tables[depvar] = formatted_table.render_latex()

            # Delete tabular environment around it
            dict_regression_tables[depvar] = dict_regression_tables[depvar].replace(
                "\\begin{table}[!htbp] \\centering\n", ""
            )
            dict_regression_tables[depvar] = dict_regression_tables[depvar].replace(
                "\\end{table}", ""
            )

        else:
            dict_regression_tables[depvar] = formatted_table

    return dict_regression_tables


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


@pytask.mark.depends_on(
    {
        "regression_data": BLD / "data" / "regression_data.pkl",
        "regression_specifications": SRC / "model_specs" / "regression_models.pkl",
        "regression_variable_names": SRC
        / "model_specs"
        / "regression_variable_names.pkl",
    }
)
@pytask.mark.produces(
    {
        "all_regression_tables": BLD / "tables" / "all_regression_tables.pkl",
        "all_regression_tables_latex": BLD
        / "tables"
        / "all_regression_tables_latex.pkl",
    }
)
def task_run_regressions(depends_on, produces):
    # Import data
    regression_data = pd.read_pickle(depends_on["regression_data"])
    regression_specifications = pd.read_pickle(depends_on["regression_specifications"])
    regression_variable_names = pd.read_pickle(depends_on["regression_variable_names"])

    all_regression_tables = ols_regression_formatted(
        data=regression_data,
        specifications=regression_specifications,
        as_latex=False,
        covariates_names=regression_variable_names,
        covariates_order=[*regression_variable_names],
    )
    all_regression_tables_latex = ols_regression_formatted(
        data=regression_data,
        specifications=regression_specifications,
        as_latex=True,
        covariates_names=regression_variable_names,
        covariates_order=[*regression_variable_names],
    )

    all_regression_tables_file = open(produces["all_regression_tables"], "wb")
    pickle.dump(all_regression_tables, all_regression_tables_file)

    all_regression_tables_latex_file = open(
        produces["all_regression_tables_latex"], "wb"
    )
    pickle.dump(all_regression_tables_latex, all_regression_tables_latex_file)


# Create "produces" dictionary for export of tables
produces_dictionary_export = {}
all_regression_tables_latex_file = open(
    BLD / "tables" / "all_regression_tables_latex.pkl", "rb"
)
all_regression_tables_latex = pickle.load(all_regression_tables_latex_file)
all_regression_tables_latex_file.close()

produces_dictionary_export = {}
for dependent_variable in [*all_regression_tables_latex]:
    produces_name = "table_regression_" + dependent_variable
    produces_file_name = produces_name + ".tex"
    produces_dictionary_export[produces_name] = (
        SRC / "paper" / "tables" / produces_file_name
    )


@pytask.mark.depends_on(BLD / "tables" / "all_regression_tables_latex.pkl")
@pytask.mark.produces(produces_dictionary_export)
def task_export_regression_tables(depends_on, produces):

    all_regression_tables_latex_file = open(depends_on, "rb")
    all_regression_tables_latex = pickle.load(all_regression_tables_latex_file)

    for produces_name in [*produces]:
        dependent_variable = produces_name.replace("table_regression_", "")
        regression_table_latex_file = open(produces[produces_name], "wb")
        regression_table_latex_file.write(
            bytes(all_regression_tables_latex[dependent_variable], "utf-8")
        )
