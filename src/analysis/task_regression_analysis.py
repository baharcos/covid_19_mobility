"""
This task runs the regressions to identify lockdown fatigue.

"""
import pickle

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


@pytask.mark.depends_on(BLD / "tables" / "all_regression_tables_latex.pkl")
@pytask.mark.produces(
    {
        "workplaces_avg_7d": BLD / "tables" / "regression_table_workplaces_avg_7d.tex",
        "retail_and_recreation_avg_7d": BLD
        / "tables"
        / "regression_table_retail_and_recreation_avg_7d_avg_7d.tex",
        "residential_avg_7d": BLD
        / "tables"
        / "regression_table_residential_avg_7d_avg_7d.tex",
        "grocery_and_pharmacy_avg_7d": BLD
        / "tables"
        / "regression_table_grocery_and_pharmacy_avg_7d_avg_7d.tex",
        "transit_stations_avg_7d": BLD
        / "tables"
        / "regression_table_transit_stations_avg_7d_avg_7d.tex",
    }
)
def task_export_regression_tables(depends_on, produces):

    all_regression_tables_latex_file = open(depends_on, "rb")
    all_regression_tables_latex = pickle.load(all_regression_tables_latex_file)

    for produces_name in [*produces]:
        dependent_variable = produces_name.replace("regression_table", "")
        regression_table_latex_file = open(produces[produces_name], "wb")
        regression_table_latex_file.write(
            bytes(all_regression_tables_latex[dependent_variable], "utf-8")
        )
