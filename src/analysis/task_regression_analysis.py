'''
This task ...
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pytask
from collections import Counter
from ordered_set import OrderedSet
from stargazer.stargazer import Stargazer
from src.config import BLD
from src.config import SRC
from datetime import datetime
from datetime import timedelta

def ols_regression_formatted(data,specifications, as_latex=False, covariates_names=None, covariates_order=None):
    
    """
    Creates formatted tables for different dependent variables and specifications
    
    Input: 
    data (df): Dataframe containing all necessary variables for OLS regression
    specifications (dictionary): dependent variables as keys and list of specifications as values
    as_latex (bool): specify whether Output as table or Latex code
    covariate_names (dict): dictionary with covariate names as in "data" as keys and new covariate names as values
    
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
            regression = smf.ols(data=data,formula=estimation_equation).fit()
            regression_list.append(regression)
            
            # Create set of all variables for this dependent variable
            
            list_all_covariates = list(set(list_all_covariates + regression.params.index.values.tolist()))
            
        # Format table with stargazer
        
        formatted_table = Stargazer(regression_list)

            # No dimension of freedoms and blank dependent variable
        
        formatted_table.show_degrees_of_freedom(False)
        formatted_table.dependent_variable_name("")
        
            # Optional: Change order of covariates
        
        if covariates_order != None:
            
            covariates_order_depvar = list(OrderedSet(covariates_order).intersection(list_all_covariates))
            list_remaining_covariates = list(OrderedSet(list_all_covariates).difference(OrderedSet(covariates_order_depvar)))
            covariates_sorted = list(OrderedSet(covariates_order_depvar).union(list_remaining_covariates))
            covariates_sorted.remove("Intercept") 
            covariates_sorted = covariates_sorted + ["Intercept"]
            
            formatted_table.covariate_order(covariates_sorted)
        
            # Optional: Change name of covariates
            
        if covariates_names != None: 
            
            formatted_table.rename_covariates(covariates_names)
                  
            # Add table or latex code to dictionary 
            
        if as_latex == True: 
            
            dict_regression_tables[depvar] = formatted_table.render_latex()
        
        else:
            dict_regression_tables[depvar] = formatted_table
                
    return(dict_regression_tables)

@pytask.mark.depends_on({"german_states_mobility": BLD/"data"/"german_states_data.csv", "eu_composed_data": BLD/"data"/"eu_composed_data_country_level.csv", "stringency": BLD/"data"/"german_stringency_data.csv"})
@pytask.mark.produces(BLD/"data"/"regression_data.csv")
def task_create_regression_tables(depends_on, produces):

    germany_mobility_country_level = pd.read_csv(depends_on["eu_composed_data"], index_col=["country", "date"], parse_dates=True)
    germany_mobility_country_level = germany_mobility_country_level.loc[["Germany"], :]
    germany_mobility_country_level = germany_mobility_country_level.reset_index(level="country", drop=True)
    data_stringency_germany = pd.read_csv(depends_on["stringency"], index_col="date",parse_dates=True)

    regression_data = pd.merge(germany_mobility_country_level,data_stringency_germany,left_index=True,right_index=True)
    regression_data.to_csv(produces)

    
    # Define important time points

    first_ld_begin = pd.to_datetime('2020-03-02')
    first_ld_end = pd.to_datetime('2020-05-03')
    first_ld_end_7d = first_ld_end - timedelta(7)

    second_ld_begin = pd.to_datetime('2020-12-09')
    second_ld_end = pd.to_datetime('2021-02-22')
    second_ld_end_7d = second_ld_end - timedelta(7)

    light_ld_begin = pd.to_datetime('2020-10-15')

    #first_ld_begin = pd.to_datetime('2020-03-22')
    #first_ld_end = pd.to_datetime('2020-05-03')
    #first_ld_end_7d = first_ld_end - timedelta(7)

    #second_ld_begin = pd.to_datetime('2020-12-16')
    #second_ld_end = pd.to_datetime('2021-02-22')
    #second_ld_end_7d = second_ld_end - timedelta(7)

    #light_ld_begin = pd.to_datetime('2020-10-15')

    # Create time variables

    regression_data["first_ld"] = ((regression_data.index >= first_ld_begin) & (regression_data.index <= first_ld_end_7d)).astype(int)
    regression_data["second_ld"] = ((regression_data.index >= second_ld_begin) & (regression_data.index <= second_ld_end_7d)).astype(int)
    regression_data["light_ld"] = ((regression_data.index >= light_ld_begin) & (regression_data.index <= second_ld_begin)).astype(int)

    regression_data["first_ld_duration"] = 0
    regression_data.loc[regression_data["first_ld"] == 1, "first_ld_duration"] = regression_data.loc[regression_data["first_ld"] == 1].reset_index().index +1

    regression_data["second_ld_duration"] = 0
    regression_data.loc[regression_data["second_ld"] == 1, "second_ld_duration"] = regression_data.loc[regression_data["second_ld"] == 1].reset_index().index +1

    regression_data["light_ld_duration"] = 0
    regression_data.loc[regression_data["light_ld"] == 1, "light_ld_duration"] = regression_data.loc[regression_data["light_ld"] == 1].reset_index().index +1

        # Specify dependent variables

    depvars = ["workplaces_avg_7d","retail_and_recreation_avg_7d","grocery_and_pharmacy_avg_7d","transit_stations_avg_7d","residential_avg_7d"]

         # Set up model specifications

    model_baseline = "first_ld + second_ld  + first_ld_duration * stringency_index_avg_7d + second_ld_duration * stringency_index_avg_7d"
    model_lockdown_interaction = "first_ld * stringency_index_avg_7d + first_ld_duration * stringency_index_avg_7d + second_ld * stringency_index_avg_7d + second_ld_duration * stringency_index_avg_7d"
    model_light_ld = "first_ld * stringency_index_avg_7d + first_ld_duration * stringency_index_avg_7d + second_ld * stringency_index_avg_7d + second_ld_duration * stringency_index_avg_7d + light_ld + light_ld_duration"
    model_cases = "first_ld * stringency_index_avg_7d + first_ld_duration * stringency_index_avg_7d + second_ld * stringency_index_avg_7d + second_ld_duration * stringency_index_avg_7d + light_ld + light_ld_duration + new_cases_avg_7d"
    model_cases_cubic = "first_ld * stringency_index_avg_7d + first_ld_duration * stringency_index_avg_7d + second_ld * stringency_index_avg_7d + second_ld_duration * stringency_index_avg_7d + light_ld + light_ld_duration + new_cases_avg_7d + np.power(new_cases_avg_7d,2) + np.power(new_cases_avg_7d,3)"

        # Create specification dictionary

    dict_specifications = {}

    for depvar in depvars:
        dict_specifications[depvar] = [model_baseline,model_lockdown_interaction,model_light_ld,model_cases,model_cases_cubic]

        # Create dict with names and get order of variables

    naming_dict = { "first_ld_duration:stringency_index_avg_7d":"1st Lockdown Duration x Stringency" ,
    "second_ld_duration:stringency_index_avg_7d":"2nd Lockdown Duration x Stringency",
    "stringency_index_avg_7d":"Stringency",
    "first_ld":"1st Lockdown",
    "second_ld":"2nd Lockdown",
    "first_ld_duration":"1st Lockdown Duration",
    "second_ld_duration":"2nd Lockdown Duration",
    "first_ld:stringency_index_avg_7d":"1st Lockdown x Stringency",
    "second_ld:stringency_index_avg_7d":"2nd Lockdown x Stringency",
    "light_ld":"Light Lockdown",
    "light_ld_duration":"Light Lockdown Duration",
    "new_cases_avg_7d":"New cases",
    "np.power(new_cases_avg_7d, 2)":"New cases Squared",
    "np.power(new_cases_avg_7d, 3)":"New cases Cubic",
    }

    variable_order = list(naming_dict.keys())

        # Create final tables

    final_tables = ols_regression_formatted(data=regression_data,specifications=dict_specifications, as_latex=False, covariates_names=naming_dict, covariates_order=variable_order)
    final_tables_latex = ols_regression_formatted(data=regression_data,specifications=dict_specifications, as_latex=True, covariates_names=naming_dict, covariates_order=variable_order)