'''
This task runs the regressions to identify lockdown fatigue. 

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


# %% 
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

new_dict = {"1":"Hallo","2":"Du"}
[*new_dict]



# %%

#@pytask.mark.depends_on({"eu_composed_data_country_level": BLD/"data"/"eu_composed_data_country_level.pkl", "stringency_data":BLD/"data"/"german_stringency_data.pkl","lockdowns_dates":SRC/"model_specifications"/"time_lockdowns.pkl"})
def prepare_regression_data(data_composed,stringency_data,dates_lockdowns):
    """
    Creates dataframe with all necessary variables (especially time variables) for regression

    Input:
    data_composed (df): dataframe containing mobility and infection data
    stringency_data (df): dataframe containing stringency data
    dates_lockdowns (dict): dictionary containing start and end points of lockdowns

    Output:
    regression_data(df): dataframe which contains all variables necessary for regression

    """

    # Read in necessary data
    eu_composed_country_level = pd.read_pickle(data_composed)
    stringency_data = pd.read_pickle(stringency_data)
    dates_lockdowns = pd.read_pickle(dates_lockdowns)

    germany_composed_country_level = eu_composed_country_level.loc[eu_composed_country_level["country"] == "Germany",]
    germany_composed_country_level = germany_composed_country_level.set_index("date")

    stringency_data = stringency_data.reset_index()
    stringency_data = stringency_data.drop("country",axis=1)
    stringency_data["date"] = stringency_data["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    stringency_data["date"] = list(map(lambda x: x.date(), stringency_data["date"]))
    stringency_data = stringency_data.set_index("date")

    # Merge the two datasets
    regression_data = pd.merge(germany_composed_country_level,stringency_data,left_index=True,right_index=True)

    # Drop unnecessary variables
    regression_data = regression_data.drop(["country","country_region_code","place_id"],axis=1)

    # Create necessary time variables
    lockdown_names = [*dates_lockdowns]
    #lockdown_7days_moving_average_names = [map(lambda x: x + "_7days_moving_average",lockdown_names)]

    #lockdown_duration_names = [map(lambda x: x + "_duration",lockdown_names)]
    #lockdown_duration_7days_moving_average_duration_names = [map(lambda x: x + "_duration",lockdown_7days_moving_average_names)]

    for lockdown in lockdown_names:

        lockdown_duration_name = lockdown + "_duration"
        lockdown_7days_moving_average_name = lockdown + "_7days_moving_average"
        lockdown_7days_moving_average_duration_name = lockdown_7days_moving_average_name + "duration"

        regression_data[lockdown] = ((regression_data.index >= datetime.strptime(dates_lockdowns[lockdown][0],"%Y-%m-%d")) & (regression_data.index <= datetime.strptime(dates_lockdowns[lockdown][1],"%Y-%m-%d"))).astype(int)
        regression_data[lockdown_7days_moving_average_name] = ((regression_data.index >= datetime.strptime(dates_lockdowns[lockdown][0],"%Y-%m-%d")) & (regression_data.index <= datetime.strptime(dates_lockdowns[lockdown][1],"%Y-%m-%d") - timedelta(7))).astype(int)
        
        regression_data[lockdown_duration_name] = 0
        regression_data.loc[regression_data[lockdown] == 1, lockdown_duration_name] = regression_data.loc[regression_data[lockdown] == 1].reset_index().index + 1
        regression_data[lockdown_7days_moving_average_duration_name] = 0
        regression_data.loc[regression_data[lockdown_7days_moving_average_name] == 1, lockdown_7days_moving_average_duration_name] = regression_data.loc[regression_data[lockdown_7days_moving_average_name] == 1].reset_index().index + 1
        
    return(regression_data)

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

#@pytask.mark.depends_on({"german_states_mobility": SRC/"original_data"/"german_states_data.csv"})
@pytask.mark.depends_on({"eu_composed_data_country_level": BLD/"data"/"eu_composed_data_country_level.pkl", "stringency_data":BLD/"data"/"german_stringency_data.pkl","dates_lockdowns":SRC/"model_specs"/"time_lockdowns.pkl"})
@pytask.mark.produces(BLD/"data"/"regression_data.csv")
def task_create_regression_data(depends_on, produces):
    regression_data = prepare_regression_data(data_composed=depends_on["eu_composed_data_country_level"], stringency_data=depends_on["stringency_data"],dates_lockdowns=depends_on["dates_lockdowns"]) 
    regression_data.to_csv(produces)

# @pytask.mark.depends_on({"regression_data": BLD/"data"/"regression_data.pkl"})
# def task_run_regressions(depends_on, produces):
#     regression_tables = ols_regression_formatted(data=depends_on["regression_data"],specifications, as_latex=False, covariates_names=None, covariates_order=None)
    

# @pytask.mark.produces(BLD/"data"/"regression_data.pkl")
# def task_create_regression_tables(depends_on, produces):

#     # Define important time points

#     first_ld_begin = pd.to_datetime('2020-03-02')
#     first_ld_end = pd.to_datetime('2020-05-03')
#     first_ld_end_7d = first_ld_end - timedelta(7)

#     second_ld_begin = pd.to_datetime('2020-12-09')
#     second_ld_end = pd.to_datetime('2021-02-22')
#     second_ld_end_7d = second_ld_end - timedelta(7)

#     light_ld_begin = pd.to_datetime('2020-10-15')

#     #first_ld_begin = pd.to_datetime('2020-03-22')
#     #first_ld_end = pd.to_datetime('2020-05-03')
#     #first_ld_end_7d = first_ld_end - timedelta(7)

#     #second_ld_begin = pd.to_datetime('2020-12-16')
#     #second_ld_end = pd.to_datetime('2021-02-22')
#     #second_ld_end_7d = second_ld_end - timedelta(7)

#     #light_ld_begin = pd.to_datetime('2020-10-15')

#     # Create time variables

#     regression_data["first_ld"] = ((regression_data.index >= first_ld_begin) & (regression_data.index <= first_ld_end_7d)).astype(int)
#     regression_data["second_ld"] = ((regression_data.index >= second_ld_begin) & (regression_data.index <= second_ld_end_7d)).astype(int)
#     regression_data["light_ld"] = ((regression_data.index >= light_ld_begin) & (regression_data.index <= second_ld_begin)).astype(int)

#     regression_data["first_ld_duration"] = 0
#     regression_data.loc[regression_data["first_ld"] == 1, "first_ld_duration"] = regression_data.loc[regression_data["first_ld"] == 1].reset_index().index +1

#     regression_data["second_ld_duration"] = 0
#     regression_data.loc[regression_data["second_ld"] == 1, "second_ld_duration"] = regression_data.loc[regression_data["second_ld"] == 1].reset_index().index +1

#     regression_data["light_ld_duration"] = 0
#     regression_data.loc[regression_data["light_ld"] == 1, "light_ld_duration"] = regression_data.loc[regression_data["light_ld"] == 1].reset_index().index +1

#         # Specify dependent variables

#     depvars = ["workplaces_avg_7d","retail_and_recreation_avg_7d","grocery_and_pharmacy_avg_7d","transit_stations_avg_7d","residential_avg_7d"]

#          # Set up model specifications

#     model_baseline = "first_ld + second_ld  + first_ld_duration * stringency_index_avg_7d + second_ld_duration * stringency_index_avg_7d"
#     model_lockdown_interaction = "first_ld * stringency_index_avg_7d + first_ld_duration * stringency_index_avg_7d + second_ld * stringency_index_avg_7d + second_ld_duration * stringency_index_avg_7d"
#     model_light_ld = "first_ld * stringency_index_avg_7d + first_ld_duration * stringency_index_avg_7d + second_ld * stringency_index_avg_7d + second_ld_duration * stringency_index_avg_7d + light_ld + light_ld_duration"
#     model_cases = "first_ld * stringency_index_avg_7d + first_ld_duration * stringency_index_avg_7d + second_ld * stringency_index_avg_7d + second_ld_duration * stringency_index_avg_7d + light_ld + light_ld_duration + new_cases_avg_7d"
#     model_cases_cubic = "first_ld * stringency_index_avg_7d + first_ld_duration * stringency_index_avg_7d + second_ld * stringency_index_avg_7d + second_ld_duration * stringency_index_avg_7d + light_ld + light_ld_duration + new_cases_avg_7d + np.power(new_cases_avg_7d,2) + np.power(new_cases_avg_7d,3)"

#         # Create specification dictionary

#     dict_specifications = {}

#     for depvar in depvars:
#         dict_specifications[depvar] = [model_baseline,model_lockdown_interaction,model_light_ld,model_cases,model_cases_cubic]

#         # Create dict with names and get order of variables

#     naming_dict = { "first_ld_duration:stringency_index_avg_7d":"1st Lockdown Duration x Stringency" ,
#     "second_ld_duration:stringency_index_avg_7d":"2nd Lockdown Duration x Stringency",
#     "stringency_index_avg_7d":"Stringency",
#     "first_ld":"1st Lockdown",
#     "second_ld":"2nd Lockdown",
#     "first_ld_duration":"1st Lockdown Duration",
#     "second_ld_duration":"2nd Lockdown Duration",
#     "first_ld:stringency_index_avg_7d":"1st Lockdown x Stringency",
#     "second_ld:stringency_index_avg_7d":"2nd Lockdown x Stringency",
#     "light_ld":"Light Lockdown",
#     "light_ld_duration":"Light Lockdown Duration",
#     "new_cases_avg_7d":"New cases",
#     "np.power(new_cases_avg_7d, 2)":"New cases Squared",
#     "np.power(new_cases_avg_7d, 3)":"New cases Cubic",
#     }

#     variable_order = list(naming_dict.keys())

#         # Create final tables

#     final_tables = ols_regression_formatted(data=regression_data,specifications=dict_specifications, as_latex=False, covariates_names=naming_dict, covariates_order=variable_order)
#     final_tables_latex = ols_regression_formatted(data=regression_data,specifications=dict_specifications, as_latex=True, covariates_names=naming_dict, covariates_order=variable_order)