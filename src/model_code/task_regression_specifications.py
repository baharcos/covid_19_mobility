"""
This file defines the relevant model specifications for the regression analysis.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pytask
import pickle

from collections import Counter
from ordered_set import OrderedSet
from stargazer.stargazer import Stargazer
from src.config import BLD
from src.config import SRC
from datetime import datetime
from datetime import timedelta

@pytask.mark.produces({"time_lockdowns":SRC/"model_specs"/"time_lockdowns.pkl","regression_models":SRC/"model_specs"/"regression_models.pkl"})
def task_define_regression_specifications(depends_on, produces):

    # Define lockdown time periods
    dict_time_lockdowns = {"first_lockdown":["2020-03-02","2020-05-03"],"second_lockdown":["2020-12-09","2021-02-22"],"light_lockdown":["2020-10-15","2021-02-22"]}

    # Define dependent variables
    depvars = ["workplaces_avg_7d","retail_and_recreation_avg_7d","grocery_and_pharmacy_avg_7d","transit_stations_avg_7d","residential_avg_7d"]

    # Set up model specifications
    model_baseline = "first_ld + second_ld  + first_ld_duration * stringency_index_avg_7d + second_ld_duration * stringency_index_avg_7d"
    model_lockdown_interaction = "first_ld * stringency_index_avg_7d + first_ld_duration * stringency_index_avg_7d + second_ld * stringency_index_avg_7d + second_ld_duration * stringency_index_avg_7d"
    model_light_lockdown = "first_ld * stringency_index_avg_7d + first_ld_duration * stringency_index_avg_7d + second_ld * stringency_index_avg_7d + second_ld_duration * stringency_index_avg_7d + light_ld + light_ld_duration"
    model_cases = "first_ld * stringency_index_avg_7d + first_ld_duration * stringency_index_avg_7d + second_ld * stringency_index_avg_7d + second_ld_duration * stringency_index_avg_7d + light_ld + light_ld_duration + new_cases_avg_7d"
    model_cases_cubic = "first_ld * stringency_index_avg_7d + first_ld_duration * stringency_index_avg_7d + second_ld * stringency_index_avg_7d + second_ld_duration * stringency_index_avg_7d + light_ld + light_ld_duration + new_cases_avg_7d + np.power(new_cases_avg_7d,2) + np.power(new_cases_avg_7d,3)"

    # Create regression model dictionary
    dict_regression_models = {}

    for depvar in depvars:
        dict_regression_models[depvar] = [model_baseline,model_lockdown_interaction,model_light_lockdown,model_cases,model_cases_cubic]

    # Export specifications to pickle format 

    file_time_lockdowns = open(produces["time_lockdowns"],"wb")
    pickle.dump(dict_time_lockdowns,file_time_lockdowns)
    
    file_regression_models = open(produces["regression_models"],"wb")
    pickle.dump(dict_regression_models,file_regression_models)

