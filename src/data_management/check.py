import numpy as np
import pandas as pd

from datetime import datetime

def create_mvg_avg(data,varlist,grouping_var,kind="backward",time=7):
    """
    Adds variables of moving average to dataframe
    
    Input: 
        data (df): dataframe containing variables of varlist; index must contain grouping_var
        varlist (list): variables for which moving average should be calculated
        time (float): time span for moving average
    
    Output: 
        out (df): dataframe with additional variables 
    
    """
    suffix = "_avg_" + str(time) + "d"
    varlist_moving_avg = list(map(lambda x: x + suffix,varlist))
    
    out = data
    
    if kind == "backward":
    
        out[varlist_moving_avg] = out[varlist].apply(lambda x: x.groupby(level=grouping_var).rolling(time).mean(),axis=0).reset_index(level=0, drop=True)        
    
    if kind == "forward":
    
        out[varlist_moving_avg] = out[varlist].apply(lambda x: x[::-1].groupby(level=grouping_var).rolling(time).mean()[::-1],axis=0).sort_index(0).reset_index(level=0, drop=True) 
    
    return(out)


european_countries = np.array(['Austria', 'Bosnia and Herzegovina', 'Belgium', 'Bulgaria',
'Belarus', 'Switzerland', 'Czechia', 'Germany', 'Denmark', 'Spain',
'Finland', 'France', 'United Kingdom', 'Georgia', 'Greece',
'Croatia', 'Hungary', 'Ireland', 'Italy', 'Liechtenstein',
'Lithuania', 'Luxembourg', 'Latvia', 'Moldova', 'North Macedonia',
'Malta', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania',
'Serbia', 'Russia', 'Sweden', 'Slovenia', 'Slovakia', 'Turkey',
'Ukraine'], dtype=object)


owid_data = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv")
#owid_data = pd.read_csv("C:/Users/antom/Desktop/Studium/Master3/EPP_projects/covid_19_mobility/src/original_data/owid_data.csv")

# Keep only european countries which are in the Google data
eu_infect_numbers = owid_data.query("location in @european_countries")

# Rename location to country
eu_infect_numbers = eu_infect_numbers.rename(columns={"location":"country"})

# # Take only columns we need (so far)
eu_infect_numbers = eu_infect_numbers.loc[:,("country", "date", "total_cases", "new_cases")]

# # Make date a datetime object
eu_infect_numbers["date"] = list(map(lambda x: datetime.strptime(x,"%Y-%m-%d"),eu_infect_numbers["date"]))

# # Use MultiIndex for better overview
eu_infect_numbers = eu_infect_numbers.set_index(["country", "date"])

# # Generate 7-day simple moving average 
create_mvg_avg(eu_infect_numbers,["new_cases"],"country",kind="forward",time=7)

# Save dataframe as csv
#eu_infect_numbers.to_csv("C:/Users/antom/Desktop/check.csv")
eu_infect_numbers.to_csv("Arbis Path")