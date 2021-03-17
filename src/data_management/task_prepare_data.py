import numpy as np
import pandas as pd
import pytask

from datetime import datetime
from datetime import timedelta

from src.config import BLD
from src.config import SRC

# Take european countries list from google data
european_countries = np.array(['Austria', 'Bosnia and Herzegovina', 'Belgium', 'Bulgaria',
    'Belarus', 'Switzerland', 'Czechia', 'Germany', 'Denmark', 'Spain',
    'Finland', 'France', 'United Kingdom', 'Georgia', 'Greece',
    'Croatia', 'Hungary', 'Ireland', 'Italy', 'Liechtenstein',
    'Lithuania', 'Luxembourg', 'Latvia', 'Moldova', 'North Macedonia',
    'Malta', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania',
    'Serbia', 'Russia', 'Sweden', 'Slovenia', 'Slovakia', 'Turkey',
    'Ukraine'], dtype=object)

# Several Divisions of Germany
list_city_states = ['Berlin', 'Bremen', 'Hamburg']
list_non_city_states = ['Baden-Württemberg', 'Bavaria', 'Brandenburg', 'Hessen', 'Lower Saxony', 'Mecklenburg-Vorpommern', 'North Rhine-Westphalia', 'Rhineland-Palatinate', 'Saarland', 'Saxony', 'Saxony-Anhalt', 'Schleswig-Holstein', 'Thuringia']
list_former_brd = ['Baden-Württemberg', 'Bavaria', 'Bremen', 'Hamburg', 'Hessen', 'Lower Saxony', 'North Rhine-Westphalia', 'Rhineland-Palatinate', 'Saarland', 'Schleswig-Holstein']
list_former_ddr = [ 'Brandenburg', 'Mecklenburg-Vorpommern', 'Saxony', 'Saxony-Anhalt', 'Thuringia']
list_west_germany = ['Hessen', 'North Rhine-Westphalia',"Rhineland-Palatinate", "Saarland"]
list_south_germany = ['Baden-Württemberg', 'Bavaria']
list_north_germany = ['Bremen', 'Hamburg',"Lower Saxony","Schleswig-Holstein"]
list_east_germany = ['Brandenburg',"Berlin", 'Mecklenburg-Vorpommern', 'Saxony', 'Saxony-Anhalt', 'Thuringia']


def create_date(data,date_name="date"):
    """
    Adds variables for datetime, day, week, weekend, month and year to dataframe

    Input: 
        data (df): dataframe containing a date variable
        data_name (str): name of date variable

    Output: 
        out (df): dataframe with additional variables 

    """

    out = data.rename(columns={date_name:"date_str"})
    out["date"] = list(map(lambda x: datetime.strptime(x,"%Y-%m-%d"),out["date_str"]))

    out["weekday"] = list(map(lambda x: x.weekday(), out["date"]))
    out["weekday"]= out["weekday"].replace({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})

    out["day"] = list(map(lambda x: x.day, out["date"]))
    out["week"] = list(map(lambda x: x.week, out["date"]))
    out["weekend"] = list(map(lambda x: int(x),(out["weekday"] == "Sat") | (out["weekday"] == "Sun")))
    out["month"] = list(map(lambda x: x.month, out["date"]))
    out["year"] = list(map(lambda x: x.year, out["date"]))

    return(out)


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


@pytask.mark.depends_on(SRC/"original_data"/"owid_data.csv")
@pytask.mark.produces(BLD/"data"/"infection_data.csv")
def task_prepare_owid_data(depends_on, produces):
    # Load in OWID data
    owid_data = pd.read_csv(depends_on)

    # Keep only european countries which are in the Google data
    eu_infect_numbers = owid_data.query("location in @european_countries")

    # Rename location to country
    eu_infect_numbers = eu_infect_numbers.rename(columns={"location":"country"})

    # # Take only columns we need (so far)
    eu_infect_numbers = eu_infect_numbers.loc[:,("country", "date", "total_cases", "new_cases")]

    # # Make date a datetime object
    eu_infect_numbers["date"] = list(map(lambda x: datetime.strptime(x,"%Y-%m-%d"),eu_infect_numbers["date"]))
    eu_infect_numbers["test1"] = 333

    # # Use MultiIndex for better overview
    eu_infect_numbers = eu_infect_numbers.set_index(["country", "date"])

    # # Generate 7-day simple moving average 
    create_mvg_avg(eu_infect_numbers,["new_cases"],"country",kind="forward",time=7)

    # Save dataframe as csv
    eu_infect_numbers.to_csv(produces)



@pytask.mark.depends_on({"google": SRC/"original_data"/"google_data.csv", "infection": BLD/"data"/"infection_data.csv"})
@pytask.mark.produces({"german_states": BLD/"data"/"german_states_data.csv", "eu_country_level": BLD/"data"/"eu_composed_data_country_level.csv"})
def task_prepare_data(depends_on, produces):
    # Load in Google data
    google_data = pd.read_csv(depends_on["google"])

    # Keep only european countries in the dataset
    eu_data = google_data.query("country_region in @european_countries")

    # Replace NaN with "country" in "sub_region_1" column
    eu_data["sub_region_1"] = eu_data["sub_region_1"].replace(np.nan, "country")

    # Drop census_fips_code (contains no information)
    eu_data.drop("census_fips_code",axis=1)

    # Rename variables
    eu_data.columns = map(lambda x: x.replace("_percent_change_from_baseline",""),eu_data.columns)
    eu_data.rename(columns={"country_region":"country"},inplace=True)

    # Drop census_fips_code because contains only NaN's
    eu_data = eu_data.drop(["census_fips_code"],axis=1)

    # Change datatypes of some columns to string
    eu_data[["sub_region_1", "sub_region_2", "metro_area","iso_3166_2_code","place_id"]] = eu_data[
            ["sub_region_1", "sub_region_2", "metro_area","iso_3166_2_code","place_id"]
            ].astype(str)

    # Create date variables
    eu_data = create_date(eu_data)

    # Use MultiIndex for better overview
    eu_data = eu_data.set_index(["country", "date"])
    eu_data = eu_data.drop("date_str",axis=1)
    
    #reate dataset for state-level comparison
    germany_state_level = eu_data.loc["Germany"]
    germany_state_level = germany_state_level.drop(['place_id','metro_area','sub_region_2',"country_region_code"],axis=1)
    germany_state_level = germany_state_level.rename(columns = {'sub_region_1':'state'})

    germany_state_level = germany_state_level.reset_index()
    germany_state_level = germany_state_level.set_index(['state',"date"])
    
    germany_state_level.loc[list_city_states,"city_noncity"] = "city state"
    germany_state_level.loc[list_non_city_states,"city_noncity"] = "territorial state"
    germany_state_level.loc[list_former_brd,"brd_ddr"] = "former BRD"
    germany_state_level.loc[list_former_ddr,"brd_ddr"] = "former DDR"
    germany_state_level.loc[list_west_germany,"four_regions"] = "West"
    germany_state_level.loc[list_south_germany,"four_regions"] = "South"
    germany_state_level.loc[list_north_germany,"four_regions"] = "North"
    germany_state_level.loc[list_east_germany,"four_regions"] = "East"

    germany_state_level = germany_state_level.drop("country")

    germany_state_level = create_mvg_avg(germany_state_level,['retail_and_recreation','grocery_and_pharmacy', 'parks', 'transit_stations', 'workplaces','residential'],"state",kind="forward")
    germany_state_level.to_csv(produces["german_states"])

    # Create dataset for comparison between different european countries
    eu_country_level_data = eu_data[eu_data["sub_region_1"] == "country"]
    eu_country_level_data = eu_country_level_data.loc[eu_country_level_data["metro_area"] == "nan",]
    eu_country_level_data = eu_country_level_data.drop(["sub_region_1","sub_region_2","metro_area","iso_3166_2_code"],axis=1)

    # Create moving average
    eu_country_level_data = create_mvg_avg(eu_country_level_data,['retail_and_recreation','grocery_and_pharmacy', 'parks', 'transit_stations', 'workplaces','residential'],"country",kind="forward")
    
    # Load in infection numbers
    eu_infect_numbers = pd.read_csv(depends_on["infection"])
    eu_infect_numbers = eu_infect_numbers.set_index(["country", "date"])
    
    # Join the two datasets
    eu_composed_data_country_level = eu_country_level_data.join(eu_infect_numbers)
    eu_composed_data_country_level = eu_composed_data_country_level.reset_index()

    eu_composed_data_country_level.to_csv(produces["eu_country_level"])


@pytask.mark.depends_on(SRC / "original_data" / "stringency_index_data.csv")
@pytask.mark.produces(BLD / "data" / "german_stringency_data.csv")
def task_prepare_data(depends_on, produces):
    stringency_data = pd.read_csv(depends_on)
    stringency_data = stringency_data.set_index("country")
    german_stringency_data = stringency_data.loc["Germany"].drop("country_code", axis=1)
    german_stringency_data.to_csv(produces)


