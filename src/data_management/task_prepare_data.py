import numpy as np
import pandas as pd
import pytask

from datetime import datetime
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


@pytask.mark.depends_on(SRC/"original_data"/"owid_data.csv")
@pytask.mark.produces(BLD/"data"/"infection_data.csv")
def task_prepare_owid_data(depends_on, produces):
    # Load in OWID data
    owid_data = pd.read_csv(depends_on)

    # Keep only european countries which are in the Google data
    eu_infect_numbers = owid_data[owid_data["location"].isin(european_countries)]

    # Rename location to country
    eu_infect_numbers = eu_infect_numbers.rename(columns={"location":"country"})

    # Take only columns we need (so far)
    eu_infect_numbers = eu_infect_numbers[["country", "date", "total_cases", "new_cases"]]

    # Make date a datetime object
    eu_infect_numbers["date"] = list(map(lambda x: datetime.strptime(x,"%Y-%m-%d"),eu_infect_numbers["date"]))

    # Use MultiIndex for better overview
    eu_infect_numbers = eu_infect_numbers.set_index(["country", "date"])

    # Generate 7-day simple moving average 
    eu_infect_numbers["new_cases_avg_7d"] = eu_infect_numbers["new_cases"].groupby(level="country").rolling(7).mean().reset_index(level=0,drop=True) 

    # Save dataframe as csv
    eu_infect_numbers.to_csv(produces)


@pytask.mark.depends_on({"google": SRC/"original_data"/"google_data.csv", "infection": BLD/"data"/"infection_data.csv"})
@pytask.mark.produces(BLD/"data"/"eu_complete_data.csv")
def task_prepare_data(depends_on, produces):
    # Load in Google data
    google_data = pd.read_csv(depends_on["google"])

    # Keep only european countries in the dataset
    eu_data = google_data[google_data["country_region"].isin(european_countries)]

    # Replace NaN with "country" in "sub_region_1" column
    eu_data["sub_region_1"].replace(np.nan, "country", inplace=True)

    # Drop census_fips_code (contains no information)
    eu_data.drop("census_fips_code",axis=1)

    # Rename variables
    eu_data.columns = map(lambda x: x.replace("_percent_change_from_baseline",""),eu_data.columns)
    eu_data.rename(columns={"country_region":"country"},inplace=True)

    # Drop census_fips_code because contains only NaN's
    eu_data = eu_data.drop(["census_fips_code"],axis=1)

    # Change datatypes of some columns to string
    eu_data[["sub_region_1", "sub_region_2", "metro_area","iso_3166_2_code","place_id"]] = eu_data[["sub_region_1", "sub_region_2", "metro_area","iso_3166_2_code","place_id"]].astype(str)

    # Create variables for date
    eu_data.rename(columns={"date":"date_str"},inplace=True)
    eu_data["date"] = list(map(lambda x: datetime.strptime(x,"%Y-%m-%d"),eu_data["date_str"]))

    eu_data["weekday"] = list(map(lambda x: x.weekday(), eu_data["date"]))
    eu_data["weekday"]= eu_data["weekday"].replace({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})

    eu_data["day"] = list(map(lambda x: x.day, eu_data["date"]))
    eu_data["week"] = list(map(lambda x: x.week, eu_data["date"]))
    eu_data["weekend"] = list(map(lambda x: int(x),(eu_data["weekday"] == "Sat") | (eu_data["weekday"] == "Sun")))
    eu_data["month"] = list(map(lambda x: x.month, eu_data["date"]))
    eu_data["year"] = list(map(lambda x: x.year, eu_data["date"]))

    # Use MultiIndex for better overview
    eu_data = eu_data.set_index(["country", "date"])
    eu_data = eu_data.drop("date_str",axis=1)

    # Rearrange columns 
    eu_data = eu_data[['weekday','day','weekend','week','month','year','country_region_code','sub_region_1','sub_region_2','metro_area','iso_3166_2_code',
                    'place_id','retail_and_recreation','grocery_and_pharmacy','parks',
                    'transit_stations','workplaces','residential']]
    
    ### Keep only country-level data
    eu_country_level_data = eu_data[eu_data["sub_region_1"] == "country"]
    eu_country_level_data = eu_country_level_data.loc[eu_country_level_data["metro_area"] == "nan",]
    eu_country_level_data = eu_country_level_data.drop(["sub_region_1","sub_region_2","metro_area","iso_3166_2_code"],axis=1)

    ### Generate 7-day simple moving average to smooth the data

    var_list = ['retail_and_recreation','grocery_and_pharmacy', 'parks', 'transit_stations', 'workplaces','residential']
    var_list_moving_avg = list(map(lambda x: x + "_avg_7d",var_list))

    eu_country_level_data[var_list_moving_avg] = eu_country_level_data[var_list].apply(lambda x: x.groupby(level="country").rolling(7).mean(),axis=0).reset_index(level=0, drop=True) 

    # Load in infection numbers
    eu_infect_numbers = pd.read_csv(depends_on["infection"])
    eu_infect_numbers = eu_infect_numbers.set_index(["country", "date"])
    
    # Join the two datasets
    eu_complete_data = eu_country_level_data.join(eu_infect_numbers)

    eu_complete_data.to_csv(produces)
