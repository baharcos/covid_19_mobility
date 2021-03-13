'''Downloads the mobility data from both Apple and Google.
'''
import pandas as pd
import pytask
import csv
import requests
import re

from src.config import BLD
from src.config import SRC

# Set up the correct url
base_url = "https://covid19-static.cdn-apple.com/covid19-mobility-data/current/v3/index.json"
base_url_text = requests.get(base_url).text
base_path = re.findall("/covid19-mobility-data/.+?/v3", base_url_text)
csv_path = re.findall("/en-us/applemobilitytrends-.+?.csv", base_url_text)
apple_url = "https://covid19-static.cdn-apple.com" + base_path[0] + csv_path[0]

google_url = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"

owid_url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"


@pytask.mark.produces(SRC/"original_data"/"apple_data.csv")
def task_get_apple_data(produces):
    df = pd.read_csv(apple_url)
    df.to_csv(produces) 

@pytask.mark.produces(SRC/"original_data"/"google_data.csv")
def task_get_google_data(produces):
    df = pd.read_csv(google_url)
    df.to_csv(produces)

@pytask.mark.produces(SRC/"original_data"/"owid_data.csv")
def task_get_owid_data(produces):
    df = pd.read_csv(owid_url)
    df.to_csv(produces)
