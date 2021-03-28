"""Read the mobility, infections and stringency index data
from urls, extract the relevant parts for the analysis and save them into csv files.
The data collected here includes:
1. Google mobility index
2. Our World in Data (OWID) infection numbers
3. Our World in Data (OWID) stringency index
"""
import re
from datetime import datetime
from datetime import timedelta

import pandas as pd
import pytask
import requests
from bs4 import BeautifulSoup
from selenium import webdriver

from src.config import SRC


google_url = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
owid_url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"


@pytask.mark.produces(SRC / "original_data" / "google_data.csv")
def task_get_google_data(produces):
    df = pd.read_csv(google_url)
    df.to_csv(produces)


@pytask.mark.produces(SRC / "original_data" / "owid_data.csv")
def task_get_owid_data(produces):
    df = pd.read_csv(owid_url)
    df.to_csv(produces)


@pytask.mark.produces(SRC / "original_data" / "stringency_index_data.csv")
def task_get_stringency_index_data(produces):
    driver = webdriver.Safari()
    driver.get("https://ourworldindata.org/grapher/covid-stringency-index")
    source_code = driver.page_source
    driver.close()

    source_code_html = BeautifulSoup(source_code, "html")
    link_selection = list(source_code_html.find_all("link"))

    link_relevant = list(filter(lambda x: 'as="fetch"' in str(x), link_selection))

    link_csv = re.findall('href="(.*?)"', str(link_relevant))[0]

    base_url = "https://ourworldindata.org"
    csv_url = base_url + link_csv
    stringency_response = requests.get(csv_url)
    data_unformatted = stringency_response.content

    data_unformatted = data_unformatted.decode("UTF-8")
    data_unformatted_split = data_unformatted.split(',"entityKey":')
    data_unformatted_variables = data_unformatted_split[0]
    data_unformatted_entities = data_unformatted_split[1]

    data_unformatted_variables = data_unformatted_variables.replace(
        '{"variables":{"142679":', ""
    )

    data_unformatted_variables_dict = (
        re.findall('(.*),"id"', data_unformatted_variables)[0] + "}"
    )
    data_unformatted_metadata = data_unformatted_variables.replace(
        data_unformatted_variables_dict[:-1], ""
    )

    data_variables = pd.read_json(data_unformatted_variables_dict)

    data_variables.rename(
        columns={
            "years": "day_from_base",
            "entities": "entity_key",
            "values": "stringency_index",
        },
        inplace=True,
    )

    start_day = re.findall('"zeroDay":"(.*?)"', data_unformatted_metadata)[0]

    data_variables["date"] = list(
        map(
            lambda x: datetime.strptime(start_day, "%Y-%m-%d") + timedelta(days=x),
            data_variables["day_from_base"],
        )
    )

    data_variables = data_variables.drop("day_from_base", axis=1)

    data_unformatted_entities_dict = data_unformatted_entities[:-1]
    data_entities = pd.read_json(data_unformatted_entities_dict, orient="index")

    data_entities = data_entities.reset_index()
    data_entities = data_entities.rename(
        columns={"index": "entity_key", "name": "country", "code": "country_code"}
    )

    data_stringency = pd.merge(data_variables, data_entities)

    data_stringency = data_stringency.set_index(["country", "date"])

    data_stringency = data_stringency.drop("entity_key", axis=1)

    data_stringency = data_stringency[["country_code", "stringency_index"]]
    data_stringency.to_csv(produces)
