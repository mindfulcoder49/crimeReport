import functions_framework
import os
import requests
import pandas as pd
from haversine import haversine
from openai import OpenAI
from flask import jsonify
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
client = OpenAI()

@functions_framework.http
def main(request):
    street_number = request.args.get('street_number', '')
    street_name = request.args.get('street_name', '')
    street_suffix = request.args.get('street_suffix', '')
    street_prefix = request.args.get('street_prefix', '')
    radius = float(request.args.get('radius', 0.5))
    language = request.args.get('language', 'English')

    #concatenate: street_number, street_name, street_suffix, radius, language
    params = f'{street_number}, {street_prefix}, {street_name}, {street_suffix}, {radius}, {language}'

    coordinatesuccess, coordinates = get_coordinates(street_number, street_prefix, street_name, street_suffix)

    
    if not coordinatesuccess:
        return jsonify({"error": "No matching address found.", "params" : params, "coordinates" : coordinates}), 404

    datasets = {
        'crime-incident-reports-august-2015-to-date-source-new-system': 'Crime Incident Reports'
    }

    dfs = load_all_datasets(datasets)
    dfs['crime-incident-reports-august-2015-to-date-source-new-system'] = access_crime_incident_reports_location(
        dfs['crime-incident-reports-august-2015-to-date-source-new-system']
    )
    dfs['crime-incident-reports-august-2015-to-date-source-new-system'] = access_date_crime_incident_reports(
        dfs['crime-incident-reports-august-2015-to-date-source-new-system']
    )

    dataset_id_array = ['crime-incident-reports-august-2015-to-date-source-new-system']
    location_filtered_result = filter_datasets_by_location(dfs, coordinates[0], coordinates[1], radius, dataset_id_array)

    start_date = pd.to_datetime('2024-04-01', utc=True)
    end_date = pd.to_datetime('2024-05-11', utc=True)
    date_filtered_result = filter_datasets_by_date(location_filtered_result, start_date, end_date)

    prompt_intro = f"The following is a list of crime reports with records limited to a central location. Provide a comprehensive report of the activities in {language} that includes location, time, and date specifics, with a focus on what might affect the average resident, investor, or property owner\n"
    report = generate_report(date_filtered_result, prompt_intro)

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "user", "content": report}
        ]
    )

    response_content = completion.choices[0].message.content

    return jsonify({"report": report, "analysis": response_content})



def get_coordinates(street_number, street_prefix, street_name, street_suffix_abr):
    resource_id = '6d6cfc99-6f26-4974-bbb3-17b5dbad49a9'
    base_url = f'https://data.boston.gov/api/3/action/datastore_search_sql'
    street_name = street_name.title()
    street_suffix_abr = street_suffix_abr.title()
    #make sure street_prefix is capitalized
    street_prefix = street_prefix.upper()
    #construct the SQL query leaving out any blank fields
    sql_query = f'SELECT * FROM "{resource_id}" WHERE'
    if street_number != '':
        sql_query += f' "STREET_NUMBER" = \'{street_number}\''
    if street_name != '':
        sql_query += f' AND "STREET_BODY" LIKE \'{street_name}\''
    if street_suffix_abr != '':
        sql_query += f' AND "STREET_SUFFIX_ABBR" = \'{street_suffix_abr}\''
    if street_prefix != '':
        sql_query += f' AND "STREET_PREFIX" = \'{street_prefix}\''


    params = {'sql': sql_query}

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        results = response.json().get('result', [])
        if results and 'records' in results and results['records']:
            coordinates = (float(results['records'][0]['Y']), float(results['records'][0]['X']))
            return "Success", coordinates
        else:
            return None, f"get_coordinates error: No matching address found. {results}"
    else:
        return None, f"get_coordinates error: Failed to fetch data. {response.status_code}"

def load_all_datasets(datasets):
    dfs = {}
    api_base_url = 'https://data.boston.gov/api/3/action/package_show?id='

    for dataset_id, dataset_name in datasets.items():
        response = requests.get(api_base_url + dataset_id)
        if response.status_code == 200:
            print(f'reading {dataset_name}')
            package = response.json()['result']

            csv_resources = [resource for resource in package['resources'] if resource['format'].lower() == 'csv']

            if csv_resources:
                most_recent_csv_resource = csv_resources[0]
                data_url = most_recent_csv_resource['url']
                df = pd.read_csv(data_url, on_bad_lines='warn')
                dfs[dataset_id] = df
                print(f'Dataframe for {dataset_name} (CSV) created.')
            else:
                print(f'No CSV resources found for {dataset_name}.')
        else:
            print(f'Failed to fetch data for {dataset_name}.')
    return dfs

def access_crime_incident_reports_location(df):
    df['lat'] = pd.to_numeric(df['Lat'], errors='coerce')
    df['long'] = pd.to_numeric(df['Long'], errors='coerce')
    return df

def access_date_crime_incident_reports(df):
    df['date'] = pd.to_datetime(df['OCCURRED_ON_DATE'], errors='coerce')
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('UTC')
    return df

def filter_datasets_by_location(dfs, lat, lon, radius, dataset_ids):
    def is_within_radius(lat1, lon1, lat2, lon2, radius):
        return haversine((lat1, lon1), (lat2, lon2)) <= radius

    filtered_dfs = {}
    for dataset_id in dataset_ids:
        if dataset_id in dfs:
            df = dfs[dataset_id]
            if 'lat' in df.columns and 'long' in df.columns:
                mask = df.apply(lambda row: is_within_radius(lat, lon, row['lat'], row['long'], radius), axis=1)
                filtered_dfs[dataset_id] = df[mask]
            else:
                print(f"Latitude/Longitude columns not found in {dataset_id}")

    return filtered_dfs

def filter_datasets_by_date(dfs, start_date, end_date):
    filtered_dfs = {}
    for dataset_id, df in dfs.items():
        print(f'filtering {dataset_id}')
        if 'date' in df.columns:
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        else:
            print(f"No suitable date column found in {dataset_id}, returning original DataFrame.")
            filtered_dfs[dataset_id] = df
            continue

        filtered_dfs[dataset_id] = df.loc[mask]
    return filtered_dfs

def generate_report(dfs, prompt_intro):
    report = prompt_intro
    for dataset_id, df in dfs.items():
        report += f"\nDataset: {dataset_id}\n"
        report += df.to_string(index=False, header=True)
        report += "\n\n"
    return report
