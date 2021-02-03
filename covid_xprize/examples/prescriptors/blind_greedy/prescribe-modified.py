# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

"""
This is the prescribe.py script for a simple example prescriptor that
generates IP schedules that trade off between IP cost and cases.

The prescriptor is "blind" in that it does not consider any historical
data when making its prescriptions.

The prescriptor is "greedy" in that it starts with all IPs turned off,
and then iteratively turns on the unused IP that has the least cost.

Since each subsequent prescription is stricter, the resulting set
of prescriptions should produce a Pareto front that highlights the
trade-off space between total IP cost and cases.

Note this file has significant overlap with ../random/prescribe.py.
"""

import os
import argparse
import numpy as np
import pandas as pd

# Generate 1 prescription instead of 10--use heuristics to generate the rest

NUM_PRESCRIPTIONS = 1

IP_MAX_VALUES = {
    'C1_School closing': 3,
    'C2_Workplace closing': 3,
    'C3_Cancel public events': 2,
    'C4_Restrictions on gatherings': 4,
    'C5_Close public transport': 2,
    'C6_Stay at home requirements': 3,
    'C7_Restrictions on internal movement': 2,
    'C8_International travel controls': 4,
    'H1_Public information campaigns': 2,
    'H2_Testing policy': 3,
    'H3_Contact tracing': 2,
    'H6_Facial Coverings': 4
}

IP_COLS = list(IP_MAX_VALUES.keys())


def prescribe(start_date_str: str,
              end_date_str: str,
              path_to_hist_file: str,
              path_to_cost_file: str,
              output_file_path) -> None:

    # Load historical IPs, just to extract the geos
    # we need to prescribe for.
    hist_df = pd.read_csv(path_to_hist_file,
                          parse_dates=['Date'],
                          encoding="ISO-8859-1",
                          keep_default_na=False,
                          error_bad_lines=True)

    # Load the IP weights, so that we can use them
    # greedily for each geo.
    weights_df = pd.read_csv(path_to_cost_file, keep_default_na=False)

    # Generate prescriptions
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
    prescription_dict = {
        'PrescriptionIndex': [],
        'CountryName': [],
        'RegionName': [],
        'Date': []
    }
    for ip in IP_COLS:
        prescription_dict[ip] = []

    for country_name in hist_df['CountryName'].unique():
        country_df = hist_df[hist_df['CountryName'] == country_name]
        for region_name in country_df['RegionName'].unique():

            # Sort IPs for this geo by weight
            geo_weights_df = weights_df[(weights_df['CountryName'] == country_name) &
                                        (weights_df['RegionName'] == region_name)][IP_COLS]
            ip_names = list(geo_weights_df.columns)
            ip_weights = geo_weights_df.values[0]
            sorted_ips = [ip for _, ip in sorted(zip(ip_weights, ip_names))]

            # Initialize the IPs to all turned off
            curr_ips = {ip: 0 for ip in IP_MAX_VALUES}

            for prescription_idx in range(NUM_PRESCRIPTIONS):

                # Turn on the next IP
                next_ip = sorted_ips[prescription_idx]
                curr_ips[next_ip] = IP_MAX_VALUES[next_ip]

                # Use curr_ips for all dates for this prescription
                for date in pd.date_range(start_date, end_date):
                    prescription_dict['PrescriptionIndex'].append(prescription_idx)
                    prescription_dict['CountryName'].append(country_name)
                    prescription_dict['RegionName'].append(region_name)
                    prescription_dict['Date'].append(date.strftime("%Y-%m-%d"))
                    for ip in IP_COLS:
                        prescription_dict[ip].append(curr_ips[ip])

    # Create dataframe from dictionary.
    prescription_df = pd.DataFrame(prescription_dict)
    
    # Generate new IPs using heuristics!!
    impact_file = "/Users/mayaburhanpurkar/Documents/georgian-xprize/covid-xprize/covid_xprize/examples/prescriptors/blind_greedy/weights.csv"
    impact_df = pd.read_csv(impact_file,
                            encoding="ISO-8859-1",
                            keep_default_na=False,
                            error_bad_lines=True)
    impact_df = impact_df.loc[:, ~impact_df.columns.str.contains('^Unnamed')]
    prescription_df = generate_heuristics(start_date, end_date, prescription_df, weights_df, impact_df)
    
    # Create the directory for writing the output file, if necessary.
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Save output csv file.
    prescription_df.to_csv(output_file_path, index=False)

    return 


def generate_heuristics(start_date, end_date, prescriptions, weights_df, impact_df):
    """
    Takes as input a prescription DataFrame (prescriptions) with one prescription per (country, region, 
    date) (i.e. PrescriptionIndex=0 for all) and generates new prescriptions based on heuristics
    """

    costs_dict = {
        'CountryName': [],  'RegionName': [],
        'MaxCostVal': [],   'MaxCostId': [],
        'MinCostVal': [],   'MinCostId': [],
    }
    costs_dict['CountryName'].append(weights_df['CountryName'].tolist()[0])
    costs_dict['RegionName'].append(weights_df['RegionName'].tolist()[0])
    costs = weights_df.drop(columns=['CountryName', 'RegionName'])
    costs_dict['MaxCostVal'].append(costs.max(axis=1).values[0])
    costs_dict['MaxCostId'].append(costs.idxmax(axis=1).values[0])
    costs_dict['MinCostVal'].append(costs.min(axis=1).values[0])
    costs_dict['MinCostId'].append(costs.idxmin(axis=1).values[0])
    costs_df = pd.DataFrame(costs_dict)
                
    prescription_dict = {
        'PrescriptionIndex': [],
        'CountryName': [],
        'RegionName': [],
        'Date': []
    }
    for ip in IP_COLS:
        prescription_dict[ip] = []
                
    # Generate the prescriptions
    for country_name in prescriptions['CountryName'].unique():
        country_df = prescriptions[prescriptions['CountryName'] == country_name]
        for region_name in country_df['RegionName'].unique():
            for date in pd.date_range(start_date, end_date):
                # Extract the current IP (at index 0)
                current_IP = country_df[(country_df['RegionName'] == region_name) & 
                                        (country_df['Date'] == date.strftime("%Y-%m-%d"))]
                
                prescription_dict['PrescriptionIndex'].append(0)
                prescription_dict['CountryName'].append(country_name)
                prescription_dict['RegionName'].append(region_name)
                prescription_dict['Date'].append(date.strftime("%Y-%m-%d"))
                for ip in IP_COLS:
                    prescription_dict[ip].append(current_IP[ip].values[0])
                
                
                mincostid = costs_df[(country_df['RegionName'] == region_name) & 
                                     (country_df['CountryName'] == country_name)]['MinCostId'].values[0]
                maxcostid = costs_df[(country_df['RegionName'] == region_name) & 
                                     (country_df['CountryName'] == country_name)]['MaxCostId'].values[0]

                # Get the the impact of each intervention in the IP from the impact_df
                hi_impacts = dict()
                lo_impacts = dict()
                for ip in IP_COLS:
                    # Get the impacts for incrementing and decrementing
                    if region_name == "":
                        rname = "nan"           
                    else:
                        rname = region_name
                    hi_impacts[ip] = impact_df[(impact_df['Country_Region'] == country_name + "_" + rname) & 
                                             (impact_df['IP Val'] == min(current_IP[ip].values[0] + 1, IP_MAX_VALUES[ip])) &
                                             (impact_df['IP'] == ip)]['impact'].values[0]

                    lo_impacts[ip] = impact_df[(impact_df['Country_Region'] == country_name + "_" + rname) & 
                                             (impact_df['IP Val'] == max(current_IP[ip].values[0] - 1, 1)) &
                                             (impact_df['IP'] == ip)]['impact'].values[0]
                # Get max and min impact id
                minimpactid = min(lo_impacts, key=lo_impacts.get)
                maximpactid = max(hi_impacts, key=lo_impacts.get)
                
                for prescription_idx in range(NUM_PRESCRIPTIONS, 5):
                    # We want to generate 9 prescriptions from each prescription given
                    prescription_dict['PrescriptionIndex'].append(prescription_idx)
                    prescription_dict['CountryName'].append(country_name)
                    prescription_dict['RegionName'].append(region_name)
                    prescription_dict['Date'].append(date.strftime("%Y-%m-%d"))
                    
                    if prescription_idx == 1:
                        # Increase min cost IP by 1
                        for ip in IP_COLS:
                            if ip == mincostid:
                                prescription_dict[ip].append(min(current_IP[ip].values[0] + 1, IP_MAX_VALUES[ip]))
                            else:
                                prescription_dict[ip].append(current_IP[ip].values[0])
                    
                    elif prescription_idx == 2:
                        # Increment min cost IP by 1
                        for ip in IP_COLS:
                            if ip == maxcostid:
                                prescription_dict[ip].append(max(current_IP[ip].values[0] - 1, 0))
                            else:
                                prescription_dict[ip].append(current_IP[ip].values[0])
                                
                    elif prescription_idx == 3:
                        # Increment max impact IP by 1
                        for ip in IP_COLS:
                            if ip == maximpactid:
                                prescription_dict[ip].append(min(current_IP[ip].values[0] + 1, IP_MAX_VALUES[ip]))
                            else:
                                prescription_dict[ip].append(current_IP[ip].values[0])
                    
                    elif prescription_idx >= 4:
                        # Decrement min impact IP by 1
                        for ip in IP_COLS:
                            if ip == minimpactid:
                                prescription_dict[ip].append(max(current_IP[ip].values[0] - 1, 0))
                            else:
                                prescription_dict[ip].append(current_IP[ip].values[0])                               
                            
    df = pd.DataFrame(prescription_dict)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to prescribe, included, as YYYY-MM-DD."
                             "For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prescription, included, as YYYY-MM-DD."
                             "For example 2020-08-31")
    parser.add_argument("-ip", "--interventions_past",
                        dest="prev_file",
                        type=str,
                        required=True,
                        help="The path to a .csv file of previous intervention plans")
    parser.add_argument("-c", "--intervention_costs",
                        dest="cost_file",
                        type=str,
                        required=True,
                        help="Path to a .csv file containing the cost of each IP for each geo")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    args = parser.parse_args()
    print(f"Generating prescriptions from {args.start_date} to {args.end_date}...")
    prescribe(args.start_date, args.end_date, args.prev_file, args.cost_file, args.output_file)
    print("Done!")
