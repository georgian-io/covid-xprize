# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import argparse
from collections import defaultdict
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime

import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr

from ortools.linear_solver import pywraplp
from covid_xprize.standard_predictor.xprize_predictor import XPrizePredictor

NPI_COLS_NAMES = ['C1_School closing',
            'C2_Workplace closing',
            'C3_Cancel public events',
            'C4_Restrictions on gatherings',
            'C5_Close public transport',
            'C6_Stay at home requirements',
            'C7_Restrictions on internal movement',
            'C8_International travel controls',
            'H1_Public information campaigns',
            'H2_Testing policy',
            'H3_Contact tracing',
            'H6_Facial Coverings']

NPI_values = [[0,1,2,3],
             [0,1,2,3],
             [0,1,2],
             [0,1,2,3,4],
             [0,1,2],
             [0,1,2,3],
             [0,1,2],
             [0,1,2,3,4],
             [0,1,2],
             [0,1,2,3],
             [0,1,2],
             [0,1,2,3,4]]

NPI_dict = {}
for i, col in enumerate(NPI_COLS_NAMES):
    NPI_dict[col] = NPI_values[i]

col_names = ['PrescriptionIndex', 'CountryName', 'RegionName', 'Date'] + NPI_COLS_NAMES

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

OUT_TEMP_DIR = "/home/xprize/work/rl/out_temp"

def prescribe(start_date: str,
              end_date: str,
              path_to_prior_ips_file: str,
              path_to_cost_file: str,
              output_file_path) -> None:
    
    original_start_date = start_date
    start_date = np.datetime64(start_date)
    if start_date > np.datetime64('2021-01-29'):
        start_date = np.datetime64('2021-01-29')
    end_date = np.datetime64(end_date)
    
    prior_ip_file = pd.read_csv(path_to_prior_ips_file,
                                    parse_dates=['Date'],
                                    encoding="ISO-8859-1",
                                    dtype={"RegionName": str,
                                        "RegionCode": str},
                                    error_bad_lines=False)
    prior_ip_file["GeoID"] = np.where(prior_ip_file["RegionName"].isnull(),
                                        prior_ip_file["CountryName"],
                                        prior_ip_file["CountryName"] + ' / ' + prior_ip_file["RegionName"])
    
    GeoIDs = list(prior_ip_file["GeoID"].unique())
#     print(GeoIDs)

    # get weights
    case_weights_dict = {}

    with open("/home/xprize/work/weights/weights_reformat.pickle", "rb") as file:
        case_weights_dict['case_weights_1'] = pickle.load(file)
    
    with open("/home/xprize/work/weights/weights_7_reformat.pickle", "rb") as file:
        case_weights_dict['case_weights_7'] = pickle.load(file)

    case_weights_names = ['case_weights_1', 'case_weights_7']
#     case_weights_names = ['case_weights_7']
    
    # get stringency
    stringency_weight_df = pd.read_csv(path_to_cost_file)
    stringency_weight_df["GeoID"] = np.where(stringency_weight_df["RegionName"].isnull(),
                                        stringency_weight_df["CountryName"],
                                        stringency_weight_df["CountryName"] + ' / ' + stringency_weight_df["RegionName"])
#     print(stringency_weight_df["GeoID"].unique())
    # process stringency
    stringency_weight = defaultdict(lambda : defaultdict(lambda : defaultdict(np.float64)))
    for index, row in stringency_weight_df.iterrows():
        row_sum = 0
        for col in NPI_COLS_NAMES:
            row_sum +=  row[col]*sum(NPI_dict[col])
    #     row_sum = row[NPI_COLS_NAMES].sum()
        if row_sum == 0:
            row_sum = 1
        for col in NPI_COLS_NAMES:
            for j in range(len(NPI_dict[col])):
    #             tmp.append((stringency_weight_tmp[i] * (NPI_values[i][j]))/stringency_weight_sum)
                stringency_weight[row['GeoID']][col][j] = (row[col] * NPI_dict[col][j])/row_sum
    
    stringency_weight = default_to_regular(stringency_weight)

    # set global variables
    data_url = "/home/xprize/work/data/OxCGRT_latest.csv"
    OxCGRT_latest = pd.read_csv(data_url,
                                    parse_dates=['Date'],
                                    encoding="ISO-8859-1",
                                    dtype={"RegionName": str,
                                        "RegionCode": str},
                                    error_bad_lines=False)
    OxCGRT_latest["GeoID"] = np.where(OxCGRT_latest["RegionName"].isnull(),
                                        OxCGRT_latest["CountryName"],
                                        OxCGRT_latest["CountryName"] + ' / ' + OxCGRT_latest["RegionName"])
    OxCGRT_latest['NewCases'] = OxCGRT_latest.groupby('GeoID').ConfirmedCases.diff().fillna(0)
    initial_day_cases = OxCGRT_latest[OxCGRT_latest['Date'] == '2020-07-31'].set_index('GeoID')[['NewCases']]
    initial_day_cases['NewCases'] = initial_day_cases['NewCases'].replace(0, 1)

    predictor = XPrizePredictor()

    ip_file_path = '/home/xprize/work/prescriptions/'
    preds_file_path = '/home/xprize/work/predictions/'


    day_count = (end_date - start_date)/np.timedelta64(1, 'D')

    previous_day_cases = initial_day_cases
    
    prescriptions_final_df = pd.DataFrame(columns = col_names)
        
    for index, w in enumerate(case_weights_names):
        print('weights: ', w)
        
        prescriptions_total_df = pd.DataFrame(columns = col_names)
        predictions_total = pd.DataFrame()
        
        cur_ip_file_path = ip_file_path + 'prescriptions_initial_' + w + '.csv'
        for i in range(int(day_count) + 1):
            cur_date = start_date + np.timedelta64(i,'D')
            print('prescribing for day ' + str(cur_date))

            prescriptions_total = []
#             print(GeoIDs)
            for geo in GeoIDs:
#                 print(geo)
#                 print(initial_day_cases.loc[geo]['NewCases'])
                cur_case_weight = case_weights_dict[w][geo]
                cur_stringency_weight = stringency_weight[geo]
                
                geo_split = geo.split(' / ')
                country = geo_split[0]
                if len(geo_split) > 1:
                    RegionName = geo_split[1]
                else:
                    RegionName = ''
                    
#                 print('previous: ', previous_day_cases.loc['United Kingdom / England'])
#                 print('initial: ', initial_day_cases.loc['United Kingdom / England'])

                ip_solution = run_opt(previous_day_cases, cur_case_weight, cur_stringency_weight, geo, initial_day_cases)
                prescriptions_total.append([index] + [country] + [RegionName] + [str(cur_date)] + ip_solution)
                
                
            prescriptions_df = pd.DataFrame()
            prescriptions_df = prescriptions_df.append(pd.DataFrame(prescriptions_total))
            prescriptions_df.columns = col_names
            
            prescriptions_total_df = prescriptions_total_df.append(prescriptions_df)

            prescriptions_total_df.to_csv(cur_ip_file_path)

            # predict for next day for all geo
#             print('predicting for day ' + str(cur_date))
            previous_day_cases = predictor.predict(start_date, cur_date, cur_ip_file_path)
            previous_day_cases = previous_day_cases[previous_day_cases['Date'] == cur_date]
            previous_day_cases["GeoID"] = np.where(previous_day_cases["RegionName"].isnull(),
                                                previous_day_cases["CountryName"],
                                                previous_day_cases["CountryName"] + ' / ' + previous_day_cases["RegionName"])
            previous_day_cases = previous_day_cases.set_index('GeoID')[['PredictedDailyNewCases']]
            previous_day_cases.rename(columns={'PredictedDailyNewCases':'NewCases'}, inplace=True)
            
#             print(previous_day_cases)
            
            predictions_total[cur_date] = previous_day_cases['NewCases']
            
        prescriptions_final_df = prescriptions_final_df.append(prescriptions_total_df)
        prescriptions_final_df[prescriptions_final_df['Date'] >= original_start_date].to_csv(output_file_path)
#         final_pred_path = preds_file_path + 'final_predictions_' + w + '_'+ str(cur_date) + '.csv'
#         predictions_total.to_csv(final_pred_path)
    
    print('running heuristics')
    case_weights_original = ['/home/xprize/work/weights/weights.csv', '/home/xprize/work/weights/weights_7.csv']
    heur_result = pd.DataFrame()
    for index in range(2): 
        prescriptions_current = prescriptions_final_df[prescriptions_final_df['PrescriptionIndex'] == index]

        weights_df = pd.read_csv(path_to_cost_file)
        
        impact_df = pd.read_csv(case_weights_original[index])

        result = generate_heuristics(original_start_date, end_date, prescriptions_current, weights_df, impact_df, index)
#         print(result.shape)
        heur_result = heur_result.append(result, ignore_index = True)
#         print(heur_result.shape)
    heur_result.to_csv(output_file_path)
    # raise NotImplementedError

def generate_heuristics(start_date, end_date, prescriptions, weights_df, impact_df, index):
    
    IP_COLS = list(IP_MAX_VALUES.keys())
    
    """
    Takes as input a prescription DataFrame (prescriptions) with one prescription per (country, region, 
    date) (i.e. PrescriptionIndex=0 for all) and generates new prescriptions based on heuristics
    """

    costs_dict = {
        'CountryName': [],  'RegionName': [],
        'MaxCostVal': [],   'MaxCostId': [],
        'MinCostVal': [],   'MinCostId': [],
    }
    new_row = []
    for idx, row in weights_df.iterrows():
        costs = row.drop(index=['CountryName', 'RegionName']).astype(float)
    #     print(costs)
        new_row.append([row['CountryName'], row['RegionName'],  costs.max(),costs.idxmax(),
                        costs.min(), costs.idxmin()])

    costs_df = pd.DataFrame(new_row, columns = ['CountryName', 'RegionName', 'MaxCostVal', 'MaxCostId', 'MinCostVal', 'MinCostId'])
    
#     print(costs_df)
    
    prescription_dict = {
        'PrescriptionIndex': [],
        'CountryName': [],
        'RegionName': [],
        'Date': []
    }
    for ip in IP_COLS:
        prescription_dict[ip] = []
                
    final_rows = []
    # Generate the prescriptions
    for idx, row in prescriptions.iterrows():
        
        if row['RegionName'] == "" or pd.isnull(row['RegionName']):
            rname = "nan"           
        else:
            rname = row['RegionName']
            
        
        country_df = prescriptions[prescriptions['CountryName'] == row['CountryName']]
        
        current_IP = country_df[(country_df['RegionName'].astype(str) == str(row['RegionName'])) & 
                                        (pd.to_datetime(country_df['Date']) == pd.to_datetime(row['Date']))]
        
#         print(costs_df['RegionName'].astype(str))
#         print(str(row['RegionName']))
#         print(costs_df[(costs_df['RegionName'].astype(str) == rname) & (costs_df['CountryName'] == row['CountryName'])])
        mincostid = costs_df[(costs_df['RegionName'].astype(str) == rname) & 
                                    (costs_df['CountryName'] == row['CountryName'])]['MinCostId'].values[0]
                
        maxcostid = costs_df[(costs_df['RegionName'].astype(str) == rname) & 
                             (costs_df['CountryName'] == row['CountryName'])]['MaxCostId'].values[0]
        
        
        hi_impacts = dict()
        lo_impacts = dict()
        for ip in IP_COLS:

            hi_impacts[ip] = impact_df[(impact_df['Country_Region'] == row['CountryName'] + "_" + rname) & 
                                     (impact_df['IP Val'] == min(current_IP[ip].values[0] + 1, IP_MAX_VALUES[ip])) &
                                     (impact_df['IP'] == ip)]['impact'].values[0]

            lo_impacts[ip] = impact_df[(impact_df['Country_Region'] == row['CountryName'] + "_" + rname) & 
                                     (impact_df['IP Val'] == max(current_IP[ip].values[0] - 1, 1)) &
                                     (impact_df['IP'] == ip)]['impact'].values[0]
            
        
        
        # Get max and min impact id
        minimpactid = min(lo_impacts, key=lo_impacts.get)
        maximpactid = max(hi_impacts, key=lo_impacts.get)
        
        new_row_1 = [1, row.CountryName, row.RegionName, row.Date]
        
        for ip in IP_COLS:
            if ip == mincostid:
                new_row_1.append(min(current_IP[ip].values[0] + 1, IP_MAX_VALUES[ip]))
            else:
                new_row_1.append(current_IP[ip].values[0])
            
        new_row_2 = [2, row['CountryName'], row['RegionName'], row['Date']]
        for ip in IP_COLS:
            if ip == maxcostid:
                new_row_2.append(max(current_IP[ip].values[0] - 1, 0))
            else:
                new_row_2.append(current_IP[ip].values[0])
         
        new_row_3 = [3, row['CountryName'], row['RegionName'], row['Date']]
        for ip in IP_COLS:
            if ip == maximpactid:
                new_row_3.append(min(current_IP[ip].values[0] + 1, IP_MAX_VALUES[ip]))
            else:
                new_row_3.append(current_IP[ip].values[0])
        
        new_row_4 = [4, row['CountryName'], row['RegionName'], row['Date']]
        for ip in IP_COLS:
            if ip == minimpactid:
                new_row_4.append(max(current_IP[ip].values[0] - 1, 0))
            else:
                new_row_4.append(current_IP[ip].values[0])
                
        row['PrescriptionIndex'] = 0
                
        final_rows.extend([row.tolist(), new_row_1, new_row_2, new_row_3, new_row_4])
    
    df = pd.DataFrame(final_rows, columns=col_names)
    
    df['PrescriptionIndex'] = df['PrescriptionIndex'] + (index * 5)
    
    return df


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d

def run_opt(previous_day_cases, case_weight, stringency_weight, geo_id, initial_day_cases):
    solver = pywraplp.Solver.CreateSolver('SCIP')
    
    # Create variables
    x = {}
    for i,col in enumerate(NPI_COLS_NAMES):
        for j in range(len(NPI_dict[col])):
            x[i, j] = solver.IntVar(0, 1, (geo_id + '_' + col + '_' + str(NPI_values[i][j])))
    
    # Create contraint

    for i,col in enumerate(NPI_COLS_NAMES):
        solver.Add(solver.Sum([x[i, j] for j in range(len(NPI_dict[col]))]) == 1)
    
    # Create objective function

    objective_terms = []
    objective_terms.append((previous_day_cases.loc[geo_id]['NewCases']/initial_day_cases.loc[geo_id]['NewCases'])) # scale to initial number of cases
    
    for i,col in enumerate(NPI_COLS_NAMES):
        for j in range(len(NPI_dict[col])):
#             print(initial_day_cases.loc[geo_id]['NewCases'])
#             print(previous_day_cases.loc[geo_id]['NewCases'])
#             if initial_day_cases.loc[geo_id]['NewCases'] < 1:
#                 print(geo_id, ' ' , col, ' ', j)
            objective_terms.append(((case_weight[col][j] * x[i,j]) * previous_day_cases.loc[geo_id]['NewCases'])/initial_day_cases.loc[geo_id]['NewCases'])
    for i,col in enumerate(NPI_COLS_NAMES):
        for j in range(len(NPI_dict[col])):
            objective_terms.append(stringency_weight[col][j] * x[i, j])
    solver.Minimize(solver.Sum(objective_terms))
    
    status = solver.Solve()
#     print(status)
    
    solution = []
    for i,col in enumerate(NPI_COLS_NAMES): 
        for j in range(len(NPI_dict[col])):
            if x[i,j].solution_value() > 0.5:
                solution.append(j)
    return solution

# !!! PLEASE DO NOT EDIT. THIS IS THE OFFICIAL COMPETITION API !!!
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
                        dest="prior_ips_file",
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
    prescribe(args.start_date, args.end_date, args.prior_ips_file, args.cost_file, args.output_file)
    print("Done!")

