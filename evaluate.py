import os
import pandas as pd
import numpy as np
import argparse

START_DATE = pd.to_datetime('2020-11-16', format='%Y-%m-%d')
END_DATE = pd.to_datetime('2020-11-30', format='%Y-%m-%d')
WINDOW_SIZE = 7

def get_predictions_from_file(predictions_file, ma_df):
    preds_df = pd.read_csv(predictions_file,
                           parse_dates=['Date'],
                           encoding="ISO-8859-1",
                           error_bad_lines=False)
#     preds_df["RegionName"] = preds_df["RegionName"].fillna("")
    preds_df["Prediction"] = True
    
    # Append the true number of cases before start date
    ma_df["Prediction"] = False
    preds_df = ma_df.append(preds_df, ignore_index=True)

    # Add GeoID column that combines CountryName and RegionName for easier manipulation of data
    # np.where usage: if A then B else C
    preds_df["GeoID"] = np.where(preds_df["RegionName"].isnull(),
                                 preds_df["CountryName"],
                                 preds_df["CountryName"] + ' / ' + preds_df["RegionName"])
    # Sort
    preds_df.sort_values(by=["GeoID","Date"], inplace=True)
    # Compute the 7 days moving average for PredictedDailyNewCases
    preds_df["PredictedDailyNewCases7DMA"] = preds_df.groupby(
        "GeoID")['PredictedDailyNewCases'].rolling(
        WINDOW_SIZE, center=False).mean().reset_index(0, drop=True)

    return preds_df

def evaluate(prediction_file):
    actual_df = pd.read_csv('s3://georgianpartners-covid-hackathon/test_actual_df_v2.csv', parse_dates=['Date'],
                           encoding="ISO-8859-1",
                           error_bad_lines=False)
#     actual_df["RegionName"] = actual_df["RegionName"].fillna("")

    ma_df = ma_df = actual_df[(actual_df["Date"] < START_DATE)]
    ma_df = ma_df[["CountryName", "RegionName", "Date", "ActualDailyNewCases"]]
    ma_df = ma_df.rename(columns={"ActualDailyNewCases": "PredictedDailyNewCases"})
    ma_df.head()
    
    preds_df = get_predictions_from_file(prediction_file, ma_df)
    preds_df = preds_df[preds_df['Date'] >= START_DATE]
    preds_df = preds_df[preds_df['Date'] <= END_DATE]
    merged_df = actual_df.merge(preds_df, on=['CountryName', 'RegionName', 'Date', 'GeoID'], how='right')
    ranking_df = pd.DataFrame()
    ranking_df = ranking_df.append(merged_df)
    ranking_df['DiffDaily'] = (ranking_df["ActualDailyNewCases"] - ranking_df["PredictedDailyNewCases"]).abs()
    ranking_df['Diff7DMA'] = (ranking_df["ActualDailyNewCases7DMA"] - ranking_df["PredictedDailyNewCases7DMA"]).abs()
    ranking_df['CumulDiff7DMA'] = ranking_df.groupby(["GeoID"])['Diff7DMA'].cumsum()
    ranking_df = ranking_df[ranking_df["Date"] >= START_DATE]
    ranking_df = ranking_df[ranking_df["Date"] <= END_DATE]
    
    geos = ranking_df.GeoID.unique()
    
    total_loss = 0
    for g in geos:
        total_loss = ranking_df[ranking_df['GeoID'] == g].Diff7DMA.sum()/(ranking_df[ranking_df.GeoID == g].Population.iloc[0]/100000)
    
    return total_loss

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--prediction_file",
                        dest="prediction_file",
                        type=str,
                        required=True)
    args = parser.parse_args()
    print("Computing evaluation score...")
    print("Diff7DMA is " + str(evaluate(args.prediction_file)))
    print("Done!")