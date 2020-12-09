import os

import numpy as np
import pandas as pd

US_PREFIX = "United States / "

ADD_DATA_PATH = "/home/ubuntu/zilun/covid-xprize/covid_xprize/examples/predictors/lstm/data"
ADDITIONAL_CONTEXT_FILE = os.path.join(ADD_DATA_PATH, "Additional_Context_Data_Global.csv")
ADDITIONAL_US_STATES_CONTEXT = os.path.join(ADD_DATA_PATH, "US_states_populations.csv")
ADDITIONAL_UK_CONTEXT = os.path.join(ADD_DATA_PATH, "uk_populations.csv")
ADDITIONAL_BRAZIL_CONTEXT = os.path.join(ADD_DATA_PATH, "brazil_populations.csv")

NPI_COLUMNS = ['C1_School closing',
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

CONTEXT_COLUMNS = ['CountryName',
                   'RegionName',
                   'GeoID',
                   'Date',
                   'ConfirmedCases',
                   'ConfirmedDeaths',
                   'Population']

WINDOW_SIZE = 7
MAX_NB_COUNTRIES = 20
NB_LOOKBACK_DAYS = 21
NB_TEST_DAYS = 14

# Copied from covid_xprize/examples/predictors/lstm/xprize_predictor.py
def _prepare_dataframe(data_url: str) -> pd.DataFrame:
    """
    Loads the Oxford dataset, cleans it up and prepares the necessary columns. Depending on options, also
    loads the Johns Hopkins dataset and merges that in.
    :param data_url: the url containing the original data
    :return: a Pandas DataFrame with the historical data
    """
    # Original df from Oxford
    df1 = _load_original_data(data_url)

    # Additional context df (e.g Population for each country)
    df2 = _load_additional_context_df()

    # Merge the 2 DataFrames
    df = df1.merge(df2, on=['GeoID'], how='left', suffixes=('', '_y'))

    # Drop countries with no population data
    df.dropna(subset=['Population'], inplace=True)

    #  Keep only needed columns
    columns = CONTEXT_COLUMNS + NPI_COLUMNS
    df = df[columns]

    # Fill in missing values
    _fill_missing_values(df)

    # Compute number of new cases and deaths each day
    df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)
    df['NewDeaths'] = df.groupby('GeoID').ConfirmedDeaths.diff().fillna(0)

    # Replace negative values (which do not make sense for these columns) with 0
    df['NewCases'] = df['NewCases'].clip(lower=0)
    df['NewDeaths'] = df['NewDeaths'].clip(lower=0)

    # Compute smoothed versions of new cases and deaths each day
    df['SmoothNewCases'] = df.groupby('GeoID')['NewCases'].rolling(
            WINDOW_SIZE, center=False).mean().fillna(0).reset_index(0, drop=True)
    df['SmoothNewDeaths'] = df.groupby('GeoID')['NewDeaths'].rolling(
            WINDOW_SIZE, center=False).mean().fillna(0).reset_index(0, drop=True)

    # Compute percent change in new cases and deaths each day
    df['CaseRatio'] = df.groupby('GeoID').SmoothNewCases.pct_change().fillna(0).replace(np.inf, 0) + 1
    df['DeathRatio'] = df.groupby('GeoID').SmoothNewDeaths.pct_change().fillna(0).replace(np.inf, 0) + 1

    # Add column for proportion of population infected
    df['ProportionInfected'] = df['ConfirmedCases'] / df['Population']

    # Create column of value to predict
    df['PredictionRatio'] = df['CaseRatio'] / (1 - df['ProportionInfected'])

    return df

# Copied from covid_xprize/examples/predictors/lstm/xprize_predictor.py
def _load_original_data(data_url):
    latest_df = pd.read_csv(data_url,
                            parse_dates=['Date'],
                            encoding="ISO-8859-1",
                            dtype={"RegionName": str,
                                   "RegionCode": str},
                            error_bad_lines=False)
    # GeoID is CountryName / RegionName
    # np.where usage: if A then B else C
    latest_df["GeoID"] = np.where(latest_df["RegionName"].isnull(),
                                  latest_df["CountryName"],
                                  latest_df["CountryName"] + ' / ' + latest_df["RegionName"])
    return latest_df

# Copied from covid_xprize/examples/predictors/lstm/xprize_predictor.py
def _load_additional_context_df():
    # File containing the population for each country
    # Note: this file contains only countries population, not regions
    additional_context_df = pd.read_csv(ADDITIONAL_CONTEXT_FILE,
                                        usecols=['CountryName', 'Population'])
    additional_context_df['GeoID'] = additional_context_df['CountryName']

    # US states population
    additional_us_states_df = pd.read_csv(ADDITIONAL_US_STATES_CONTEXT,
                                          usecols=['NAME', 'POPESTIMATE2019'])
    # Rename the columns to match measures_df ones
    additional_us_states_df.rename(columns={'POPESTIMATE2019': 'Population'}, inplace=True)
    # Prefix with country name to match measures_df
    additional_us_states_df['GeoID'] = US_PREFIX + additional_us_states_df['NAME']

    # Append the new data to additional_df
    additional_context_df = additional_context_df.append(additional_us_states_df)

    # UK population
    additional_uk_df = pd.read_csv(ADDITIONAL_UK_CONTEXT)
    # Append the new data to additional_df
    additional_context_df = additional_context_df.append(additional_uk_df)

    # Brazil population
    additional_brazil_df = pd.read_csv(ADDITIONAL_BRAZIL_CONTEXT)
    # Append the new data to additional_df
    additional_context_df = additional_context_df.append(additional_brazil_df)

    return additional_context_df

# Copied from covid_xprize/examples/predictors/lstm/xprize_predictor.py
def _fill_missing_values(df):
    """
    # Fill missing values by interpolation, ffill, and filling NaNs
    :param df: Dataframe to be filled
    """
    df.update(df.groupby('GeoID').ConfirmedCases.apply(lambda group: group.interpolate(limit_area='inside')))
    # Drop country / regions for which no number of cases is available
    df.dropna(subset=['ConfirmedCases'], inplace=True)
    df.update(df.groupby('GeoID').ConfirmedDeaths.apply(lambda group: group.interpolate(limit_area='inside')))
    # Drop country / regions for which no number of deaths is available
    df.dropna(subset=['ConfirmedDeaths'], inplace=True)
    for npi_column in NPI_COLUMNS:
        df.update(df.groupby('GeoID')[npi_column].ffill().fillna(0))
    return

# Copied from covid_xprize/examples/predictors/lstm/xprize_predictor.py
def _most_affected_geos(df, nb_geos, min_historical_days):
    """
    Returns the list of most affected countries, in terms of confirmed deaths.
    :param df: the data frame containing the historical data
    :param nb_geos: the number of geos to return
    :param min_historical_days: the minimum days of historical data the countries must have
    :return: a list of country names of size nb_countries if there were enough, and otherwise a list of all the
    country names that have at least min_look_back_days data points.
    """
    # By default use most affected geos with enough history
    gdf = df.groupby('GeoID')['ConfirmedDeaths'].agg(['max', 'count']).sort_values(by='max', ascending=False)
    filtered_gdf = gdf[gdf["count"] > min_historical_days]
    geos = list(filtered_gdf.head(nb_geos).index)
    return geos


# Copied from covid_xprize/examples/predictors/lstm/xprize_predictor.py
def _create_country_samples(df: pd.DataFrame, geos: list) -> dict:
    """
    For each country, creates numpy arrays for Keras
    :param df: a Pandas DataFrame with historical data for countries (the "Oxford" dataset)
    :param geos: a list of geo names
    :return: a dictionary of train and test sets, for each specified country
    """
    context_column = 'PredictionRatio'
    action_columns = NPI_COLUMNS
    outcome_column = 'PredictionRatio'
    country_samples = {}
    for g in geos:
        cdf = df[df.GeoID == g]
        cdf = cdf[cdf.ConfirmedCases.notnull()]
        context_data = np.array(cdf[context_column])
        action_data = np.array(cdf[action_columns])
        outcome_data = np.array(cdf[outcome_column])
        context_samples = []
        action_samples = []
        outcome_samples = []
        nb_total_days = outcome_data.shape[0]
        for d in range(NB_LOOKBACK_DAYS, nb_total_days):
            context_samples.append(context_data[d - NB_LOOKBACK_DAYS:d])
            action_samples.append(action_data[d - NB_LOOKBACK_DAYS:d])
            outcome_samples.append(outcome_data[d])
        if len(outcome_samples) > 0:
            X_context = np.expand_dims(np.stack(context_samples, axis=0), axis=2)
            X_action = np.stack(action_samples, axis=0)
            y = np.stack(outcome_samples, axis=0)
            country_samples[g] = {
                'X_context': X_context,
                'X_action': X_action,
                'y': y,
                'X_train_context': X_context[:-NB_TEST_DAYS],
                'X_train_action': X_action[:-NB_TEST_DAYS],
                'y_train': y[:-NB_TEST_DAYS],
                'X_test_context': X_context[-NB_TEST_DAYS:],
                'X_test_action': X_action[-NB_TEST_DAYS:],
                'y_test': y[-NB_TEST_DAYS:],
            }
    return country_samples

if __name__ == "__main__":
    DATA_FILE_PATH = "/home/ubuntu/zilun/covid-xprize/covid_xprize/examples/predictors/data/OxCGRT_latest.csv"

    df = _prepare_dataframe(DATA_FILE_PATH)
    print(df.head())

    geos = _most_affected_geos(df, MAX_NB_COUNTRIES, NB_LOOKBACK_DAYS)
    print(type(geos))
    country_samples = _create_country_samples(df, geos)
    print(type(country_samples))

    # Aggregate data for training
    all_X_context_list = [country_samples[c]['X_train_context']
                          for c in country_samples]
    all_X_action_list = [country_samples[c]['X_train_action']
                         for c in country_samples]
    all_y_list = [country_samples[c]['y_train']
                  for c in country_samples]
    X_context = np.concatenate(all_X_context_list)
    X_action = np.concatenate(all_X_action_list)
    y = np.concatenate(all_y_list)

    print(X_context[:5])
    print(X_action[:5])
    print(y[:5])
