# Adapted from abudhkar/covid-xprize/ken/covid_xprize_pipeline/predictor.py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from rnn_model import Simple_RNN

from constants import (
    NPI_COLUMNS,
    NB_LOOKBACK_DAYS,
    WINDOW_SIZE,
    MAX_NB_COUNTRIES,
    CONTEXT_COLUMNS,
    NB_TEST_DAYS,
    NUM_TRIALS,
    ADDITIONAL_CONTEXT_FILE,
    ADDITIONAL_US_STATES_CONTEXT,
    US_PREFIX,
    ADDITIONAL_UK_CONTEXT,
    ADDITIONAL_BRAZIL_CONTEXT,
    USE_GPU,
    NUM_EPOCHS,
    PATIENCE,
    BATCH_SIZE
)

seed_everything(32)
class XPrizePredictor(object):
    """
    A class that computes a fitness for Prescriptor candidates.
    """

    def __init__(self, data_url, static_df=None, saved_model_path=None,
                 dynamic_df=None):

        self.df = self._prepare_dataframe(data_url)
        self.new_columns = []
        self.static_df = None
        # if dynamic_df is not None:
        #     assert type(dynamic_df) is pd.DataFrame
        #     new_columns = set(dynamic_df.columns)
        #     orig_columns = set(self.df.columns)
        #     new_unique_columns = list(new_columns - orig_columns)
        #     self.new_columns.extend(new_unique_columns)
        #     print(f'New dynamic features:\n{new_unique_columns}')
        #     self.df = self.df.merge(dynamic_df, on=['GeoID', 'Date'], how='left', suffixes=('', '_y'))
        #     self.df = self.df[list(orig_columns.union(new_columns))]

        if static_df is not None:
            assert type(static_df) is pd.DataFrame
            merge_df, new_columns = self._merge_static_df(self.df, static_df)
            print(f'New static features:\n{new_columns}')
            self.static_df = static_df
            self.df = merge_df
            self.new_columns = new_columns

        if saved_model_path:

            # Load model weights
            nb_context = 1 
            #nb_action = len(NPI_COLUMNS)
            nb_action = 46
            nb_static = len(self.new_columns)
            self.predictor = Simple_RNN( nb_context = nb_context,
                                         nb_action = 46,
                                         nb_static = nb_static,
                                         nb_lookback = NB_LOOKBACK_DAYS)
            self.predictor.load_state_dict(torch.load(saved_model_path)["state_dict"])


        geos = self.df.GeoID.unique()
        self.country_samples = self._create_country_samples(self.df, geos,
                                                            new_cols=self.new_columns)


    def _merge_static_df(self, target_df, to_merge_df):
        new_columns = set(to_merge_df.columns)
        orig_columns = set(target_df.columns)
        target_df = target_df.merge(to_merge_df, on=['GeoID'], how='left', suffixes=('', '_y'))
        target_df = target_df[list(orig_columns.union(new_columns))]
        new_unique_columns = list(new_columns - orig_columns)
        return target_df, new_unique_columns

    def predict(self,
                start_date_str: str,
                end_date_str: str,
                path_to_ips_file: str) -> pd.DataFrame:
        start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
        nb_days = (end_date - start_date).days + 1

        # Load the npis into a DataFrame, handling regions
        npis_df = self._load_original_data(path_to_ips_file)
        self._fill_missing_values(npis_df)

        # Prepare the output
        forecast = {"CountryName": [],
                    "RegionName": [],
                    "Date": [],
                    "PredictedDailyNewCases": []}

        # For each requested geo
        geos = npis_df.GeoID.unique()
        for g in tqdm(geos, desc='looping through regions'):
            cdf = self.df[self.df.GeoID == g]
            if len(cdf) == 0:
                # we don't have historical data for this geo: return zeroes
                pred_new_cases = [0] * nb_days
                geo_start_date = start_date
            else:
                last_known_date = cdf.Date.max()
                # Start predicting from start_date, unless there's a gap since last known date
                geo_start_date = start_date#min(last_known_date + np.timedelta64(1, 'D'), start_date)
                npis_gdf = npis_df[(npis_df.Date >= geo_start_date) & (npis_df.Date <= end_date)]
                npis_gdf, _ = self._merge_static_df(npis_gdf, self.static_df)
                pred_new_cases = self._get_new_cases_preds(cdf, g, npis_gdf)

            # Append forecast data to results to return
            country = npis_df[npis_df.GeoID == g].iloc[0].CountryName
            region = npis_df[npis_df.GeoID == g].iloc[0].RegionName
            for i, pred in enumerate(pred_new_cases):
                forecast["CountryName"].append(country)
                forecast["RegionName"].append(region)
                current_date = geo_start_date + pd.offsets.Day(i)
                forecast["Date"].append(current_date)
                forecast["PredictedDailyNewCases"].append(pred)

        forecast_df = pd.DataFrame.from_dict(forecast)
        # Return only the requested predictions
        return forecast_df[(forecast_df.Date >= start_date) & (forecast_df.Date <= end_date)]

    def one_hot_action_input(self, x_action):
        x_action = x_action.astype(int)
        one_hot_x_actions = []
        max_vals = [3, 3, 2, 4, 2, 3, 2, 4, 2, 3, 2, 4]
        num_lookback_days, num_actions = x_action.shape

        for j in range(num_lookback_days):
            action_per_day = []
            for k in range(num_actions):
                cur_action = x_action[j][k]
                one_hot_action = [0]*(max_vals[k]+1)
                one_hot_action[cur_action] = 1
                action_per_day.extend(one_hot_action)
            one_hot_x_actions.append(action_per_day)

        x_action = np.array(one_hot_x_actions)
        x_action = x_action.astype(float)
        return x_action

    def _get_new_cases_preds(self, c_df, g, npis_df):
        cdf = c_df[c_df.ConfirmedCases.notnull()]
        initial_context_input = self.country_samples[g]['X_test_context'][-1]
        initial_action_input = self.country_samples[g]['X_test_action'][-1]
        initial_action_input = self.one_hot_action_input(initial_action_input)
        static_input = self.country_samples[g]['X_test_static'][-1]
        # Predictions with passed npis
        cnpis_df = npis_df[npis_df.GeoID == g]
        npis_sequence = np.array(cnpis_df[NPI_COLUMNS])
        npis_sequence = self.one_hot_action_input(npis_sequence)
        # Get the predictions with the passed NPIs
        preds = self._roll_out_predictions(self.predictor,
                                           initial_context_input,
                                           initial_action_input,
                                           static_input,
                                           npis_sequence)
        # Gather info to convert to total cases
        prev_confirmed_cases = np.array(cdf.ConfirmedCases)
        prev_new_cases = np.array(cdf.NewCases)
        initial_total_cases = prev_confirmed_cases[-1]
        pop_size = np.array(cdf.Population)[-1]  # Population size doesn't change over time
        # Compute predictor's forecast
        pred_new_cases = self._convert_ratios_to_total_cases(
            preds,
            WINDOW_SIZE,
            prev_new_cases,
            initial_total_cases,
            pop_size)

        return pred_new_cases

    def _prepare_dataframe(self, data_url: str) -> pd.DataFrame:
        """
        Loads the Oxford dataset, cleans it up and prepares the necessary columns. Depending on options, also
        loads the Johns Hopkins dataset and merges that in.
        :param data_url: the url containing the original data
        :return: a Pandas DataFrame with the historical data
        """
        # Original df from Oxford
        df1 = self._load_original_data(data_url)

        # Additional context df (e.g Population for each country)
        df2 = self._load_additional_context_df()

        # Merge the 2 DataFrames
        df = df1.merge(df2, on=['GeoID'], how='left', suffixes=('', '_y'))

        # Drop countries with no population data
        df.dropna(subset=['Population'], inplace=True)

        #  Keep only needed columns
        columns = CONTEXT_COLUMNS + NPI_COLUMNS
        df = df[columns]

        # Fill in missing values
        self._fill_missing_values(df)

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
        df['CaseRatio'] = df.groupby('GeoID').SmoothNewCases.pct_change(
        ).fillna(0).replace(np.inf, 0) + 1
        df['DeathRatio'] = df.groupby('GeoID').SmoothNewDeaths.pct_change(
        ).fillna(0).replace(np.inf, 0) + 1

        # Add column for proportion of population infected
        df['ProportionInfected'] = df['ConfirmedCases'] / df['Population']

        # Create column of value to predict
        df['PredictionRatio'] = df['CaseRatio'] / (1 - df['ProportionInfected'])

        return df

    @staticmethod
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

    @staticmethod
    def _fill_missing_values(df):
        """
        # Fill missing values by interpolation, ffill, and filling NaNs
        :param df: Dataframe to be filled
        """
        df.update(df.groupby('GeoID').ConfirmedCases.apply(
            lambda group: group.interpolate(limit_area='inside')))
        # Drop country / regions for which no number of cases is available
        # df['ConfirmedCases'].fillna(method='ffill')
        df.dropna(subset=['ConfirmedCases'], inplace=True)
        df.update(df.groupby('GeoID').ConfirmedDeaths.apply(
            lambda group: group.interpolate(limit_area='inside')))
        # Drop country / regions for which no number of deaths is available
        df['ConfirmedDeaths'].fillna(method='ffill')
        df.dropna(subset=['ConfirmedDeaths'], inplace=True)
        for npi_column in NPI_COLUMNS:
            df.update(df.groupby('GeoID')[npi_column].ffill().fillna(0))

    @staticmethod
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

    @staticmethod
    def _create_country_samples(df: pd.DataFrame, geos: list, new_cols=None) -> dict:
        """
        For each country, creates numpy arrays for Keras
        :param df: a Pandas DataFrame with historical data for countries (the "Oxford" dataset)
        :param geos: a list of geo names
        :return: a dictionary of train and test sets, for each specified country
        """
        if new_cols is None:
            new_cols = []
        context_columns = ['PredictionRatio'] 
        action_columns = NPI_COLUMNS
        outcome_column = 'PredictionRatio'
        country_samples = {}
        for g in geos:
            cdf = df[df.GeoID == g]
            cdf = cdf[cdf.ConfirmedCases.notnull()]
            # print(f'Geo {g} num test days: {num_test_days}')
            context_data = np.array(cdf[context_columns])
            action_data = np.array(cdf[action_columns])
            static_data = np.array(cdf[new_cols])
            outcome_data = np.array(cdf[outcome_column])
            
            context_samples = []
            action_samples = []
            static_samples = []
            outcome_samples = []
            nb_total_days = outcome_data.shape[0]
            for d in range(NB_LOOKBACK_DAYS, nb_total_days):
                context_samples.append(context_data[d - NB_LOOKBACK_DAYS:d])
                action_samples.append(action_data[d - NB_LOOKBACK_DAYS:d])
                static_samples.append(static_data[d])
                outcome_samples.append(outcome_data[d])
            if len(outcome_samples) > 0:
                # if len(new_cols) > 0:
                X_context = np.stack(context_samples, axis=0)
                # else:
                #     X_context = np.expand_dims(np.stack(context_samples, axis=0), axis=2)
                X_action = np.stack(action_samples, axis=0)
                X_static = np.stack(static_samples, axis=0)
                y = np.stack(outcome_samples, axis=0)
                country_samples[g] = {
                    'X_context': X_context,
                    'X_action': X_action,
                    'X_static': X_static,
                    'y': y,
                    'X_train_context': X_context[:-NB_TEST_DAYS],
                    'X_train_action': X_action[:-NB_TEST_DAYS],
                    'X_train_static': X_static[:-NB_TEST_DAYS],
                    'y_train': y[:-NB_TEST_DAYS],
                    'X_test_context': X_context[-NB_TEST_DAYS:],
                    'X_test_action': X_action[-NB_TEST_DAYS:],
                    'X_test_static': X_static[-NB_TEST_DAYS:],
                    'y_test': y[-NB_TEST_DAYS:],
                }
        return country_samples

    # Function for performing roll outs into the future
    @staticmethod
    def _roll_out_predictions(predictor, initial_context_input, initial_action_input, static_input, future_action_sequence):
        nb_roll_out_days = future_action_sequence.shape[0]
        pred_output = np.zeros(nb_roll_out_days)
        context_input = np.expand_dims(np.copy(initial_context_input), axis=0)
        action_input = np.expand_dims(np.copy(initial_action_input), axis=0)
        static_input = np.expand_dims(np.copy(static_input), axis=0)
        for d in range(nb_roll_out_days):
            action_input[:, :-1] = action_input[:, 1:]
            # Use the passed actions
            action_sequence = future_action_sequence[d]
            action_input[:, -1] = action_sequence
            pred = predictor.predict(torch.Tensor(context_input, device=predictor.device),
                                     torch.Tensor(action_input, device=predictor.device),
                                     torch.Tensor(static_input, device=predictor.device))
            pred_output[d] = pred.detach().numpy()
            context_input[:, :-1] = context_input[:, 1:]
            context_input[:, -1] = pred.detach().numpy()
        return pred_output

    # Functions for converting predictions back to number of cases
    @staticmethod
    def _convert_ratio_to_new_cases(ratio,
                                    window_size,
                                    prev_new_cases_list,
                                    prev_pct_infected):
        return (ratio * (1 - prev_pct_infected) - 1) * \
               (window_size * np.mean(prev_new_cases_list[-window_size:])) \
               + prev_new_cases_list[-window_size]

    def _convert_ratios_to_total_cases(self,
                                       ratios,
                                       window_size,
                                       prev_new_cases,
                                       initial_total_cases,
                                       pop_size):
        new_new_cases = []
        prev_new_cases_list = list(prev_new_cases)
        curr_total_cases = initial_total_cases
        for ratio in ratios:
            new_cases = self._convert_ratio_to_new_cases(ratio,
                                                         window_size,
                                                         prev_new_cases_list,
                                                         curr_total_cases / pop_size)
            # new_cases can't be negative!
            new_cases = max(0, new_cases)
            # Which means total cases can't go down
            curr_total_cases += new_cases
            # Update prev_new_cases_list for next iteration of the loop
            prev_new_cases_list.append(new_cases)
            new_new_cases.append(new_cases)
        return new_new_cases

    @staticmethod
    def _smooth_case_list(case_list, window):
        return pd.Series(case_list).rolling(window).mean().to_numpy()

    def one_hot_encoding_action_list(self, x_action):
        x_action = x_action.astype(int)
        one_hot_x_actions = []
        max_vals = [3, 3, 2, 4, 2, 3, 2, 4, 2, 3, 2, 4]
        num_data, num_lookback_days, num_actions = x_action.shape

        for i in range(num_data):
            x_action_per_row = []
            for j in range(num_lookback_days):
                action_per_day = []
                for k in range(num_actions):
                    cur_action = x_action[i][j][k]
                    one_hot_action = [0]*(max_vals[k]+1)
                    one_hot_action[cur_action] = 1
                    action_per_day.extend(one_hot_action)
                x_action_per_row.append(action_per_day)
            one_hot_x_actions.append(x_action_per_row)

        x_action = np.array(one_hot_x_actions)
        x_action = x_action.astype(float)
        return x_action

    def train(self):
        print("Creating numpy arrays for Keras for each country...")
        geos = self._most_affected_geos(self.df, MAX_NB_COUNTRIES, NB_LOOKBACK_DAYS)
        country_samples = self._create_country_samples(self.df, geos, new_cols=self.new_columns)
        print("Numpy arrays created")

        # Aggregate data for training
        all_X_context_list = [country_samples[c]['X_train_context']
                              for c in country_samples]
        all_X_action_list = [country_samples[c]['X_train_action']
                             for c in country_samples]
        all_X_static_list = [country_samples[c]['X_train_static']
                             for c in country_samples]
        all_y_list = [country_samples[c]['y_train']
                      for c in country_samples]
        X_context = np.concatenate(all_X_context_list)
        X_action = np.concatenate(all_X_action_list)
        X_static = np.concatenate(all_X_static_list)
        y = np.concatenate(all_y_list)

        # One hot encoding action list
        X_action = self.one_hot_encoding_action_list(X_action)

        print(f'X context shape {X_context.shape}')
        print(f'X action shape {X_action.shape}')
        # Clip outliers
        MIN_VALUE = 0.
        MAX_VALUE = 2.
        X_context = np.clip(X_context, MIN_VALUE, MAX_VALUE)
        y = np.clip(y, MIN_VALUE, MAX_VALUE)
        nb_context = X_context.shape[-1]
        nb_action = X_action.shape[-1]

        # Convert data in numpy arrays to tensors
        X_context = torch.Tensor(X_context)
        X_action = torch.Tensor(X_action)
        X_static = torch.Tensor(X_static)
        y = torch.Tensor(y)

        # Aggregate data for testing only on top countries
        test_all_X_context_list = [country_samples[g]['X_test_context']
                                   for g in geos]
        test_all_X_action_list = [country_samples[g]['X_test_action']
                                  for g in geos]
        test_all_X_static_list = [country_samples[g]['X_test_static']
                                  for g in geos]
        test_all_y_list = [country_samples[g]['y_test']
                           for g in geos]
        test_X_context = np.concatenate(test_all_X_context_list)
        test_X_action = np.concatenate(test_all_X_action_list)
        test_X_static = np.concatenate(test_all_X_static_list)
        test_y = np.concatenate(test_all_y_list)

        test_X_context = np.clip(test_X_context, MIN_VALUE, MAX_VALUE)
        test_y = np.clip(test_y, MIN_VALUE, MAX_VALUE)

        test_X_context = torch.Tensor(test_X_context)
        test_X_action = torch.Tensor(test_X_action)
        test_X_static = torch.Tensor(test_X_static)
        test_y = torch.Tensor(test_y)

        # Run full training several times to find best model
        # and gather data for setting acceptance threshold
        models = []
        for t in range(NUM_TRIALS):
            print('Trial', t)
            train_data_size = int(len(y) * 0.9)
            val_data_size = len(y) - train_data_size
            all_dataset = TensorDataset(X_context, X_action, X_static, y)
            train_dataset, val_dataset = torch.utils.data.random_split(all_dataset,
                                                                       [train_data_size, val_data_size],
                                                                       generator=torch.Generator().manual_seed(42))
            test_dataset = TensorDataset(test_X_context, test_X_action, test_X_static, test_y)

            train_dataloader = DataLoader(train_dataset, num_workers=4,
                                          batch_size=BATCH_SIZE)
            val_dataloader = DataLoader(val_dataset, num_workers=4,
                                        batch_size=BATCH_SIZE)
            test_dataloader = DataLoader(test_dataset, num_workers=4,
                                         batch_size=BATCH_SIZE)
            model = Simple_RNN(nb_context=nb_context, nb_action=nb_action, nb_static=len(self.new_columns),
                               nb_lookback=NB_LOOKBACK_DAYS,
                               train_data_size=train_data_size,
                               val_data_size=val_data_size)
            early_stop_callback = EarlyStopping(
                monitor='validation_loss',
                patience=PATIENCE,
                strict=False,
                verbose=False,
                mode='min'
            )

            checkpoint_callback = ModelCheckpoint(
                monitor='validation_loss',
                dirpath='save/checkpoints_callback/',
                filename='rnn_model-{epoch:02d}-{validation_loss:.2f}',
                save_top_k=3,
                mode='min',
            )


            trainer = Trainer(max_epochs=NUM_EPOCHS,
                              gpus=1 if USE_GPU else 0,
                              deterministic=True,
                              progress_bar_refresh_rate=1,
                              logger = None,
                              #default_root_dir='save/checkpoints',
                              #callbacks=[checkpoint_callback, early_stop_callback]
                              )

            # Start training
            trainer.fit(model, train_dataloader, val_dataloader)
            #test_result = trainer.test(test_dataloaders=test_dataloader)
            #print(test_result)
            #model = model.load_from_checkpoint(checkpoint_callback.best_model_path,
            #                           nb_context=nb_context,
            #                           nb_action=nb_action,
            #                           nb_static=len(self.new_columns),
            #                           nb_lookback=NB_LOOKBACK_DAYS, map_location=model.device)
            #print(f'Best model path: {checkpoint_callback.best_model_path}')

            models.append(model)
        return

        # Gather test info
        #country_indeps = []
        #country_predss = []
        #country_casess = []
        #for model in models:
        #    country_indep, country_preds, country_cases = self._lstm_get_test_rollouts(model,
        #                                                                               self.df,
        #                                                                               geos,
        #                                                                               country_samples)
        #    country_indeps.append(country_indep)
        #    country_predss.append(country_preds)
        #    country_casess.append(country_cases)

        # Compute cases mae
        #test_case_maes = []
        #for m in range(len(models)):
        #    total_loss = 0
        #    for g in geos:
        #        true_cases = np.sum(np.array(self.df[self.df.GeoID == g].NewCases)[-NB_TEST_DAYS:])
        #        pred_cases = np.sum(country_casess[m][g][-NB_TEST_DAYS:])
        #        total_loss += np.abs(true_cases - pred_cases)
        #    test_case_maes.append(total_loss)

        # Select best model
        #best_model = models[np.argmin(test_case_maes)]
        #self.predictor = best_model
        #print("Done")
        #return best_model

    @staticmethod
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

    def _save_predictor(self, to_save_path):
        torch.save(self.predictor.state_dict(), to_save_path)

    # Functions for computing test metrics
    def _lstm_roll_out_predictions(self, model, initial_context_input, initial_action_input, static_input, future_action_sequence):
        nb_test_days = future_action_sequence.shape[0]
        pred_output = np.zeros(nb_test_days)
        context_input = np.expand_dims(np.copy(initial_context_input), axis=0)
        action_input = np.expand_dims(np.copy(initial_action_input), axis=0)
        static_input = np.expand_dims(np.copy(static_input), axis=0)
        
        for d in range(nb_test_days):
            action_input[:, :-1] = action_input[:, 1:]
            action_input[:, -1] = future_action_sequence[d]
            pred = model.predict(torch.Tensor(context_input, device=model.device),
                                 torch.Tensor(action_input, device=model.device),
                                 torch.Tensor(static_input, device=model.device))
            pred_output[d] = pred.detach().numpy()
            context_input[:, :-1] = context_input[:, 1:]
            context_input[:, -1] = pred.detach().numpy()
        return pred_output

    def _lstm_get_test_rollouts(self, model, df, top_geos, country_samples):
        country_indep = {}
        country_preds = {}
        country_cases = {}
        for g in top_geos:
            X_test_context = country_samples[g]['X_test_context']
            X_test_action = country_samples[g]['X_test_action']
            X_test_static = country_samples[g]['X_test_static']
            
            test_X_context = torch.Tensor(X_test_context, device=model.device)
            test_X_action = torch.Tensor(X_test_action, device=model.device)
            test_X_static = torch.Tensor(X_test_static, device=model.device)
            
            temp = model.predict(test_X_context, test_X_action, test_X_static).detach().numpy()
            country_indep[g] = temp
            initial_context_input = country_samples[g]['X_test_context'][0]
            initial_action_input = country_samples[g]['X_test_action'][0]
            static_input = country_samples[g]['X_test_static'][0]
            
            y_test = country_samples[g]['y_test']

            nb_test_days = y_test.shape[0]
            nb_actions = initial_action_input.shape[-1]

            future_action_sequence = np.zeros((nb_test_days, nb_actions))
            future_action_sequence[:nb_test_days] = country_samples[g]['X_test_action'][:, -1, :]
            current_action = country_samples[g]['X_test_action'][:, -1, :][-1]
            future_action_sequence[14:] = current_action
            preds = self._lstm_roll_out_predictions(model,
                                                    initial_context_input,
                                                    initial_action_input,
                                                    static_input, #not that confused
                                                    future_action_sequence)
            country_preds[g] = preds

            prev_confirmed_cases = np.array(
                df[df.GeoID == g].ConfirmedCases)[:-nb_test_days]
            prev_new_cases = np.array(
                df[df.GeoID == g].NewCases)[:-nb_test_days]
            initial_total_cases = prev_confirmed_cases[-1]
            pop_size = np.array(df[df.GeoID == g].Population)[0]

            pred_new_cases = self._convert_ratios_to_total_cases(
                preds, WINDOW_SIZE, prev_new_cases, initial_total_cases, pop_size)
            country_cases[g] = pred_new_cases

        return country_indep, country_preds, country_cases
