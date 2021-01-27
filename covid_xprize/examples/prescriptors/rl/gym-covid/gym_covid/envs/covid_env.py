import gym
import numpy as np
import pandas as pd
from gym import spaces
from covid_xprize.standard_predictor.predict import predict
from covid_xprize.examples.prescriptors.rl.utilities import IP_MAX_VALUES, IPS


class CovidEnv(gym.Env):
    """
    Custom Environment that follows gym interface
    country: the CovidEnv can only make policies for a single region at a time--specify the desired country
    IP_history: path to dataframe consisting of the IP history for the relevant (CountryName, RegionName)
    costs_file: (not currently used) cost of interventions for the relevant (CountryName, RegionName)
    
    NB: currently, this framework can only handle training on a particular (CountryName, NaN)
    TODO: does it matter whether we start generating policies immediately after the IP history ends? I don't think
          so, since we're considering the long-run optimal policy, but it's worth checking...
    TODO: allow policies to be "frozen" for several days so we make fewer calls to predict(...)
    TODO: add a "lookback" parameter to prevent predict(...) from running on all historical data
    TODO: run timing test to determine which part of the code is slow
    TODO:considering ending the simulation if it gets "too bad" and try to reset
    TODO: fix the normalization of the reward function later
    """

    def __init__(self, country, IP_history_file, costs_file):
        super(CovidEnv, self).__init__()
        # When actions are sampled, one value from each dimension will be selected
        # For quick tests, use self.action_space = spaces.Box(low=np.zeros(12), high=np.ones(12), dtype=np.int32)
        self.action_space = spaces.Box(low=np.array([0] * len(IP_MAX_VALUES.values())),
                                       high=np.array(list(IP_MAX_VALUES.values())), 
                                       dtype=np.int32)

        # Observations are number of current DailyNewCases (the predictor outputs a float)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1, 1), dtype=np.float32)

        # Load in the IP history from the file 
        self.IP_history = pd.read_csv(IP_history_file,
                                      parse_dates=['Date'],
                                      encoding="ISO-8859-1",
                                      dtype={"RegionName": str},
                                      error_bad_lines=True)

        # Keep track of first date and initial history for env.reset()
        self.first_date = pd.to_datetime(self.IP_history['Date'].max())
        self.IP_history_file = IP_history_file

        # Date counter and action counter variables
        self.date = self.first_date
        self.action = np.zeros(self.action_space.shape[0])

        # Quick assert
        countries = self.IP_history["CountryName"].to_numpy()
        assert((country == countries).all())


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        # Arbitrarily normalized reward function!
        reward = - (sum(action) + self.state / 500.) 

        # Decide when to end the simulation
        done = False

        # Update the state to be the new number of cases        
        return self.state, reward, done, {}


    def _take_action(self, action):
        self.action = action
        self.date += pd.DateOffset(days=1)
        print("Actions for " + str(self.date) + ": " + str(action))

        # Convert the actions into their expanded form--first, add CountryName and RegionName
        prescription_df = pd.DataFrame([[self.IP_history.loc[0, "CountryName"], self.IP_history.loc[0, "RegionName"], self.date] + \
                                            list(action)], columns=["CountryName", "RegionName", "Date"] + IPS)

        # Update the IP_history by appending on the new prescription
        self.IP_history = self.IP_history.append(prescription_df, ignore_index=True)

        # Write out file (for the predictor... sigh)
        self.IP_history.to_csv("prescriptions.csv", index=False)

        # The predictor gets us to the next state
        predict(self.date, self.date, "prescriptions.csv", "predictions/preds.csv")
        df = pd.read_csv("predictions/preds.csv",
                         parse_dates=['Date'],
                         encoding="ISO-8859-1",
                         error_bad_lines=True)

        # Update the state variable 
        self.state = df.loc[0, 'PredictedDailyNewCases']


    def reset(self):
        # Reset: get the state (number of cases) from the predict function
        predict(self.first_date, self.first_date, self.IP_history_file, "predictions/preds.csv")
        pred_df = pd.read_csv("predictions/preds.csv",
                              parse_dates=['Date'],
                              encoding="ISO-8859-1",
                              error_bad_lines=True)
        self.state = pred_df.loc[0, "PredictedDailyNewCases"]
        return self.state
        

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print("Daily new cases", self.state, "\t\tCurrent action sum", sum(self.action))

