import gym
import numpy as np
import pandas as pd
from gym import spaces
from covid_xprize.standard_predictor.predict import predict
from covid_xprize.examples.prescriptors.rl.utilities import IP_MAX_VALUES, IPS


class CovidEnv(gym.Env):
    """
    Custom Environment that follows gym interface
    IP_history: path to dataframe consisting of the IP history for the relevant (CountryName, RegionName)
    costs_file: (not currently used) cost of interventions for the relevant (CountryName, RegionName)
    
    NB: currently, this framework can only handle training on a particular (CountryName, RegionName)
    NB: currently, we assume that we start generating policies for the day the IP history ends
    NB: currently, we only generate policies of 1 day before we allow the agent to change action -- this number
            should be optimized later!
    NB: may want to add a lookback parameter to prevent predict(...) from running on all historical data
    """

    def __init__(self, IP_history_file, costs_file):
        super(CovidEnv, self).__init__()
        # When actions are sampled, one value from each dimension will be selected
        self.action_space = spaces.Box(low=np.zeros(12), high=np.ones(12), dtype=np.int32)
        # self.action_space = spaces.Box(low=np.array([0] * len(IP_MAX_VALUES)),
        #                                high=np.array(IP_MAX_VALUES.values()), 
        #                                dtype=np.int32)

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

        # Date counter variable
        self.date = self.first_date


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        # TODO: fix the normalization later
        reward = - (sum(action) + self.state / 500.) 

        # TODO:considering ending the simulation if it gets "too bad" and try to reset
        done = False

        # Update the state to be the new number of cases        
        return self.state, reward, done, {}


    def _take_action(self, action):
        # TODO: consider incrementing the date by more than 1 and applying IPs for more time
        self.date += pd.DateOffset(days=1)

        # Convert the actions into their expanded form--first, add CountryName and RegionName
        prescription_df = pd.DataFrame([[self.IP_history.loc[0, "CountryName"], self.IP_history.loc[0, "RegionName"], self.date] + list(action)], 
                                        columns=["CountryName", "RegionName", "Date"] + IPS)

        # Update the IP_history by appending on the new prescription
        self.IP_history = pd.concat([self.IP_history, prescription_df])

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
        

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print("Daily new cases", self.state)

