import gym
import pandas as pd
from gym import spaces
from utilities import expand_IP
from covid_xprize.standard_predictor.predict import predict


class CovidEnv(gym.Env):
    """
    Custom Environment that follows gym interface
    IP_history: path to dataframe consisting of the IP history for the relevant (CountryName, RegionName)
    costs_file: (not currently used) cost of interventions for the relevant (CountryName, RegionName)
    
    NB: currently, this framework can only handle training on a particular (CountryName, RegionName)
    NB: currently, we assume that we start generating policies for the day the IP history ends
    NB: currently, we freeze all policies for 5 days before we allow the agent to change them -- this number
            should be optimized later!
    """

    def __init__(self, IP_history_file, costs_file):
        super(CovidEnv, self).__init__()
        # The action_space must be a gym.spaces object
        self.action_space = spaces.Box(low=np.zeros(12), high=np.ones(12), dtype=np.int32)
        # self.action_space = spaces.Box(low=np.array([0] * len(utilities.IP_MAX_VALUES)),
        #                                high=np.array(utilities.IP_MAX_VALUES.values()), 
        #                                dtype=np.int32)

        # The observation_space must also inherit from gym.spaces
        # self.observation_space will hold the number of current DailyNewCases
        self.observation_space = spaces.Box(low=0, high=np.inf, dtype=np.float32)

        # Load in the IP history from the file 
        self.IP_history = pd.read_csv(IP_history_file,
                                      parse_dates=['Date'],
                                      encoding="ISO-8859-1",
                                      dtype={"RegionName": str},
                                      error_bad_lines=True)
        recent_date = IP_history['Date'].max()

        # Let the start date be the last date in the IP history file for now
        self.date = pd.to_datetime(self.IP_history[self.IP_history["Date"] == recent_date][["CountryName", "RegionName"]])


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        # TODO: update the reward function later!!! 
        reward = - (sum(action) + self.observation_space['current_cases'] * 1/500) 
        # done = self.net_worth <= 0
        done = False
        obs = self.observation_space
        
        return obs, reward, done, {}


    def _take_action(self, actions):
        # Convert the actions into their expanded form--this is now indexed by CountryName, RegionName, and IPS

        prescription_df = pd.DataFrame({'CountryName': "Canada", 'RegionName': "British Columbia"})
        
        For id, ip in enumerate(IPS):
            prescription_df[ip] = action[id]
            action[i] for id, ip in enumerate(IPS)
        prescription_df = expand_IP(prescription_df)

        # Update the IP history for new predictions--add a date and tack on to the end of the IP_history
        prescription_df["Date"] = self.date
        prescription_df = prescription_df[["CountryName", "RegionName", "Date"] + IPS]
        self.observation_space['IP_history'] = pd.concat([self.observation_space['IP_history'], prescription_df])

        # the predictor gets us to the next state
        self.date += pd.DateOffset(days=1)
        # the last argument to predict is just needed for outputting predicted cases
        predict(self.date, self.date, self.observation_space, "preds.csv")
        df = pd.read_csv("preds.csv",
                         parse_dates=['Date'],
                         encoding="ISO-8859-1",
                         error_bad_lines=True)
       
        self.observation_space['current_case'] = df[['CountryName', 'RegionName', 'PredictedDailyNewCases']]

  def reset(self):
        # Reset the state of the environment to an initial state

    def render(self, mode='human', close=False):
        # Render the environment to the screen

# TODO: to test it, we can try it with random actions 
