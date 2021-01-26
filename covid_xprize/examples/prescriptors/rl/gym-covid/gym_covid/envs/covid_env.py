import gym
import utilities
import pandas as pd
from gym import spaces
from covid_xprize.standard_predictor.predict import predict


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
        # self.action_space = spaces.Box(low=np.array([0] * len(utilities.IP_MAX_VALUES)),
        #                                high=np.array(utilities.IP_MAX_VALUES.values()), 
        #                                dtype=np.int32)

        # Observations are number of current DailyNewCases (the predictor outputs a float)
        self.observation_space = spaces.Box(low=0, high=np.inf, dtype=np.float32)

        # Load in the IP history from the file 
        self.IP_history = pd.read_csv(IP_history_file,
                                      parse_dates=['Date'],
                                      encoding="ISO-8859-1",
                                      dtype={"RegionName": str},
                                      error_bad_lines=True)

        # Keep track of first date and initial history for env.reset()
        self.first_date = pd.to_datetime(IP_history['Date'].max())
        self.IP_history_file = IP_history_file

        # Number of days of history to use in predict (can be adjusted!)
        self.lookback = 30


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        # TODO: fix the normalization later
        reward = - (sum(action) + self.observation_space['current_cases'] / 500.) 

        # TODO:considering ending the simulation if it gets "too bad" and try to reset
        done = False

        # Update the state to be the new number of cases        
        return self.state, reward, done, {}


    def _take_action(self, action):
        # Convert the actions into their expanded form--first, add CountryName and RegionName
        prescription_df = pd.DataFrame({'CountryName': "Canada", 'RegionName': "British Columbia"})
        
        # Add the IPs based on the actions taken--use the utility function to convert integer actions into
        # values the dataframe can understand!
        For i, ip in enumerate(utilities.IPS):
            prescription_df[ip] = action[i]

        # Update the IP history for new predictions--add a date and re-order the columns
        prescription_df["Date"] = self.date
        prescription_df = prescription_df[["CountryName", "RegionName", "Date"] + IPS]

        # Update the IP_history by appending on the new prescription
        self.IP_history = pd.concat([self.IP_history['IP_history'], prescription_df])

        # TODO: consider incrementing the date by more than 1 and applying IPs for more time
        self.date += pd.DateOffset(days=1)

        # The predictor gets us to the next state
        predict(self.date, self.date, self.observation_space, "preds.csv")
        df = pd.read_csv("preds.csv",
                         parse_dates=['Date'],
                         encoding="ISO-8859-1",
                         error_bad_lines=True)

        # Update the state variable
        self.state = df['PredictedDailyNewCases'][0]


    def reset(self):
        # Reset: get the state (number of cases) from the predict function
        hist = pd.read_csv(self.IP_history_file,
                           parse_dates=['Date'],
                           encoding="ISO-8859-1",
                           dtype={"RegionName": str},
                           error_bad_lines=True)
        predict(self.first_date, self.first_date, hist, "preds.csv")
        pred_df = pd.read_csv("prescriptions/preds.csv",
                              parse_dates=['Date'],
                              encoding="ISO-8859-1",
                              error_bad_lines=True)
        self.state = pred_df["PredictedDailyNewCases"][0]
        

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print("Daily new cases", self.observation_space)

