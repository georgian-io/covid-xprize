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
    lookback: how many days of IP history to preserve (if None, everything is used)
    freeze: number of days for which to freeze a policy before allowing the agent to change
    
    NB: currently, this framework can only handle training on a particular (CountryName, NaN)
    TODO: does it matter whether we start generating policies immediately after the IP history ends? I don't think
          so, since we're considering the long-run optimal policy, but it's worth checking...
    TODO: run timing test to determine which part of the code is slow
    """

    def __init__(self, country, IP_history_file, lookback=180, freeze=7, action_weight=0.5):
        super(CovidEnv, self).__init__()
        self.is_async = True
        # When actions are sampled, one value from each dimension will be selected
        # For quick tests, use self.action_space = spaces.Box(low=np.zeros(12), high=np.ones(12), dtype=np.int32)
        self.action_space = spaces.Box(low=np.array([0] * len(IP_MAX_VALUES.values())),
                                       high=np.array(list(IP_MAX_VALUES.values())), 
                                       dtype=np.int32)
        
        # Placeholder: this variable is used to map scalar number to which sets of actions it refers to
        self._action_map = None

        # Observations are number of current DailyNewCases (the predictor outputs a float)--although the dict
        # structure exists, it is not compatible with Tianshou, so we will make the first indes of the
        # observations correspond to cases and the remaining indices correspond to costs
        self.observation_space = spaces.Box(low=np.array([0]), # * (len(IP_MAX_VALUES.values()) + 1)),
                                            high=np.array([np.inf]), # + list(IP_MAX_VALUES.values())),
                                            dtype=np.float64)

        # Lookback tells us how much data we should call on predict(...) at any time
        if lookback is None:
            self.lookback = 0
        else:
            self.lookback = -lookback
        self.freeze = freeze

        # Load in the IP history from the file 
        self.IP_history_file = IP_history_file
        self.IP_history = pd.read_csv(self.IP_history_file,
                                      parse_dates=['Date'],
                                      encoding="ISO-8859-1",
                                      dtype={"RegionName": str},
                                      error_bad_lines=True)
        self.IP_history = self.IP_history.iloc[self.lookback:].reset_index()

        # Keep track of first date and initial history for env.reset()
        self.first_date = pd.to_datetime(self.IP_history['Date'].max())

        # Date counter and action counter variables
        self.date = self.first_date
        self.action = np.zeros(self.action_space.shape[0])
        self.action_weight = action_weight

        # Counter for the number of rounds we've done
        self.round_counter = 0

        # Quick assert
        countries = self.IP_history["CountryName"].to_numpy()
        assert((country == countries).all())


    def step(self, action):
        import timeit; start = timeit.timeit()
        # If action is a scalar, need to map back to original action space, e.g. in DQN
        if np.isscalar(action):
            action = self._map_scalar_action(action)
        
        # Check that action is an integer and is within the IP_MAX_VALUES. Otherwise, replace with max e.g. in SAC
        action = np.round(action)
        for i in range(len(list(action))):
            if i > self.action_space.high[i]:
                action[i] = self.action_space.high[i]
            if i < self.action_space.low[i]:
                action[i] = self.action_space.low[i]

        # Execute one time step within the environment
        self._take_action(action)

        # Arbitrarily normalized reward function!
        reward = self._reward(action)

        # Decide when to end the simulation--always end after 180 days?
        self.round_counter += 1
        if self.round_counter * self.freeze >= 180:
            done = True
        else:
            done = False
        #print(reward)
        end = timeit.timeit()
        #print("Step function", end - start
        # Update the state to be the new number of cases        
        return self.state, reward, done, {}


    def _map_scalar_action(self, action):
        # Map back scalar action to what actual IPs it refers to
        # print(action) # for sanity check that the action map is not wrong
        if self._action_map is None:
            ip_levels = []
            for ip in np.array(list(IP_MAX_VALUES.values())):
                ip_levels.append(np.arange(ip + 1))
            self._action_map = np.array(np.meshgrid(*ip_levels)).T.reshape(-1, len(IP_MAX_VALUES))
        return self._action_map[action]


    def _reward(self, action):
        # Normalize the costs and compute a reward
        #normed_costs = self.state[1:] / sum(self.state[1:])
        action_reward = self.action_weight * np.sum(action) / 12. #np.dot(action, normed_costs) / 12.
        state_reward = (1 - self.action_weight) * self.state[0] / self.initial_state
        return -1. * (action_reward + state_reward)


    def _take_action(self, action):
        import timeit; start = timeit.timeit()
        self.action = action

        # Append the new prescriptions to the IP_history
        prescription_df = pd.DataFrame({"CountryName": self.IP_history.loc[:self.freeze - 1, "CountryName"],
                                        "RegionName": self.IP_history.loc[:self.freeze - 1, "RegionName"],
                                        "Date": pd.date_range(start=self.date, end=self.date + pd.DateOffset(days=self.freeze - 1))})
        #print(action)
        for i, ip in enumerate(IPS):
            prescription_df[ip] = [action[i]] * self.freeze

        self.IP_history = self.IP_history.append(prescription_df, ignore_index=True)

        # Increment the date and write out some debugging info
        self.date += pd.DateOffset(days=self.freeze)

        # Remove the oldest rows of the dataframe (in accordance with lookback parameter)
        if self.lookback != 0:
            self.IP_history = self.IP_history.iloc[self.freeze:].reset_index(drop=True)

        # Write out file (for the predictor... sigh)
        self.IP_history.to_csv("prescriptions_temp.csv", index=False)

        # The predictor gets us to the next state
        predict(self.date, self.date + pd.DateOffset(days=self.freeze - 1), "prescriptions_temp.csv", "predictions/preds_temp.csv")
        df = pd.read_csv("predictions/preds_temp.csv",
                         parse_dates=['Date'],
                         encoding="ISO-8859-1",
                         error_bad_lines=True)

        # Update the state variable 
        self.state[0] = np.sum(df['PredictedDailyNewCases'])
        end = timeit.timeit()
        #print("_take_action", end - start)

    def reset(self):
        # Define state to be a dictionary containing cases and costs
        #self.state = np.zeros(1 + len(IP_MAX_VALUES))
        self.state = np.zeros(1)

        # Reset the date + counters
        self.date = self.first_date
        self.round_counter = 0
        self.action = np.zeros(self.action_space.shape[0])

        # Reset the IP history
        self.IP_history = pd.read_csv(self.IP_history_file,
                                      parse_dates=['Date'],
                                      encoding="ISO-8859-1",
                                      dtype={"RegionName": str},
                                      error_bad_lines=True)
        self.IP_history = self.IP_history.iloc[self.lookback:].reset_index()

        # Reset: get the state (number of cases) from the predict function
        predict(self.first_date, self.first_date, self.IP_history_file, "predictions/preds_temp.csv")
        pred_df = pd.read_csv("predictions/preds_temp.csv",
                              parse_dates=['Date'],
                              encoding="ISO-8859-1",
                              error_bad_lines=True)
        self.state[0] = pred_df.loc[0, "PredictedDailyNewCases"]

        # Keep track of the initial number of cases
        self.initial_state = self.state[0]

        # Randomly generate some costs
        #self.state[1:] = np.random.rand(len(IP_MAX_VALUES))

        # Return the state
        return self.state
        

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print("Daily new cases", round(self.state[0]), "\t\tCurrent action sum", sum(self.action))

