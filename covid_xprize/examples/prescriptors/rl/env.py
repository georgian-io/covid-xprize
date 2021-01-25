import gym
from gym import spaces
from covid_xprize.standard_predictor.predict import predict
import pandas as pd
from covid_xprize.examples.prescriptors.rl.constants import expand_IP


class CovidEnv(gym.Env):
  """Custom Environment that follows gym interface"""

    def __init__(self, IP_history, start_prescription_date, costs_file, pred_cases_file="prescriptions/preds.csv"):
        super(CovidEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # TODO: add the actions here
        # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # self.action_space = spaces.Box(low=np.array([0]*12), high=np.array([3, 3, 2, 4, 2, 3, 2, 4, 2, 3, 2, 4]), dtype=np.int32)

        self.action_space = spaces.Box(low=np.zeros(12), high=np.ones(12), dtype=np.int32)


        self.observation_space = {
            'current_cases': None,
            'IP_history': IP_history
        }

        self.date = pd.to_datetime(start_prescription_date)
        self.pred_cases_file = pred_cases_file


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

        prescription_df = pd.DataFrame({'CountryName': "Canada", 'RegionName': "NaN"]})
        
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
        predict(self.date, self.date, self.observation_space, "prescriptions/preds.csv")
        df = pd.read_csv("prescriptions/preds.csv",
                         parse_dates=['Date'],
                         encoding="ISO-8859-1",
                         error_bad_lines=True)
       
        self.observation_space['current_case'] = df[['CountryName', 'RegionName', 'PredictedDailyNewCases']]

  def reset(self):
        # Reset the state of the environment to an initial state

    def render(self, mode='human', close=False):
        # Render the environment to the screen

# TODO: to test it, we can try it with random actions 
