import gym
from gym import spaces
from covid_xprize.standard_predictor.predict import predict
import pandas as pd


class CovidEnv(gym.Env):
  """Custom Environment that follows gym interface"""

    def __init__(self, IP_history, start_prescription_date, costs_file, pred_cases_file="prescriptions/preds.csv"):
        super(CovidEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

        self.observation_space = {
            'current_case': None,
            'IP_history': IP_history
        }

        self.date = pd.to_datetime(start_prescription_date)
        self.pred_cases_file = pred_cases_file
        self.costs_file = costs_file


    def step(self, action):
        # Execute one time step within the environment
        next_state = self._take_action(action)


    def _take_action(self, prescription_df):
        # Convert the actions into their expanded form--this is now indexed by CountryName, RegionName, and IPS
        prescription_df = expand_IP(prescription_df)

        # Update the IP history for new predictions--add a date and tack on to the end of the IP_history
        prescription_df["Date"] = self.date
        prescription_df = prescription_df[["CountryName", "RegionName", "Date"] + IPS]
        self.IP_history = pd.concat([self.IP_history, prescription_df])

        # the predictor gets us to the next state
        self.date += pd.DateOffset(days=1)
        predict(self.date, self.date, self.observation_space, self.pred_cases_file)
        df = pd.read_csv(self.pred_cases_file,
                         parse_dates=['Date'],
                         encoding="ISO-8859-1",
                         error_bad_lines=True)
        # Get reward
        df['Reward'] = 1 / (df["Stringency"] + df["PredictedDailyNewCases"])  # TODO placeholder reward function!!!
        next_state = df[['CountryName', 'RegionName', 'PredictedDailyNewCases']]
        return next_state

  def reset(self):
        # Reset the state of the environment to an initial state

    def render(self, mode='human', close=False):
        # Render the environment to the screen

