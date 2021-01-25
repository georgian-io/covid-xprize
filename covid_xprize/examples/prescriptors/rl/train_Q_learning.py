import numpy as np
import pandas as pd
# from covid_xprize.scoring.predictor_scoring import load_dataset
# from covid_xprize.validation.scenario_generator import generate_scenario
from covid_xprize.standard_predictor.predict import predict

"""

RL using Q-learning! Inputs:
DESCRIBE INPUTS HERE


Outputs:
Array containing optimal policy (a vector of prescriptions)
for each state (geo_id * n_cases * current_IP)


The state of the system is represented by
    (Geo_ID, # of cases, current_IP)
i.e. The state only depends on the current IP and number of cases.
     This will likely change in future implementations.
From the state, the MDP then choses from a range of potential next 
IPs. The predictor is then called to determine the impact on the
number of cases and the stringency. This process is repeated until 
the Q-learning converges upon the optimal policy FOR A PARTICULAR 
REGION given a particular case count and IP. 


This model is extremely simplistic. Some other considerations
-> Can make the states more complicated (e.g. retain more IP history)
-> It might make sense to build in relationships beteween caseloads
       in all regions within a country (e.g. so increases in Quebec
       caseloads can influence caseloads in Canada)
-> Some experimentation with the cost function will be required 
       once I get a better idea of how we want to judge outcomes
-> This seems super inefficient, since we're duplicating work for 
       each country/region. Maybe it would be better to train
       everything together and then specialize later using RL?

"""

IPS = ['C1_School closing',
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

IPS_CONDENSED = ['C1,C2',
                 'C4',
                 'C3,C5,C7',
                 'C6',
                 'C8',
                 'H1',
                 'H2',
                 'H3',
                 'H6']

IP_MAX_VALUES = {
    'C1,C2': 2,  # (0, 1) -> 0, (2) -> 1, (3) -> 2
    'C4': 2,  # (0, 1) -> 0, (2, 3) -> 1, (4) -> 2
    'C3,C5,C7': 2,
    'C6': 2,  # (0, 1) -> 0, (2) -> 1, (3) -> 2
    'C8': 2,  # (0, 1) -> 0, (2, 3) -> 1, (4) -> 2
    'H1': 0,  # temporarily adjusted
    'H2': 1,  # (0, 1) -> 0, (2, 3) -> 1
    'H3': 1,  # (0, 1) -> 0, (2) -> 1
    'H6': 2  # (0, 1) -> 0, (2, 3) -> 1, (4) -> 2
}  # Groupings get us down to 2916 actions


def expand_IP(df):
    # Map values back--take the lower of the paranthetical numbers
    df.loc[df['C1,C2'] == 2, 'C1,C2'] = 3
    df.loc[df['C1,C2'] == 1, 'C1,C2'] = 2
    df.loc[df['C4'] == 2, 'C4'] = 4
    df.loc[df['C4'] == 1, 'C4'] = 2
    df.loc[df['6'] == 2, '6'] = 3
    df.loc[df['6'] == 1, '6'] = 2
    df.loc[df['C8'] == 2, 'C8'] = 4
    df.loc[df['C8'] == 1, 'C8'] = 2
    df.loc[df['H6'] == 2, 'H6'] = 4
    df.loc[df['H6'] == 1, 'H6'] = 2
    df.loc[df['H2'] == 1, 'H2'] = 2
    df.loc[df['H3'] == 1, 'H3'] = 2

    # Change column names
    df['C2'] = df['C1,C2']
    df['C5'] = df['C3,C5,C7']
    df['C7'] = df['C3,C5,C7']
    names = ['C1,C2', 'C2', 'C3,C5,C7', 'C4', 'C5', 'C6', 'C7', 'C8', 'H1', 'H2', 'H3', 'H6']
    names = {names[i]: IPS[i] for i in range(len(names))}
    return df.rename(columns=names)


actions = np.zeros([IP_MAX_VALUES[i] + 1 for i in IP_MAX_VALUES])


class Q_learning:
    def __init__(self, states, costs_file, IP_history, date, gamma=0.9, alpha=1.0, epochs=100000):
        # States is a df with columns geo_id, cases (and eventually recent IPs)
        self.state = states
        self.IP_history = IP_history
        self.gamma = gamma
        self.alpha = alpha
        self.epochs = epochs
        self.costs_file = costs_file
        self.date = pd.to_datetime(date)

        # q_table: for each state, give the value of each action
        self.q_table = pd.DataFrame({'CountryName': states['CountryName'],
                                     'RegionName': states['RegionName'],
                                     'PredictedDailyNewCases': states['PredictedDailyNewCases']})
        # Add a new column for values--this is quite inefficient -> TODO find a better way of representing actions
        self.q_table["action_vals"] = [actions for i in range(len(self.q_table))]
        # Expect to see each state label and an array for each

        # Make a prescription df--will be updated regulalry
        self.prescription_df = pd.DataFrame({'CountryName': states['CountryName'],
                                             'RegionName': states['RegionName']})

        # For metrics--take the sum for total reward
        self.reward_df = pd.DataFrame({'CountryName': states['CountryName'],
                                       'RegionName': states['RegionName']})
        self.reward_df["Reward"] = 0
        self.total_reward_bc = 0  # just for fun

    def run_Q(self):
        # Note: because we expect the prescriptor to be run immediately-ish following the IP_history
        # we probably want to fix a horizon and have the simultion periodically reset instead of
        # considering the infinite horizon as we do below
        # self.state = self.rand_init()  # perhaps we don't want to entirely randomly initialize
        n = 0
        while n < self.epochs:
            n += 1

            v_and_a = self.get_action_eps()

            # the predictor gets us to the next state
            predict(self.date, self.date, self.IP_history, "prescriptions/preds.csv")
            df = pd.read_csv("prescriptions/preds.csv",
                             parse_dates=['Date'],
                             encoding="ISO-8859-1",
                             error_bad_lines=True)
            df['Reward'] = 1 / (df["Stringency"] + df["PredictedDailyNewCases"])  # TODO placeholder reward function!!!
            next_state = df[['CountryName', 'RegionName', 'PredictedDailyNewCases']]

            # compute new q-values--extract relevant qs then find max
            new_q = pd.merge(next_state, self.q_table, how='inner')
            neq_q["Values"] = new_q["action_vals"].apply(np.max)

            # update the Q table
            new_values = v_and_a['Values'] + self.alpha * (
                        neq_q['Reward'] - v_and_a['Values'] + self.gamma * new_q["Values"])
            for i in range(len(new_values)):
                self.current_q.loc[i, "action_vals"][v_and_a['Actions'][i]] = new_values[i]
                self.q_table.update(self.current_q)

            # Save the rewards
            self.reward_df = pd.concat([self.reward_df, df[['CountryName', 'RegionName', 'Reward']]])
            self.total_reward_bc = np.sum(
                df.loc[(df['CountryName'] == 'Canada') & (df['RegionName'] == 'BritishColumbia')]['Reward'])

            # Update the state
            self.state = next_state

            if i % 50 == 0:
                # Could break down cases and cost later
                print(f"Epoch: {n},\tReward: {self.total_reward_bc}")

    def rand_init(self):
        # Randomly initialize the state
        pass

    def get_action_eps(self):
        # Recall that we are running this "in parallel" for each geoid. That is, the current state is
        # the number of cases for each geo_id and then we need to pick IPs for each one separately
        self.date += pd.DateOffset(days=1)

        # First, extract the relevant parts of the q table depending on our current state
        self.current_q = pd.merge(self.state, self.q_table, how='inner')
        # current_q = self.q_table.loc[self.q_table["state"].isin(self.state)] # this selects the relevant rows

        # For each row in current_q, we want to extract the argmax at "action_vals"
        state_actions = self.current_q["action_vals"]
        state_actions["Actions"] = state_actions.apply(np.argmax)  # get the corresponding action
        state_actions["Values"] = state_actions.apply(np.max)  # get the value of max q

        expanded_actions = {npi: [] for npi in IP_MAX_VALUES}
        print(expanded_actions)
        selected_values = []
        for a in state_actions:
            for i, npi in enumerate(IP_MAX_VALUES):
                print(npi)
                expanded_actions[npi].append(np.argmax(a)[i])
                for npi in IP_MAX_VALUES:
                    self.prescription_df[npi] = expanded_actions[npi]

        # Use epsilon greedy to update some actions--generate len(self.prescription_df) * epsilon indices
        rand_ids = np.random.randint(0, len(len(self.prescription_df)), size=int(len(self.prescription_df) * epsilon))
        rands = np.empty((int(len(self.prescription_df) * epsilon), len(IP_MAX_VALUES)))
        for i, j in enumerate(IP_MAX_VALUES):
            rands[:, i] = np.random.randint(0, IP_MAX_VALUES[j] + 1, size=nrand)
        for i in rand_ids:
            for j, npi in enumerate(IP_MAX_VALUES):
                self.prescription_df.loc[i, npi] = rands[i, j]
            state_actions.loc[i, "Actions"] = rands[i]
            # This is terrible--I wonder if there's a better way...
            state_actions.loc[i, "Values"] = self.current_q.loc[i, "action_vals"][rands[i]]

        # Convert the actions into their expanded form--this is now indexed by CountryName, RegionName, and IPS
        self.prescription_df = expand_IP(self.prescription_df)

        # Update the IP history for new predictions--add a date and tack on to the end of the IP_history
        self.prescription_df["Date"] = self.date
        self.prescription_df = self.prescription_df[["CountryName", "RegionName", "Date"] + IPS]
        self.IP_history = pd.concat([self.IP_history, self.prescription_df])

        return state_actions


# LATEST_DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
# GEO_FILE = "/Users/mayaburhanpurkar/Documents/covid-xprize/countries_regions.csv"

# latest_df = load_dataset(LATEST_DATA_URL, GEO_FILE)
# TESTING: Limit to Canada
START_DATE = "2021-01-01"
END_DATE = "2021-01-03"
IP_FILE = "prescriptions/robojudge_test_scenario_historical.csv"
# scenario_df = generate_scenario(START_DATE, END_DATE, latest_df, countries=["Canada"], scenario="Historical")
# scenario_df.to_csv(IP_FILE, index=False)

IP_history = pd.read_csv(IP_FILE,
                         parse_dates=['Date'],
                         encoding="ISO-8859-1",
                         dtype={"RegionName": str},
                         error_bad_lines=True)
recent_date = IP_history['Date'].max()
states = IP_history[IP_history["Date"] == recent_date][["CountryName", "RegionName"]]
# predict("2021-01-03", "2021-01-03", IP_FILE, "prescriptions/preds.csv")
pred_df = pd.read_csv("prescriptions/preds.csv",
                      parse_dates=['Date'],
                      encoding="ISO-8859-1",
                      error_bad_lines=True)
states = pd.merge(states, pred_df, how='inner')

Q = Q_learning(states, 'not_currently_used', IP_history, END_DATE, gamma=0.9, alpha=1.0, epochs=10)
Q.run_Q()
