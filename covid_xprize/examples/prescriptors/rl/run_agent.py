import gym
import gym_covid
import numpy as np
import pandas as pd


# To generate this file, run the prescriptor_robojudge notebook
# Change countries = None to countries = ["Canada"]
# Change scenario="Freeze" to scenario="Historical"
# Modify the start and end dates as desired
# You will need to make the prescriptions/ directory in the root of the xprize repo
# You will also need to make the predictions/ directory where this file is located
IP_history_file = "/Users/mayaburhanpurkar/Documents/georgian-xprize/covid-xprize/prescriptions/robojudge_test_scenario.csv"
# pred_df = pd.read_csv(IP_history_file,
#                       parse_dates=['Date'],
#                       encoding="ISO-8859-1",
#                       error_bad_lines=True)
# print(pred_df.head())
# print(pred_df.loc[0, "RegionName"])
env = gym.make("covid-env-v0", IP_history_file=IP_history_file, costs_file="costs_file_not_currently_used.txt")
observation = env.reset()

for _ in range(1000):
	env.render()
	action = env.action_space.sample() # your agent here (this takes random actions)
	observation, reward, done, info = env.step(action)

	if done:
		observation = env.reset()

env.close()