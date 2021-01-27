import os
import gym
import gym_covid
import numpy as np
import pandas as pd
from covid_xprize.scoring.predictor_scoring import load_dataset
from covid_xprize.validation.scenario_generator import generate_scenario


# Modify these
path_to_xprize = "/Users/mayaburhanpurkar/Documents/georgian-xprize/covid-xprize/"
country = "Canada"
IP_FILE = path_to_xprize + "robojudge_test_scenario_" + country + ".csv"

if not os.path.isfile(IP_FILE):
	print("Generating IP history for", country)

	# Generate a scenario for the country
	LATEST_DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
	GEO_FILE = path_to_xprize + "countries_regions.csv"
	latest_df = load_dataset(LATEST_DATA_URL, GEO_FILE)
	countries = [country]

	# If we are using historical data, the start and end dates are irrelevant (they specify the dates
	# between which we would like to freeze the IP, for example). Note also that END_DATE does not cut
	# off the data in the scenario early! 
	START_DATE = "2021-01-01"
	END_DATE = "2021-01-02"
	scenario_df = generate_scenario(START_DATE, END_DATE, latest_df, countries, scenario="Historical")
	scenario_df.to_csv(IP_FILE, index=False)

# Create the environment
env = gym.make("covid-env-v0", country=country, IP_history_file=IP_FILE, costs_file="costs_file_not_currently_used.txt")
observation = env.reset()

for _ in range(1000):
	# Render prints the number of cases to the terminal
	env.render()

	# Specify an agent here
	action = env.action_space.sample()

	# Observation contains the number of cases
	observation, reward, done, info = env.step(action)

	# The simulation currently never ends
	if done:
		observation = env.reset()

env.close()