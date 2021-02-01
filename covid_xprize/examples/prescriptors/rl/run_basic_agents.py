import gym
import gym_covid
import numpy as np
import pandas as pd
import os, pickle, json
# For data generation
from covid_xprize.scoring.predictor_scoring import load_dataset
from covid_xprize.validation.scenario_generator import generate_scenario
# For random agent and CEM agent
from covid_xprize.examples.prescriptors.rl.agents import RandomAgent
from covid_xprize.examples.prescriptors.rl.agents import noisy_evaluation, cem, MultiActionLinearPolicy


# Modify these
path_to_xprize = "/Users/mayaburhanpurkar/Documents/georgian-xprize/covid-xprize/"
country = "Canada"
IP_FILE = path_to_xprize + "robojudge_test_scenario_" + country + ".csv"
method = "cem"


# Generate data files if necessary
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


# Create the environment and agent
env = gym.make("covid-env-v0", country=country, IP_history_file=IP_FILE)
observation = env.reset()
if method == "random":
	agent = RandomAgent(env.action_space)
else:
	num_steps = 200
	params = dict(n_iter=10, batch_size=25, elite_frac=0.2)
	outdir = '/cem-agent-results'  # You will need to mkdir cem-agent-results the first time you run


# Two examples of running the training
# random: runs a very simplistic random agent, doesn't save policies
# cem: not practically useful, but offers a better idea of how to code your own agent!
if method == "random":
	for _ in range(1000):
		# Render prints the number of cases to the terminal
		env.render()

		# Get the next action from the agent
		action = agent.act()

		# Observation contains the number of cases
		observation, reward, done, info = env.step(action)

		# The simulation currently never ends
		if done:
			observation = env.reset()

elif method == "cem":
    for (i, iterdata) in enumerate(cem(noisy_evaluation, env, num_steps, np.zeros(env.action_space.shape[0] * 2), **params)):
        print('Iteration %2i. Episode mean reward: %7.3f'%(i, iterdata['y_mean']))

        # Write agent to file
        agent = MultiActionLinearPolicy(iterdata['theta_mean'])
        with open(path.join(outdir, 'cem-agent-results/agent-%.4i.pkl' % i), 'w') as fh: fh.write(str(pickle.dumps(agent, -1)))

    # Write params to file
    with open(path.join(outdir, 'info.json'), 'w') as fh: fh.write(json.dumps(params))

env.close()
