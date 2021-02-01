import gym
import gym_covid
import numpy as np
import pandas as pd
import os, pickle, json
# For data generation
from covid_xprize.scoring.predictor_scoring import load_dataset
from covid_xprize.validation.scenario_generator import generate_scenario
# For dqn from tianshou
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
from tianshou.utils.net.common import Net


# Modify these
path_to_xprize = "/Users/mayaburhanpurkar/Documents/georgian-xprize/covid-xprize/"
country = "Canada"
IP_FILE = path_to_xprize + "robojudge_test_scenario_" + country + ".csv"


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
task = "covid-env-v0"
env = gym.make(task, country=country, IP_history_file=IP_FILE)
observation = env.reset()


# Hyperparameters
lr, epoch, batch_size = 1e-3, 10, 64
train_num, test_num = 100, 8
gamma, n_step, target_freq = 0.9, 3, 320
buffer_size = 20000
eps_train, eps_test = 0.1, 0.05
step_per_epoch, collect_per_step = 1000, 10
writer = SummaryWriter('dqn-agent-results/dqn0')


# Make environments
train_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(train_num)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(test_num)])


# Define the network
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


# Set uo the policy and collectors
policy = ts.policy.DQNPolicy(net, optimizer, gamma, n_step, target_update_freq=target_freq)
train_collector = ts.data.Collector(policy, train_envs, ts.data.ReplayBuffer(buffer_size))
test_collector = ts.data.Collector(policy, test_envs)


# Train + display progress
result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector, epoch, step_per_epoch, collect_per_step,
    test_num, batch_size,
    train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
    test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
    #stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
    writer=writer)
print(f'Finished training! Use {result["duration"]}')


env.close()
