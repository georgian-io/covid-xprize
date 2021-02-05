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
from tianshou.data import Batch
# For contants
from covid_xprize.examples.prescriptors.rl.utilities import IP_MAX_VALUES

# Modify these
path_to_xprize = "/home/ubuntu/mburhanpurkar/covid-xprize/"
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
task = "covid-env-states-v0"
env = gym.make(task, country=country, IP_history_file=IP_FILE, prescription_temp_file="prescription_temp_lr0.1_dqn.csv", prediction_temp_file="predictions/prediction_temp_lr0.1_dqn.csv")


# Hyperparameters
lr, epoch, batch_size = 1e-1, 2, 64
train_num, test_num = 1, 1
gamma, n_step, target_freq = 0.9, 3, 320
buffer_size = 20000
eps_train, eps_test = 0.1, 0.05
step_per_epoch, collect_per_step = 4, 10
writer = SummaryWriter('dqn-agent-results/dqn3-lr0.1')


# Make environments
train_envs = ts.env.DummyVectorEnv([lambda: gym.make(task, country=country, IP_history_file=IP_FILE, prescription_temp_file="prescription_temp_lr0.1_dqn.csv", prediction_temp_file="predictions/prediction_temp_lr0.1_dqn.csv") for _ in range(train_num)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task, country=country, IP_history_file=IP_FILE, prescription_temp_file="prescription_temp_lr0.1_dqn.csv", prediction_temp_file="predictions/prediction_temp_lr0.1_dqn.csv") for _ in range(test_num)])
train_envs.is_async = True
test_envs.is_async = True

# Define the network
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = np.array(list(IP_MAX_VALUES.values())) + 1 #env.action_space.shape or env.action_space.n
print("Action shape", action_shape)
print("State shape", state_shape)
#import pdb; pdb.set_trace()
#import timeit; start = timeit.timeit()
net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[256, 256, 256]) #[128, 128, 128])
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
#end = timeit.timeit()
#print("Initializing net", end - start)

# Set uo the policy and collectors
#start = timeit.timeit()
policy = ts.policy.DQNPolicy(net, optimizer, gamma, n_step, target_update_freq=target_freq)
train_collector = ts.data.Collector(policy, train_envs)
test_collector = ts.data.Collector(policy, test_envs)
#end = timeit.timeit()
#print("Initializing policy and collectors", end - start)

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
