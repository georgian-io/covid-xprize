import gym
import gym_covid
import numpy as np
import pandas as pd
import os, pickle, json, sys
# For data generation
from covid_xprize.scoring.predictor_scoring import load_dataset
from covid_xprize.validation.scenario_generator import generate_scenario
# For dqn from tianshou
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.data import Batch
# For contants
from covid_xprize.examples.prescriptors.rl.utilities import IP_MAX_VALUES

# Modify these
path_to_xprize = "/home/ubuntu/mburhanpurkar/covid-xprize/"
try:
    country = sys.argv[1] #"Canada"
except:
    country = "Canada"
IP_FILE = path_to_xprize + "robojudge_test_scenario_" + country + ".csv"
save_dir = "rl_saved_policies"

if not os.path.exists(path_to_xprize + save_dir):
    os.mkdir(path_to_xprize + save_dir)

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
env = gym.make(task, country=country, IP_history_file=IP_FILE, prescription_temp_file="prescription_temp.csv", prediction_temp_file="predictions/prediction_temp.csv")


# Hyperparameters
lr, epoch, batch_size = 1e-1, 10, 64
train_num, test_num = 1, 1
gamma, n_step, target_freq = 0.9, 7, 320
buffer_size = 20000
eps_train, eps_test = 0.1, 0.05
step_per_epoch, collect_per_step = 24, 10
writer = SummaryWriter(f'dqn-agent-results/sac4_long_run')


# Make environments
train_envs = ts.env.DummyVectorEnv([lambda: gym.make(task, country=country, IP_history_file=IP_FILE, prescription_temp_file="prescription_temp.csv", prediction_temp_file="predictions/prediction_temp.csv") for _ in range(train_num)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task, country=country, IP_history_file=IP_FILE, prescription_temp_file="prescription_temp.csv", prediction_temp_file="predictions/prediction_temp.csv") for _ in range(test_num)])
train_envs.is_async = True
test_envs.is_async = True

# Define the network
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
print("Action shape", action_shape)
print("State shape", state_shape)
#import pdb; pdb.set_trace()
#import timeit; start = timeit.timeit()
net = Net(state_shape=state_shape, action_shape=128, hidden_sizes=[64])
actor_net = ActorProb(net, action_shape=action_shape)
actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=lr)
critic1_net = Critic(net)
critic1_optimizer = torch.optim.Adam(critic1_net.parameters(), lr=lr)
critic2_net = Critic(net)
critic2_optimizer = torch.optim.Adam(critic2_net.parameters(), lr=lr)

#end = timeit.timeit()
#print("Initializing net", end - start)

# Set uo the policy and collectors
#start = timeit.timeit()
policy = ts.policy.SACPolicy(actor_net, actor_optimizer, critic1_net, critic1_optimizer, critic2_net, critic2_optimizer,
                                     (np.min(env.action_space.low), np.max(env.action_space.high)), gamma=gamma)
train_collector = ts.data.Collector(policy, train_envs)
test_collector = ts.data.Collector(policy, test_envs)
#end = timeit.timeit()
#print("Initializing policy and collectors", end - start)

# Train + display progress
#policy.set_eps(eps_train)
result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector, epoch, step_per_epoch, collect_per_step,
    test_num, batch_size,
    #train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
    #test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
    #stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
    writer=writer)
print(f'Finished training! Use {result["duration"]}')

torch.save(policy.state_dict(), f"{path_to_xprize + save_dir}/sac_{country}.pth")
#policy.load_state_dict(torch.load(f'{path_to_xprize + save_dir}/sac_{country}.pth'))

env.close()
