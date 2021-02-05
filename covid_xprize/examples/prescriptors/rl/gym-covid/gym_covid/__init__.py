import logging
from gym.envs.registration import register


logger = logging.getLogger(__name__)

# See documentation here https://github.com/openai/gym/blob/master/gym/envs/registration.py
register(
    id='covid-env-v0',
    entry_point='gym_covid.envs:CovidEnv',
    # timestep_limit=1000,    # optional: maximum number of steps that an episode can consist of
    # reward_threshold=1.0,   # optional: the reward threshold before the task is considered solved
    nondeterministic = True,  # optional: is the environment nondeterministic even after seeding
)
register(
    id='covid-env-states-v0',
    entry_point='gym_covid.envs:CovidEnvStates',
    # timestep_limit=1000,    # optional: maximum number of steps that an episode can consist of
    # reward_threshold=1.0,   # optional: the reward threshold before the task is considered solved
    nondeterministic = True,  # optional: is the environment nondeterministic even after seeding
)
