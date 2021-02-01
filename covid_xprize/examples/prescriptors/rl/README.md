## Train an RL agent using the OpenAI Gym!

### Setup
1. `mkdir predictions` (needed to call the `predict(...)` function)
2. `cd gym-covid`
3. `pip install -e .`
4. Make sure the `covid-xprize` repo is in your `$PYTHONPATH`
5. `python run_basic_agents.py` or `python run_dqn.py`

Note: the file `train_Q_learning.py` is buggy, but contains a rough outline of a standalone epsilon-greedy Q-leaning code!

### Structure
The `gym-covid` directory contains the information for the initial covid gym environment. To add new environments, just add your code to the `envs/` folder, edit the two `__init__.py` files to register and import the new environment respectively then re-install the environment with `pip`. 

The `run_basic_agents.py` and `python run_dqn.py` files contain the basic code needed to generate data for the XPrize competition, initialize an environment, and train a policy using an agent. Some examples of agents are provided in `agents.py`. 

