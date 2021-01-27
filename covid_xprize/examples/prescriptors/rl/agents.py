# File containing choices of agents
import numpy as np
from covid_xprize.examples.prescriptors.rl.utilities import IP_MAX_VALUES


"""
Agent based on https://github.com/openai/gym/blob/master/examples/agents/random_agent.py
"""

class RandomAgent(object):

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()


"""
Agent inspired by https://github.com/openai/gym/blob/master/examples/agents/cem.py
"""

class MultiActionLinearPolicy(object):
    def __init__(self, theta):
        self.W = theta[:int(len(theta) / 2)]
        self.b = theta[int(len(theta) / 2):]
    def act(self, ob):
        a = np.dot(ob, self.W) + self.b
        # To prevent values from exceeding IP_MAX_VALUES, just make boolean :)
        return (a < 0).astype(int)

def do_rollout(agent, env, num_steps, render=True):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        a = agent.act(ob)
        (ob, reward, done, _) = env.step(a)
        total_rew += reward
        if render: env.render()
        if done: break
    return total_rew, t + 1

def noisy_evaluation(theta, env, num_steps):
	agent = MultiActionLinearPolicy(theta)
	rew, T = do_rollout(agent, env, num_steps)
	return rew

def cem(f, env, num_steps, th_mean, batch_size, n_iter, elite_frac, initial_std=1.0):
    """
    Generic implementation of the cross-entropy method for maximizing a black-box function
    f: a function mapping from vector -> scalar
    th_mean: initial mean over input distribution
    batch_size: number of samples of theta to evaluate per batch
    n_iter: number of batches
    elite_frac: each batch, select this fraction of the top-performing samples
    initial_std: initial standard deviation over parameter vectors
    """
    n_elite = int(np.round(batch_size * elite_frac))
    th_std = np.ones_like(th_mean) * initial_std

    for _ in range(n_iter):
        ths = np.array([th_mean + dth for dth in  th_std[None,:] * np.random.randn(batch_size, th_mean.size)])
        ys = np.array([f(th, env, num_steps) for th in ths])
        elite_inds = ys.argsort()[::-1][:n_elite]
        elite_ths = ths[elite_inds]
        th_mean = elite_ths.mean(axis=0)
        th_std = elite_ths.std(axis=0)
        yield {'ys' : ys, 'theta_mean' : th_mean, 'y_mean' : ys.mean()}


