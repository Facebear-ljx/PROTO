from typing import Dict

import flax.linen as nn
import gym
import numpy as np


def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int, offline: bool, verbose: bool = False) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    temperature = 1. if env.spec._env_name.split("-")[0] == 'antmaze' else 0.
    
    for _ in range(num_episodes):
        observation, done = env.reset(), False

        while not done:
            action = agent.sample_actions(observation, temperature=temperature, offline=offline)
            observation, _, done, info = env.step(action)

        for k in stats.keys():
            stats[k].append(info['episode'][k]) 
            if verbose:
                v = info['episode'][k]
                print(f'{k}:{v}')

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
