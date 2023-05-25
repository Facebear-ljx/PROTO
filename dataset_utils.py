import collections
from typing import Optional

import d4rl
import gym
import numpy as np
from tqdm import tqdm

import os
try:
    import mj_envs
except:
    pass

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


AWAC_DATA_DIR = "data"


def process_expert_dataset(expert_datset):
    """This is a mess, but works"""
    all_observations = []
    all_next_observations = []
    all_actions = []
    all_rewards = []
    all_terminals = []

    for x in expert_datset:
        all_observations.append(
            np.vstack([xx["state_observation"] for xx in x["observations"]])
        )
        all_next_observations.append(
            np.vstack([xx["state_observation"] for xx in x["next_observations"]])
        )
        all_actions.append(np.vstack([xx for xx in x["actions"]]))
        # for some reason rewards has an extra entry, so in rlkit they just remove the last entry: https://github.com/rail-berkeley/rlkit/blob/354f14c707cc4eb7ed876215dd6235c6b30a2e2b/rlkit/demos/source/dict_to_mdp_path_loader.py#L84
        all_rewards.append(x["rewards"][:-1])
        all_terminals.append(x["terminals"])

    return {
        "observations": np.concatenate(all_observations, dtype=np.float32),
        "next_observations": np.concatenate(all_next_observations, dtype=np.float32),
        "actions": np.concatenate(all_actions, dtype=np.float32),
        "rewards": np.concatenate(all_rewards, dtype=np.float32),
        "terminals": np.concatenate(all_terminals, dtype=np.float32),
    }


def process_bc_dataset(bc_dataset):
    final_bc_dataset = {k: [] for k in bc_dataset[0] if "info" not in k}

    for x in bc_dataset:
        for k in final_bc_dataset:
            final_bc_dataset[k].append(x[k])

    return {
        k: np.concatenate(v, dtype=np.float32).squeeze()
        for k, v in final_bc_dataset.items()
    }


def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
            next_observations)


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])


class D4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']))


class BinaryDataset(Dataset):
    def __init__(
        self,
        env: gym.Env,
        clip_to_eps: bool = True,
        eps: float = 1e-5,
        remove_terminals=True,
        include_bc_data=True,
    ):

        env_prefix = env.spec._env_name.split("-")[0]

        expert_dataset = np.load(
            os.path.join(
                os.path.expanduser(AWAC_DATA_DIR), f"{env_prefix}2_sparse.npy"
            ),
            allow_pickle=True,
        )

        # this seems super random, but I grabbed it from here: https://github.com/rail-berkeley/rlkit/blob/c81509d982b4d52a6239e7bfe7d2540e3d3cd986/rlkit/launchers/experiments/awac/awac_rl.py#L124 and here https://github.com/rail-berkeley/rlkit/blob/354f14c707cc4eb7ed876215dd6235c6b30a2e2b/rlkit/demos/source/dict_to_mdp_path_loader.py#L153
        dataset_split = 0.9
        last_train_idx = int(dataset_split * len(expert_dataset))

        dataset_dict = process_expert_dataset(expert_dataset[:last_train_idx])

        if include_bc_data:
            bc_dataset = np.load(
                os.path.join(
                    os.path.expanduser(AWAC_DATA_DIR), f"{env_prefix}_bc_sparse4.npy"
                ),
                allow_pickle=True,
            )

            # this seems super random, but I grabbed it from here: https://github.com/rail-berkeley/rlkit/blob/c81509d982b4d52a6239e7bfe7d2540e3d3cd986/rlkit/launchers/experiments/awac/awac_rl.py#L124 and here https://github.com/rail-berkeley/rlkit/blob/354f14c707cc4eb7ed876215dd6235c6b30a2e2b/rlkit/demos/source/dict_to_mdp_path_loader.py#L153
            bc_dataset_split = 0.9
            bc_dataset = bc_dataset[: int(bc_dataset_split * len(bc_dataset))]
            bc_dataset = process_bc_dataset(bc_dataset)

            dataset_dict = {
                k: np.concatenate([dataset_dict[k], bc_dataset[k]])
                for k in dataset_dict
            }

        if clip_to_eps:
            lim = 1 - eps
            dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        dones_float = np.zeros_like(dataset_dict['rewards'])

        for i in range(len(dones_float) - 1):
            if (dataset_dict['rewards'][i] == 0. or dataset_dict["terminals"][i] == 1.0):
                dones_float[i] = 1.
        
        dataset_dict["terminals"] = dones_float.astype(bool)

        dataset_dict["masks"] = 1.0 - dones_float
        # del dataset_dict["terminals"]
        
        truncated_index = []
        for i in range(len(dones_float) - 1):
            if (dones_float[i] == 1. and dones_float[i+1] == 1.):
                # truncated_index.append(i)
                pass
            else:
                truncated_index.append(i)
                

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)


        super().__init__(dataset_dict['observations'][truncated_index].astype(np.float32),
                         actions=dataset_dict['actions'][truncated_index].astype(np.float32),
                         rewards=dataset_dict['rewards'][truncated_index].astype(np.float32),
                         masks=dataset_dict['masks'][truncated_index].astype(np.float32),
                         dones_float=dones_float[truncated_index].astype(np.float32),
                         next_observations=dataset_dict['next_observations'][truncated_index].astype(np.float32),
                         size=len(dataset_dict['observations'][truncated_index]))


class ReplayBuffer(Dataset):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int):

        observations = np.empty((capacity, *observation_space.shape),
                                dtype=observation_space.dtype)
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity, ), dtype=np.float32)
        masks = np.empty((capacity, ), dtype=np.float32)
        dones_float = np.empty((capacity, ), dtype=np.float32)
        next_observations = np.empty((capacity, *observation_space.shape),
                                     dtype=observation_space.dtype)
        super().__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         size=0)

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset,
                                num_samples: Optional[int]):
        assert self.insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert self.capacity >= num_samples, 'Dataset cannot be larger than the replay buffer capacity.'

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[
            indices]

        self.insert_index = min(num_samples, self.capacity - 1)
        self.size = num_samples

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
