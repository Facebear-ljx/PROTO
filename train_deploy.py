import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from typing import Tuple
from typing import Tuple

import datetime
import time
import gym
import numpy as np
import sys
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
from dataclasses import dataclass

import wrappers
from dataset_utils import (Batch, D4RLDataset, ReplayBuffer,
                           split_into_trajectories)
from evaluation import evaluate
from learner import Learner
import wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'hopper-medium-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 100,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('num_pretraining_steps', int(1e5),
                     'Number of pretraining steps.')
flags.DEFINE_float('temp', 0.6, 'Loss temperature')
flags.DEFINE_float('tau_actor', 0.005, 'actor moving average')
flags.DEFINE_boolean('double', True, 'Use double q-learning')
flags.DEFINE_integer('replay_buffer_size', None,
                     'Replay buffer size (=max_steps if unspecified).')
flags.DEFINE_integer('init_dataset_size', None,
                     'Offline data size (uses all data if unspecified).')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('vanilla', False, 'Use vanilla RL training')
flags.DEFINE_boolean('auto_alpha', True, 'SAC temperature auto adjustment')
flags.DEFINE_boolean('symmetric', False, 'symmetric sampling trick')
flags.DEFINE_integer('sample_random_times', 0, 'Number of random actions to add to smooth dataset')
flags.DEFINE_integer('update_per_step', 1, 'Number of gradient updates per online sample')
flags.DEFINE_boolean('grad_pen', False, 'Add a gradient penalty to critic network')
flags.DEFINE_float('lambda_gp', 1, 'Gradient penalty coefficient')
flags.DEFINE_float('max_clip', 7., 'Loss clip value')
flags.DEFINE_integer('num_v_updates', 1, 'Number of value updates per iter')
flags.DEFINE_boolean('log_loss', False, 'Use log gumbel loss')
flags.DEFINE_boolean('noise', False, 'Add noise to actions')

flags.DEFINE_string('CUDA_id', '2', 'CUDA_VISIBLE_DEVICES')

config_flags.DEFINE_config_file(
    'config',
    'configs/antmaze_finetune_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


@dataclass(frozen=True)
class ConfigArgs:
    sample_random_times: int
    grad_pen: bool
    noise: bool
    lambda_gp: int
    max_clip: float
    num_v_updates: int
    # dual: bool
    log_loss: bool
    # log_grad: bool
    # euler_bias: bool
    # mod: bool


def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0
    return 1000/(compute_returns(trajs[-1]) - compute_returns(trajs[0]))


def make_env_and_dataset(env_name: str,
                         seed: int) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    if 'antmaze' in FLAGS.env_name:
        # dataset.rewards -= 1.0
        normalize_factor = 1.0
        pass  # normalized in the batch instead
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
          or 'hopper' in FLAGS.env_name):
        normalize_factor = normalize(dataset)

    return env, dataset, normalize_factor


def symmetric_sample(replay_buffer, replay_buffer_online, batch_size):
    indx_off = np.random.randint(replay_buffer.size, size=int(batch_size/2))
    indx_on = np.random.randint(replay_buffer_online.size, size=int(batch_size/2))
    
    return Batch(observations=np.concatenate([replay_buffer.observations[indx_off], replay_buffer_online.observations[indx_on]], axis=0),
                    actions=np.concatenate([replay_buffer.actions[indx_off], replay_buffer_online.actions[indx_on]], axis=0),
                    rewards=np.concatenate([replay_buffer.rewards[indx_off], replay_buffer_online.rewards[indx_on]], axis=0),
                    masks=np.concatenate([replay_buffer.masks[indx_off], replay_buffer_online.masks[indx_on]], axis=0),
                    next_observations=np.concatenate([replay_buffer.next_observations[indx_off], replay_buffer_online.next_observations[indx_on]], axis=0))

def main(_):
    # os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.CUDA_id
    # export CUDA_VISIBLE_DEVICES=2
    # tags = FLAGS.save_dir.split('/')[-1]
    if 'antmaze' not in FLAGS.env_name:
        FLAGS.eval_episodes = 10
        FLAGS.eval_interval = 5000
        FLAGS.temp = 2
    symmetric = FLAGS.symmetric
    np.random.seed(FLAGS.seed)
    
    wandb.init(project='XQL'+'deploy_eff',
               sync_tensorboard=True, reinit=True,  settings=wandb.Settings(_disable_stats=True))
    wandb.config.update(flags.FLAGS)
    wandb.run.name = f"{FLAGS.env_name}_{FLAGS.temp}"

    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(FLAGS.save_dir, ts_str)

    hparam_str_dict = dict(seed=FLAGS.seed, env=FLAGS.env_name)
    hparam_str = ','.join([
        '%s=%s' % (k, str(hparam_str_dict[k]))
        for k in sorted(hparam_str_dict.keys())
    ])

    summary_writer = SummaryWriter(os.path.join(save_dir, 'tb', hparam_str), write_to_disk=True)
    os.makedirs(save_dir, exist_ok=True)

    env, dataset, normalize_factor = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)

    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                                 np.maximum(int(2e+6), len(dataset.observations)))
    replay_buffer.initialize_with_dataset(dataset, FLAGS.init_dataset_size)

    # symmetric sampling
    if symmetric:
        replay_buffer_online = ReplayBuffer(env.observation_space, action_dim, 250000)
        replay_buffer_online.initialize_with_dataset(dataset, 25000)
    
    kwargs = dict(FLAGS.config)
    wandb.config.update(kwargs)

    args = ConfigArgs(sample_random_times=FLAGS.sample_random_times,
                      grad_pen=FLAGS.grad_pen,
                      lambda_gp=FLAGS.lambda_gp,
                      noise=FLAGS.noise,
                      max_clip=FLAGS.max_clip,
                      num_v_updates=FLAGS.num_v_updates,
                      log_loss=FLAGS.log_loss)
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    loss_temp=FLAGS.temp,
                    double_q=FLAGS.double,
                    vanilla=FLAGS.vanilla,
                    auto_alpha=FLAGS.auto_alpha,
                    tau_actor=FLAGS.tau_actor,
                    args=args,
                    **kwargs)

    best_eval_returns = -np.inf
    eval_returns = []
    observation, done = env.reset(), False

    # Use negative indices for pretraining steps.
    for i in tqdm.tqdm(range(1 - FLAGS.num_pretraining_steps,
                             FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i >= 1:
            action = agent.sample_actions(observation, offline=False, temperature=0)
            action = np.clip(action, -1, 1)
            next_observation, reward, done, info = env.step(action)

            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0
            
            if symmetric:
                # symmetric inserting
                replay_buffer_online.insert(observation, action, reward * normalize_factor, mask,
                                    float(done), next_observation)
            else:
                replay_buffer.insert(observation, action, reward * normalize_factor, mask,
                                    float(done), next_observation)                
            observation = next_observation

            if done:
                observation, done = env.reset(), False
                summary_writer.add_scalar(f'steps', i, info['total']['timesteps'])
                for k, v in info['episode'].items():
                    summary_writer.add_scalar(f'online_samples/{k}', v, info['total']['timesteps'])
        else:
            info = {}
            info['total'] = {'timesteps': i}

        if i >= 1 and symmetric:
            # online symmetric sampling
            batch = symmetric_sample(replay_buffer, replay_buffer_online, FLAGS.batch_size)
        else:
            batch = replay_buffer.sample(FLAGS.batch_size)
        
        if 'antmaze' in FLAGS.env_name:
            batch = Batch(observations=batch.observations,
                          actions=batch.actions,
                          rewards=batch.rewards - 1,
                          masks=batch.masks,
                          next_observations=batch.next_observations)
        if i < 0:
            update_info = agent.update_deploy_efficient(batch, offline=True, upon_bc=True)  # offline
            
            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f'steps', i, i)
                    if v.ndim == 0:
                        summary_writer.add_scalar(f'training/{k}', v, i)
                    else:
                        summary_writer.add_histogram(f'training/{k}', v, i, max_bins=512)
                summary_writer.flush()
                
            if i % FLAGS.eval_interval == 0:
                offline = False
                eval_stats = evaluate(agent, env, FLAGS.eval_episodes, offline)
 
                for k, v in eval_stats.items():
                    summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
                summary_writer.flush()
                print('reward:', eval_stats['return'])
                if eval_stats['return'] > best_eval_returns:
                    # Store best eval returns
                    best_eval_returns = eval_stats['return']
                    
                summary_writer.add_scalar(f'evaluation/best_returns', best_eval_returns, i)
                wandb.run.summary["best_returns"] = best_eval_returns
                
                eval_returns.append((i, eval_stats['return']))
                np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                        eval_returns,
                        fmt=['%d', '%.1f'])
        elif i % 100000 == 0:
            # agent.deployment_transfer() # offline2online transfer
            replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                        np.maximum(int(1e+6), len(dataset.observations)))
            replay_buffer.initialize_with_dataset(dataset, 1)
            pass            
        else:
            if (i + 1) % 100000 == 0:
                for j in range(100000):
                    batch = replay_buffer.sample(FLAGS.batch_size) 
                    update_info = agent.update_deploy_efficient(batch, offline=True, upon_bc=True)  # offline
 
                    if j % FLAGS.log_interval == 0:
                        for k, v in update_info.items():
                            summary_writer.add_scalar(f'steps', i - 100000 + j, i - 100000 + j)
                            if v.ndim == 0:
                                summary_writer.add_scalar(f'training/{k}', v, i - 100000 + j)
                            else:
                                summary_writer.add_histogram(f'training/{k}', v, i - 100000 + j, max_bins=512)
                        summary_writer.flush()
 
                    if j % FLAGS.eval_interval == 0:
                        offline = False
                        eval_stats = evaluate(agent, env, FLAGS.eval_episodes, offline)
 
                        for k, v in eval_stats.items():
                            summary_writer.add_scalar(f'evaluation/average_{k}s', v, i - 100000 + j)
                        summary_writer.flush()
                        print('reward:', eval_stats['return'])
                        if eval_stats['return'] > best_eval_returns:
                            # Store best eval returns
                            best_eval_returns = eval_stats['return']
                            
                        summary_writer.add_scalar(f'evaluation/best_returns', best_eval_returns, i - 100000 + j)
                        wandb.run.summary["best_returns"] = best_eval_returns
                        
                        eval_returns.append((i, eval_stats['return']))
                        np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                                eval_returns,
                                fmt=['%d', '%.1f'])

    wandb.finish()
    sys.exit(0)


if __name__ == '__main__':
    app.run(main)
