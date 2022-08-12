import collections.abc

import numpy as np
import torch
import torch.nn.functional as F
import gym
import os
import random
import math
import dmc2gym
import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict

from collections import deque
from gym.wrappers.time_limit import TimeLimit
from rlkit.envs.wrappers import NormalizedBoxEnv
from collections import deque
from skimage.util.shape import view_as_windows
from torch import nn
from torch import distributions as pyd

from stable_baselines3.common.vec_env.base_vec_env import VecEnv
    
def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=False)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env

def ppo_make_env(env_id, seed):
    """Helper function to create dm_control environment"""
    if env_id == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = env_id.split('_')[0]
        task_name = '_'.join(env_id.split('_')[1:])

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=seed,
                       visualize_reward=True)
    env.seed(seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias
    
def make_metaworld_env(cfg):
    env_name = cfg.env.replace('metaworld_','')
    if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]
    
    env = env_cls()
    
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(cfg.seed)
    
    return TimeLimit(NormalizedBoxEnv(env), env.max_path_length)

def ppo_make_metaworld_env(env_id, seed):
    env_name = env_id.replace('metaworld_','')
    if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]
    
    env = env_cls()
    
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(seed)
    
    return TimeLimit(env, env.max_path_length)

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))
    
class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu
    
class TorchRunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, axis=0)
            batch_var = torch.var(x, axis=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    @property
    def std(self):
        return torch.sqrt(self.var)


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta + batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def log_episode(env, agent, writer, tag, meta, log_video=True, log_info=True, log_obs=False, teachers=None):
    done = False
    obs = env.reset()
    agent.reset()

    frames = []
    step=0
    while not done:
        if log_video:
            frame = env.render(offscreen=True) if meta else env.render(mode='rgb_array')
            frames.append(frame)
        with eval_mode(agent):
            action = agent.act(obs, sample=False)
        obs, _reward, done, info = env.step(action)
        if log_info:
            for key, value in info.items():
                writer.add_scalar(f'{tag}/{key}', value, step)
        step += 1

        if log_obs:
            for i, ob_val in enumerate(obs):
                writer.add_scalar(f'obs/obs_{i}', ob_val, step)
        if teachers:
            sa_t = np.concatenate((obs, action))[None,:]
            info_t = [[info]]
            for i, teacher in enumerate(teachers.teachers):
                beta = teacher.get_beta(sa_t, info_t)
                writer.add_scalar(f'{tag}/{type(teacher)}_{i}', beta, step)
    if log_video:
        frames = torch.from_numpy(np.array(frames))
        frames = frames.unsqueeze(0)
        frames = frames.permute(0, 1, 4, 2, 3)
        writer.add_video(f'{tag}/video', frames, step, fps=30)


def get_info_list(info, keys):
    list_dict = {key: [] for key in keys}
    for info_dict in info:
        for key in keys:
            list_dict[key].append(info_dict[key])
    return list_dict


def get_partial_reward(info, reward_key):
    info_list_dict = get_info_list(info, [reward_key])
    rew = info_list_dict[reward_key]
    return np.array(rew)[:, None]


def box_vol(box):
    widths = box_widths(box)
    return widths.prod()

def box_widths(box):
    return box.high - box.low

def remove_duplicates(obs_space):
    return obs_space[np.r_[0:18, 36:]]

def strata_width(strata):
    return [np.array(stratum[1]) - np.array(stratum[0]) for stratum in strata]

def strata_center(strata):
    return [np.array(stratum[0]) + ((np.array(stratum[1]) - np.array(stratum[0])) / 2) for stratum in strata]

def extend_param(param, n):
    if not isinstance(param, collections.abc.Sized):
        param = [param] * n
    elif len(param) != n:
        raise ValueError('number of params must match')
    return param

def arr_to_list(arr):
    return [float(el) for el in arr]

def dict_to_struct_array(dict):
    dtype = [(key, type(value)) for key, value in dict.items()]
    return np.array(tuple(dict.values()), dtype=dtype)

def apply_map(func, l):
    return [el for el in map(func, l)]

def remove_duplicates(df):
    return df.loc[:, ~df.columns.duplicated()]

def attr_arr(ls, attr):
    return np.array([getattr(el, attr) for el in ls])
