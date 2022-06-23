import argparse
import os

import hydra
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from teacher.grasp import GraspingTeachers

import utils
from teacher.standard import StandardTeachers

def log_agent(exp_path, log_dir, episodes):
    cfg = OmegaConf.load(os.path.join(exp_path, '.hydra', 'config.yaml'))
    sw = SummaryWriter(os.path.join(exp_path, log_dir))

    # make env
    if 'metaworld' in cfg.env:
        env = utils.make_metaworld_env(cfg)
    else:
        env = utils.make_env(cfg)


    cfg.agent.params.obs_dim = env.observation_space.shape[0]
    cfg.agent.params.action_dim = env.action_space.shape[0]
    cfg.agent.params.action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max())
    ]

    #load agent
    agent = hydra.utils.instantiate(cfg.agent)
    agent.load(model_dir=exp_path, step=1000000)

    #load teachers
    teachers = GraspingTeachers(0.5, env.observation_space.shape[0], 
        env.action_space.shape[0], 1, 0.5, 1, 0, 0, 0, 1, 0.5, 1, 0, 0, 0)
    teachers.set_env(env)
    
    for episode in range(episodes):
        utils.log_episode(env, agent, sw, f'episode_{episode}', log_video=True,
                          log_info=True, teachers=teachers)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', type=str, required=True)
    parser.add_argument('--agent_step', type=int, required=True)
    parser.add_argument('--log_dir', type=str, default='episode_logs')
    parser.add_argument('--episodes', type=int, default=1)
    
    args = parser.parse_args()

    log_agent(args.exp_path, args.log_dir, args.episodes)
