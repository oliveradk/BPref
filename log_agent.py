import argparse
import os

import hydra
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from teacher.grasp import GraspingTeachers
from teacher.depricated.gaussian_beta import GaussianBetaTeachers
from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerSweepEnvV2

import utils
from teacher.standard import StandardTeachers

def log_agent(exp_path, steps, log_dir, episodes, log_obs):
    cfg = OmegaConf.load(os.path.join(exp_path, '.hydra', 'config.yaml'))
    sw = SummaryWriter(os.path.join(exp_path, log_dir))

    meta = 'metaworld' in cfg.env
    # make env
    if meta:
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
    agent.load(model_dir=exp_path, step=steps)

    #load teachers
    cfg.teacher.params.ds = env.observation_space.shape[0]
    cfg.teacher.params.da = env.action_space.shape[0]
    teachers = hydra.utils.instantiate(cfg.teacher)
    teachers.set_env(env, log_dir)
    
    for episode in range(episodes):
        utils.log_episode(env=env,
                          agent=agent, 
                          writer=sw, 
                          tag=f'episode_{episode}',
                          meta=meta, 
                          log_video=True,
                          log_info=False, 
                          teachers=teachers, 
                          log_obs=log_obs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', type=str, required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--log_dir', type=str, default='episode_logs')
    parser.add_argument('--log_obs', default=False, action='store_true')
    parser.add_argument('--episodes', type=int, default=1)
    
    args = parser.parse_args()

    log_agent(args.exp_path, args.steps, args.log_dir, args.episodes, args.log_obs)
