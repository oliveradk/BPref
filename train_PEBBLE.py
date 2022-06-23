#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import tqdm
from collections import deque
from pyvirtualdisplay import Display
import hydra

from logger import Logger
from replay_buffer import ReplayBuffer
from reward_model import RewardModel
from teacher import Teachers, Teacher
import utils


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False
        self.log_info = False
        
        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
            self.log_info = True
        else:
            self.env = utils.make_env(cfg)
        
        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device)
        
        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # instantiating the reward model
        self.reward_model = RewardModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation, 
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch, 
            large_batch=cfg.large_batch, 
            label_margin=cfg.label_margin)
        
        #instantiate the teaches
        cfg.teacher.params.ds = self.env.observation_space.shape[0]
        cfg.teacher.params.da = self.env.action_space.shape[0]
        self.teachers = hydra.utils.instantiate(cfg.teacher)
        self.teachers.set_env(self.env)
        
    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0
        if self.log_info:
            avg_env_info = {}
            sample_action = self.env.action_space.sample()
            obs = self.env.reset()
            _, env_info = self.env.evaluate_state(obs, sample_action)
            for key in env_info.keys():
                avg_env_info[key] = 0.0
        
        if self.cfg.log_episode:
            utils.log_episode(self.env, self.agent, self.logger._sw, f'step_{self.step}')
        
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, extra = self.env.step(action)
                
                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])
                if self.log_info:
                    _, env_info = self.env.evaluate_state(obs, action)
                    for key, value in env_info.items():
                        avg_env_info[key] += value
                    
                
            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success
            
        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0
        if self.log_info:
            for key in avg_env_info.keys():
                avg_env_info[key] /= self.cfg.num_eval_episodes
        
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward,
                        self.step)
        #TODO: parameter for metaworld state info
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                    self.step)
            self.logger.log('train/true_episode_success', success_rate,
                        self.step)
        if self.log_info:
            for key, value in avg_env_info.items():
                self.logger.log(f'eval/avg_{key}', value, self.step)
        self.logger.dump(self.step)
    
    def learn_reward(self, first_flag=0): 
        # get querries
        queries, noisy_queries = 0, 0
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            queries = self.reward_model.uniform_sampling()
        else:
            if self.cfg.feed_type == 0:
                queries = self.reward_model.uniform_sampling()
            elif self.cfg.feed_type == 1:
                queries = self.reward_model.disagreement_sampling()
            elif self.cfg.feed_type == 2:
                queries = self.reward_model.entropy_sampling()
            elif self.cfg.feed_type == 3:
                queries = self.reward_model.kcenter_sampling()
            elif self.cfg.feed_type == 4:
                queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg.feed_type == 5:
                queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError
        sa_t_1, sa_t_2, r_t_1, r_t_2 = queries
        save_dir = '/home/danielskoch/BPref/saved_objs'
        # np.save(os.path.join(save_dir, 'sa_t_1'), sa_t_1)
        # np.save(os.path.join(save_dir, 'sa_t_2'), sa_t_2)
        # np.save(os.path.join(save_dir, 'r_t_1'), r_t_1)
        # np.save(os.path.join(save_dir, 'r_t_2'), r_t_2)
        # get teacher
        teacher = self.teachers.uniform_sampling(sa_t_1, sa_t_2)
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = teacher.get_label(*queries)
        
        #  put querries
        if len(labels) > 0:
            self.reward_model.put_queries(sa_t_1, sa_t_2, labels)
        
        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += len(labels)
        
        train_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or teacher.eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)
                
                if total_acc > 0.97:
                    break;
                    
        print("Reward function is updated!! ACC: " + str(total_acc))

    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        if self.log_info:
            sum_env_info = {}
            sample_obs = self.env.reset()
            sample_action = self.env.action_space.sample()
            _, env_info = self.env.evaluate_state(sample_obs, sample_action)
            for key in env_info.keys():
                sum_env_info[key] = 0.0

        true_episode_reward = 0
        
        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10) 
        start_time = time.time()
        interact_count = 0
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)
                
                if self.log_success:
                    self.logger.log('train/episode_success', episode_success,
                        self.step)
                    self.logger.log('train/true_episode_success', episode_success,
                        self.step)
                if self.log_info:
                    for key, value in sum_env_info.items():
                        self.logger.log(f'train/sum_{key}', value, self.step)
                
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                if self.log_info:
                    for key in sum_env_info.keys():
                        sum_env_info[key] = 0.0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)
                        
            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update                
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                # update schedule
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)
                
                # update margin --> not necessary / will be updated soon
                new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                self.teachers.set_teacher_thres_skip(new_margin)
                self.teachers.set_teacher_thres_equal(new_margin)
                
                # first learn reward
                self.learn_reward(first_flag=1)
                
                # relabel buffer
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                
                # reset Q due to unsuperivsed exploration
                self.agent.reset_critic()
                
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
                
                # reset interact_count
                interact_count = 0
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                        # update schedule
                        if self.cfg.reward_schedule == 1: #decay 
                            frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2: #increase 
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                        else:
                            frac = 1 #uniform
                        self.reward_model.change_batch(frac)
                        
                        # update margin --> not necessary / will be updated soon
                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                        self.teachers.set_teacher_thres_skip(new_margin)
                        self.teachers.set_teacher_thres_equal(new_margin)
                        
                        # corner case: new total feed > max feed
                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                            
                        self.learn_reward()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        interact_count = 0
                        
                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, 
                                            gradient_update=1, K=self.cfg.topK)
                
            next_obs, reward, done, extra = self.env.step(action)
            reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward_hat
            true_episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
            
            if self.log_info:
                _, env_info = self.env.evaluate_state(obs, action) 
                for key, value in env_info.items():
                    sum_env_info[key] += value
                
            # adding data to the reward training data
            self.reward_model.add_data(obs, action, reward, done)
            self.replay_buffer.add(
                obs, action, reward_hat, 
                next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1
            
        self.agent.save(self.work_dir, self.step)
        self.reward_model.save(self.work_dir, self.step)
        
@hydra.main(config_path='config/train_PEBBLE.yaml')
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    with Display(backend='xvfb') as disp:
        main()  