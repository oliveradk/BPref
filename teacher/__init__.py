import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

class Teacher():
    def __init__(self,
        ds,
        da, 
        gamma,
        eps_mistake, 
        eps_skip,
        eps_equal
    ):
        self.ds = ds
        self.da = da
        self.gamma = gamma
        self.eps_mistake = eps_mistake
        self.eps_skip = eps_skip
        self.eps_equal = eps_equal
        self.thresh_skip = 0
        self.thres_equal = 0
        self.env = None
    
    def set_env(self, env):
        self.env = env
    
    def set_thres_skip(self, new_margin):
        self.thres_skip = new_margin * self.eps_skip
    
    def set_thres_equal(self, new_margin):
        self.thres_equal = new_margin * self.eps_equal
    
    def get_beta(self, sa_t, info_t):
        raise NotImplementedError
    
    def process_reward(self, sa_t_1, sa_t_2, r_t_1, r_t_2, info_t_1, info_t_2):
        return r_t_1, r_t_2

    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2, info_t_1, info_t_2):
        r_t_1, r_t_2 = self.process_reward(sa_t_1, sa_t_2, r_t_1, r_t_2, info_t_1, info_t_2)
        
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # skip the query
        if self.thres_skip > 0: 
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            max_index = (max_r_t > self.thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []

            sa_t_1 = sa_t_1[max_index]
            sa_t_2 = sa_t_2[max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.thres_equal).reshape(-1)
        
        # perfectly rational
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size-1):
            temp_r_t_1[:,:index+1] *= self.gamma
            temp_r_t_2[:,:index+1] *= self.gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)
            
        
        # Bradley-Terry rational model #TODO: allow for -beta to be perfect rationality
        beta_1 = self.get_beta(sa_t_1, info_t_1)
        beta_2 = self.get_beta(sa_t_2, info_t_2)
       
        beta = np.mean([beta_1, beta_2], axis=0)

        assert sum_r_t_1.shape == beta.shape
        assert sum_r_t_2.shape == beta.shape

        r_hat = torch.cat([torch.Tensor(sum_r_t_1), 
                            torch.Tensor(sum_r_t_2)], axis=-1)
        r_hat = r_hat * beta
        ent = F.softmax(r_hat, dim=-1)[:, 1]
        labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        
        # making a mistake
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.eps_mistake
        labels[noise_index] = 1 - labels[noise_index]
 
        # equally preferable
        labels[margin_index] = -1 
        
        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels

class Teachers():
    def __init__(self, teachers: List[Teacher], sampling='uniform'):
        self.teachers = teachers
        self.sampling = sampling
    
    def sample_teacher(self, sa_t_1, sa_t_2, info_t_1, info_t_2):
        if self.sampling == 'uniform':
            return self.uniform_sampling(sa_t_1, sa_t_2, info_t_1, info_t_2)
        elif self.sampling == 'max_beta':
            return self.max_beta(sa_t_1, sa_t_2, info_t_1, info_t_2)
        else: 
            raise ValueError(f"invalid teacher sampling method {self.sampling}")
    
    def uniform_sampling(self, sa_t_1, sa_t_2, info_t_1, info_t_2):
        return random.choice(self.teachers)
    
    def max_beta(self, sa_t_1, sa_t_2, info_t_1, info_t_2):
        betas = []
        for teacher in self.teachers:
            beta_1 = teacher.get_beta(sa_t_1, info_t_1)
            beta_2 = teacher.get_beta(sa_t_2, info_t_2)
            beta_sum_mean = (beta_1 + beta_2).mean()
            betas.append(beta_sum_mean)
        return self.teachers[np.argmax(betas)]


    def set_teacher_thres_skip(self, new_margin):
        for teacher in self.teachers:
            teacher.set_thres_skip(new_margin)
    
    def set_teacher_thres_equal(self, new_margin):
        for teacher in self.teachers:
            teacher.set_thres_equal(new_margin)
    
    def set_env(self, env, log_dir=None):
        for teacher in self.teachers:
            teacher.set_env(env)