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
    
    def get_betas(self, sa_t_1, sa_t_2):
        raise NotImplementedError

    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
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
        betas = self.get_betas(sa_t_1, sa_t_2)
        r_hat = torch.cat([torch.Tensor(sum_r_t_1), 
                            torch.Tensor(sum_r_t_2)], axis=-1)
        r_hat = r_hat*betas
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
    def __init__(self, teachers: List[Teacher]):
        self.teachers = teachers
    
    def uniform_sampling(self, sa_t_1, sa_t_2):
        return random.choice(self.teachers)
    
    def set_teacher_thres_skip(self, new_margin):
        for teacher in self.teachers:
            teacher.set_thres_skip(new_margin)
    
    def set_teacher_thres_equal(self, new_margin):
        for teacher in self.teachers:
            teacher.set_thres_equal(new_margin)
    
    def set_env(self, env):
        for teacher in self.teachers:
            teacher.set_env(env)