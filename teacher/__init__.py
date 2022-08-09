import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

import utils

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
    
    def get_beta(self, sa, info):
        raise NotImplementedError
    
    def process_reward(self, sa_1, sa_2, r_1, r_2, info_1, info_2):
        return r_1, r_2


class Teachers():
    def __init__(self, teachers: List[Teacher], sampling='uniform'):
        self.teachers = teachers
        self.sampling = sampling
    
    def __len__(self):
        return len(self.teachers)
    
    def get_labels(self, teacher_ids, sa_t_1, sa_t_2, r_t_1, r_t_2, info_t_1, info_t_2):
        # get teaches
        teachers = [self.teachers[int(i)] for i in teacher_ids]
        # process rewards
        r_ts = [teacher.process_reward(sa_t_1[i], sa_t_2[i], r_t_1[i], r_t_2[i], info_t_1[i], info_t_2[i]) \
            for i, teacher in enumerate(teachers)]
        r_t_1 = np.concatenate([r_t[0] for r_t in r_ts], axis=1).transpose(1, 0)[:, :, None]
        r_t_2 = np.concatenate([r_t[1] for r_t in r_ts], axis=1).transpose(1, 0)[:, :, None]
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # skip the query
        thresh_skip = np.array([teacher.thresh_skip for teacher in teachers])[:, None]
        if thresh_skip.sum() > 0:
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            assert max_r_t.shape == thresh_skip.shape
            max_index = (max_r_t > thresh_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []

            sa_t_1 = sa_t_1[max_index]
            sa_t_2 = sa_t_2[max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)
            teacher_ids = teacher_ids[max_index]
            teachers = [teachers[int(i)] for i in teacher_ids]
    
        # equally preferable
        eps_equal = np.array([teacher.eps_equal for teacher in teachers])[:, None]
        assert sum_r_t_1.shape == eps_equal.shape
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < eps_equal).reshape(-1)
        
        # perfectly rational
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        gamma = np.array([teacher.gamma for teacher in teachers])[:, None, None]
        for index in range(seg_size-1):
            assert len(gamma.shape) == len(temp_r_t_1[:,:index+1].shape)
            temp_r_t_1[:,:index+1] = (temp_r_t_1[:,:index+1] * gamma)
            temp_r_t_2[:,:index+1]  = (temp_r_t_2[:,:index+1] * gamma)
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)
            
        # Bradley-Terry rational model #TODO: allow for -beta to be perfect rationality
        beta_1 = np.array([teacher.get_beta(sa_t_1[i], info_t_1[i]) for i, teacher in enumerate(teachers)])[:, None]
        beta_2 = np.array([teacher.get_beta(sa_t_2[i], info_t_2[i]) for i, teacher in enumerate(teachers)])[:, None]
        betas = np.concatenate([beta_1, beta_2], axis=1)
        beta = np.mean(betas, axis=1)[:, None]

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
        eps_mistake = np.array([teacher.eps_mistake for teacher in teachers])
        assert rand_num.shape == eps_mistake.shape 
        noise_index = rand_num <= eps_mistake
        labels[noise_index] = 1 - labels[noise_index]
 
        # equally preferable
        labels[margin_index] = -1 
        
        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels, teacher_ids


    def set_teacher_thres_skip(self, new_margin):
        for teacher in self.teachers:
            teacher.set_thres_skip(new_margin)
    
    def set_teacher_thres_equal(self, new_margin):
        for teacher in self.teachers:
            teacher.set_thres_equal(new_margin)
    
    def set_env(self, env, log_dir=None):
        for teacher in self.teachers:
            teacher.set_env(env)
