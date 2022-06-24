import collections.abc

import numpy as np

from teacher import Teacher, Teachers
import utils

def extend_param(param, n):
    if not isinstance(param, collections.abc.Sized):
        param = [param] * n
    elif len(param) != n:
        raise ValueError('number of params must match')
    return param

class RewardBetaTeacher(Teacher):

    UNSCALED_REWARD = 'unscaled_reward'

    def __init__(self, 
        reward_key,
        ds, 
        da,
        beta_scale, 
        gamma,
        eps_mistake,
        eps_skip,
        eps_equal
    ):
        self.reward_key = reward_key
        self.beta_scale = beta_scale
        super().__init__(ds, da, gamma, eps_mistake, eps_skip, eps_equal)
    
    def get_beta(self, sa_t):
        info = utils.get_info(sa_t, self.env, 
                [self.UNSCALED_REWARD, self.reward_key], self.ds, self.da)
        key_mean = np.array([np.mean(info_dict[self.reward_key]) for info_dict in info])
        reward_mean = np.array([np.mean(info_dict[self.UNSCALED_REWARD]) for info_dict in info])
        return (key_mean / reward_mean) * self.beta_scale
        

class RewardBetaTeachers(Teachers):

    def __init__(self,
        reward_keys,
        ds,
        da, 
        beta_scale,
        gamma,
        eps_mistake,
        eps_skip,
        eps_equal
    ):
        self.reward_keys = reward_keys
        n = len(self.reward_keys)
        beta_scale = extend_param(beta_scale, n)
        gamma = extend_param(gamma, n)
        eps_mistake = extend_param(eps_mistake, n)
        eps_skip = extend_param(eps_skip, n)
        eps_equal = extend_param(eps_equal, n)

        teachers = []
        for i, reward_key in enumerate(reward_keys):
            teacher = RewardBetaTeacher(reward_key, ds, da, beta_scale[i], 
            gamma[i], eps_mistake[i], eps_skip[i], eps_equal[i])
            teachers.append(teacher)
        super().__init__(teachers)

# class RewardBetaGraspInPlace(RewardBetaTeachers):

#     def __init__(self,
#         ds,
#         da, 
#         beta,
#         gamma,
#         eps_mistake,
#         eps_skip,
#         eps_equal
#     ):
#         super().__init__(['grasp_reward, in_place_reward'], ds, da, beta,
#             gamma, eps_mistake, eps_skip, eps_equal)