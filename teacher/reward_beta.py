import numpy as np

from teacher import Teacher, Teachers
import utils


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
    
    def get_beta(self, sa_t, info_t):
        info_lists = utils.get_info_lists(info_t, [self.UNSCALED_REWARD, self.reward_key])
        key_mean = np.array([np.mean(info_dict[self.reward_key]) for info_dict in info_lists])
        reward_mean = np.array([np.mean(info_dict[self.UNSCALED_REWARD]) for info_dict in info_lists])
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
        eps_equal,
        sampling
    ):
        self.reward_keys = reward_keys
        n = len(self.reward_keys)
        beta_scale = utils.extend_param(beta_scale, n)
        gamma = utils.extend_param(gamma, n)
        eps_mistake = utils.extend_param(eps_mistake, n)
        eps_skip = utils.extend_param(eps_skip, n)
        eps_equal = utils.extend_param(eps_equal, n)

        teachers = []
        for i, reward_key in enumerate(reward_keys):
            teacher = RewardBetaTeacher(reward_key, ds, da, beta_scale[i], 
            gamma[i], eps_mistake[i], eps_skip[i], eps_equal[i])
            teachers.append(teacher)
        super().__init__(teachers, sampling=sampling)
