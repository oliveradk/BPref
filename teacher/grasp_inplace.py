import numpy as np

from teacher import Teacher, Teachers

import utils


class GraspOnlyTeacher(Teacher):

    def __init__(self,
        ds,
        da,
        beta, 
        gamma, 
        eps_mistake,
        eps_skip,
        eps_equal
    ):
        super().__init__(ds, da, gamma, eps_mistake, eps_skip, eps_equal)
        self.beta = beta
        self.ds = ds
        self.da = da
        
    def get_beta(self, sa_t):
        return np.ones((sa_t.shape[0], 1)) * self.beta
    
    def process_reward(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        grasp_r_t_1 = utils.get_partial_reward(sa_t_1, self.env, 'grasp_reward', self.ds, self.da)
        grasp_r_t_2 = utils.get_partial_reward(sa_t_2, self.env, 'grasp_reward', self.ds, self.da)
        return grasp_r_t_1, grasp_r_t_2

class InPlaceOnlyTeacher(Teacher):

    def __init__(self,
        ds, 
        da,
        beta, 
        gamma, 
        eps_mistake,
        eps_skip,
        eps_equal
    ):
        super().__init__(ds, da, gamma, eps_mistake, eps_skip, eps_equal)
        self.beta = beta
        self.ds = ds
        self.da = da
        
    def get_beta(self, sa_t):
        return np.ones((sa_t.shape[0], 1)) * self.beta
    
    def process_reward(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        grasp_r_t_1 = utils.get_partial_reward(sa_t_1, self.env, 'in_place_reward', self.ds, self.da)
        grasp_r_t_2 = utils.get_partial_reward(sa_t_2, self.env, 'in_place_reward', self.ds, self.da)
        return grasp_r_t_1, grasp_r_t_2

class GraspingInPlaceOnlyTeachers(Teachers):
    def __init__(self,
        ds, 
        da,
        beta,
        gamma,
        eps_mistake,
        eps_skip,
        eps_equal 
    ):
        n=2
        gamma = utils.extend_param(gamma, n)
        eps_mistake = utils.extend_param(eps_mistake, n)
        eps_skip = utils.extend_param(eps_skip, n)
        eps_equal = utils.extend_param(eps_equal, n)
        
        grasp_teacher = GraspOnlyTeacher(ds, da, beta[0],
        gamma[0], eps_mistake[0], eps_skip[0], eps_equal[0])

        nongrasp_teacher = InPlaceOnlyTeacher(ds, da, beta[1], gamma[1], 
        eps_mistake[1], eps_skip[1], eps_equal[1])
        
        super().__init__([grasp_teacher, nongrasp_teacher])