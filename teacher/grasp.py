import numpy as np

from teacher import Teacher, Teachers

import utils


class GraspTeacher(Teacher):

    def __init__(self,
        beta_1,
        beta_2, 
        grasp_thresh,
        ds,
        da, 
        gamma, 
        eps_mistake,
        eps_skip,
        eps_equal
    ):
        super().__init__(ds, da, gamma, eps_mistake, eps_skip, eps_equal)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.grasp_thresh = grasp_thresh
        self.ds = ds
        self.da = da
        
    def get_betas(self, sa_t_1, sa_t_2):
        sa_1_info = utils.get_info(sa_t_1, self.env, ['grasp_success'], self.ds, self.da)
        sa_1_grasp_mean = np.array([np.mean(info_dict['grasp_success']) for info_dict in sa_1_info])

        sa_2_info = utils.get_info(sa_t_2, self.env, ['grasp_success'], self.ds, self.da)
        sa_2_grasp_mean = np.array([np.mean(info_dict['grasp_success']) for info_dict in sa_2_info])

        grasp_mean = np.mean([sa_1_grasp_mean, sa_2_grasp_mean], axis=0)

        betas = np.zeros(grasp_mean.shape)
        betas[grasp_mean <= self.grasp_thresh] = self.beta_1
        betas[grasp_mean > self.grasp_thresh] = self.beta_2

        betas = betas[:,None]

        return betas

class NonGraspTeacher(Teacher):

    def __init__(self,
        beta_1,
        beta_2, 
        grasp_thresh,
        ds, 
        da, 
        gamma, 
        eps_mistake,
        eps_skip,
        eps_equal
    ):
        super().__init__(ds, da, gamma, eps_mistake, eps_skip, eps_equal)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.grasp_thresh = grasp_thresh
        self.ds = ds
        self.da = da
        
    def get_betas(self, sa_t_1, sa_t_2):
        sa_1_info = utils.get_info(sa_t_1, self.env, ['grasp_success'], self.ds, self.da)
        sa_1_grasp_mean = np.array([np.mean(info_dict['grasp_success']) for info_dict in sa_1_info])

        sa_2_info = utils.get_info(sa_t_2, self.env, ['grasp_success'], self.ds, self.da)
        sa_2_grasp_mean = np.array([np.mean(info_dict['grasp_success']) for info_dict in sa_2_info])

        grasp_mean = np.mean([sa_1_grasp_mean, sa_2_grasp_mean], axis=0)

        betas = np.zeros(grasp_mean.shape)
        betas[grasp_mean > self.grasp_thresh] = self.beta_1
        betas[grasp_mean <= self.grasp_thresh] = self.beta_2

        betas = betas[:,None]

        return betas

class GraspingTeachers(Teachers):
    def __init__(self,
        grasp_thresh,
        ds, 
        da,
        beta_1,
        beta_2,
        gamma,
        eps_mistake,
        eps_skip,
        eps_equal 
    ):
        grasp_teacher = GraspTeacher(beta_1[0], beta_2[0], grasp_thresh, ds, da,
        gamma[0], eps_mistake[0], eps_skip[0], eps_equal[0])

        nongrasp_teacher = NonGraspTeacher(beta_1[1], beta_2[1], grasp_thresh, 
            ds, da, gamma[1], eps_mistake[1], eps_skip[1], eps_equal[1])
        
        super().__init__([grasp_teacher, nongrasp_teacher])
