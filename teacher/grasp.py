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
        
    def get_beta(self, sa, info):
        info_list_dict = utils.get_info_list(info, ['grasp_success'])
        grasp_mean = np.mean(info_list_dict['grasp_success'])
        beta = self.beta_1 if grasp_mean < self.grasp_thresh else self.beta_2
        return beta

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
        
    def get_beta(self, sa, info):
        info_list_dict = utils.get_info_list(info, ['grasp_success'])
        grasp_mean = np.mean(info_list_dict['grasp_success'])
        beta = self.beta_2 if grasp_mean < self.grasp_thresh else self.beta_1
        return beta

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
        eps_equal,
    ):
        n=2
        gamma = utils.extend_param(gamma, n)
        eps_mistake = utils.extend_param(eps_mistake, n)
        eps_skip = utils.extend_param(eps_skip, n)
        eps_equal = utils.extend_param(eps_equal, n)
        
        grasp_teacher = GraspTeacher(beta_1[0], beta_2[0], grasp_thresh, ds, da,
        gamma[0], eps_mistake[0], eps_skip[0], eps_equal[0])

        nongrasp_teacher = NonGraspTeacher(beta_1[1], beta_2[1], grasp_thresh, 
            ds, da, gamma[1], eps_mistake[1], eps_skip[1], eps_equal[1])
        
        super().__init__([grasp_teacher, nongrasp_teacher])
