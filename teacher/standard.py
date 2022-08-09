import numpy as np

from teacher import Teacher, Teachers

import utils


class StandardTeacher(Teacher):
    
    def __init__(self, 
        beta, 
        ds, 
        da,  
        gamma, 
        eps_mistake,
        eps_skip,
        eps_equal
    ):
        self.beta = beta
        super().__init__(ds, da, gamma, eps_mistake, eps_skip, eps_equal)
    
    def get_beta(self, sa, info):
        return float(self.beta)


class StandardTeachers(Teachers):
    
    def __init__(self,
        beta, 
        ds,
        da, 
        gamma, 
        eps_mistake,
        eps_skip,
        eps_equal
    ):
        n = len(beta)
        gamma = utils.extend_param(gamma, n)
        eps_mistake = utils.extend_param(eps_mistake, n)
        eps_skip = utils.extend_param(eps_skip, n)
        eps_equal = utils.extend_param(eps_equal, n)

        teachers = []
        for i in range(n):
            teachers.append(StandardTeacher(beta[i], ds, da, gamma[i], 
                            eps_mistake[i], eps_skip[i], eps_equal[i]))
        super().__init__(teachers)
