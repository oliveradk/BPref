import numpy as np

from teacher import Teacher, Teachers


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
    
    def get_betas(self, sa_t_1, sa_t_2):
        return np.ones((sa_t_1.shape[0], 1)) * self.beta


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
        teachers = [StandardTeacher(beta, ds, da, gamma, eps_mistake, eps_skip,
        eps_equal)]
        super().__init__(teachers)
