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
    
    def get_beta(self, sa_t):
        return np.ones((sa_t.shape[0], 1)) * self.beta


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
        teachers = []
        for i in range(n):
            teachers.append(StandardTeacher(beta[i], ds, da, gamma[i], 
                            eps_mistake[i], eps_skip[i], eps_equal[i]))
        super().__init__(teachers)
