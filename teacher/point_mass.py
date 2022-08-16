import numpy as np

from teacher import Teachers
from teacher.standard import StandardTeacher
from .kernel.gaussian import GaussianThreshTeacher, Gaussian1DThreshTeachers
import utils

X_POS = 0
Y_POS = 1
X_VEL = 2
Y_VEL = 3

X_RANGE = [-.3, .3]
Y_RANGE = [-.3, .3]

QUADRANT_CENTROIDS = [[-0.15, -0.15], [-0.15, 0.15], [0.15, -0.15], [0.15, 0.15]]

class PointMassXGaussian(GaussianThreshTeacher):
    
    def get_point(self, sa_1, sa_2, info_1, info_2):
        x_1 = sa_1[:, X_POS]
        x_2 = sa_2[:, X_POS]
        x = np.stack([x_1, x_2], axis=1)
        assert x.shape == (sa_1.shape[0], 2)
        return x

class PointMassYGaussian(GaussianThreshTeacher):
    
    def get_point(self, sa_1, sa_2, info_1, info_2):
        x_1 = sa_1[:, Y_POS]
        x_2 = sa_2[:, Y_POS]
        x = np.stack([x_1, x_2], axis=1)
        assert x.shape == (sa_1.shape[0], 2)
        return x

class PointMassXYGaussian(GaussianThreshTeacher):

    def get_point(self, sa_1, sa_2, info_1, info_2):
        x_1 = sa_1[:, [X_POS, Y_POS]]
        x_2 = sa_2[:, [X_POS, Y_POS]]
        x = np.concatenate([x_1, x_2], axis=1)
        assert x.shape == (sa_1.shape[0], 4)
        return x

class PointMassXGaussianTeachers(Gaussian1DThreshTeachers):
    @property
    def bounds(self):
        return X_RANGE
    
    @property 
    def expert_type(self):
        return PointMassXGaussian

class PointMassYGaussianTeachers(Gaussian1DThreshTeachers):
    @property
    def bounds(self):
        return Y_RANGE
    
    @property 
    def expert_type(self):
        return PointMassYGaussian

class PointMassXYGaussianTeachers(Teachers):

    def __init__(
            self,
            ds,
            da,
            n_generalists,
            gamma,
            eps_mistake,
            eps_skip,
            eps_equal,
            beta_general,
            scale,
            thresh_val,
        ):
            self.ds = ds
            self.da = da
            self.n_experts=4
            self.n_generalists = n_generalists
            self.n_teachers = self.n_experts + self.n_generalists
            self.beta_general = beta_general
            self.scale = utils.extend_param(scale, self.n_experts)
            self.thresh_val = utils.extend_param(thresh_val, self.n_experts)
            self.gamma = utils.extend_param(gamma, self.n_teachers)
            self.eps_mistake = utils.extend_param(eps_mistake, self.n_teachers)
            self.eps_skip = utils.extend_param(eps_skip, self.n_teachers)
            self.eps_equal = utils.extend_param(eps_equal, self.n_teachers)
            self.define_teachers()

    def define_teachers(self):
        #define experts
        experts = []
        for i, centroid in enumerate(QUADRANT_CENTROIDS):
            expert = PointMassXYGaussian(
                ds=self.ds,
                da=self.da,
                centroid=np.concatenate([centroid, centroid]), 
                scale=self.scale[i],
                weights=np.ones(4),
                thresh=np.zeros(4),
                thresh_val=self.thresh_val[i],
                gamma=self.gamma[i],
                eps_mistake=self.eps_mistake[i],
                eps_skip=self.eps_skip[i],
                eps_equal=self.eps_equal[i]
            )
            experts.append(expert)
        
        # define generalists
        generalists = []
        for j in range(self.n_experts, self.n_teachers):
            generalist = StandardTeacher(
                beta=self.beta_general,
                ds=self.ds,
                da=self.da,
                gamma=self.gamma[j],
                eps_mistake=self.eps_mistake[j],
                eps_skip=self.eps_skip[i],
                eps_equal=self.eps_equal[j],
            )
            generalists.append(generalist)

        teachers = experts + generalists
        super().__init__(teachers)
        

