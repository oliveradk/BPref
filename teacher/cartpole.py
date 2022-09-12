import numpy as np

from .kernel.gaussian import GaussianThreshTeacher, Gaussian1DThreshTeachers

X_RANGE_CARTPOLE = [-1.8, 1.8]
X_INDEX_CARTPOLE = 0


class CartpoleXGaussian(GaussianThreshTeacher):
    def get_point(self, sa_1, sa_2, info_1, info_2):
        x_1 = sa_1[:, X_INDEX_CARTPOLE]
        x_2 = sa_2[:, X_INDEX_CARTPOLE]
        x = np.stack([x_1, x_2], axis=1)
        assert x.shape == (sa_1.shape[0], 2)
        return x


class CartpoleXGaussianTeachers(Gaussian1DThreshTeachers):
    @property
    def bounds(self):
        return X_RANGE_CARTPOLE

    @property
    def expert_type(self):
        return CartpoleXGaussian
