import numpy as np

from .kernel.gaussian import GaussianThreshTeacher, Gaussian1DThreshTeachers

Y_RANGE_WALKER = [0, 1.8] #TODO: change
Y_INDEX_WALKER = 14


class WalkerYGaussian(GaussianThreshTeacher):
    def get_point(self, sa_1, sa_2, info_1, info_2):
        x_1 = sa_1[:, Y_INDEX_WALKER]
        x_2 = sa_2[:, Y_INDEX_WALKER]
        x = np.stack([x_1, x_2], axis=1)
        assert x.shape == (sa_1.shape[0], 2)
        return x


class WalkerYGaussianTeachers(Gaussian1DThreshTeachers):
    @property
    def bounds(self):
        return Y_RANGE_WALKER

    @property
    def expert_type(self):
        return WalkerYGaussian
