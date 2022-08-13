import numpy as np

from .kernel.gaussian import GaussianThreshTeacher, Gaussian1DThreshTeachers

X_UPPER = 0
X_LOWER = 1
Y_UPPER = 2
Y_LOWER = 3
VEL_UPPER = 4
VEL_LOWER = 5

X_RANGE = [-1, 1]
Y_RANGE = [-1, 1]


class AcrobotXUpperGaussian(GaussianThreshTeacher):
    def get_point(self, sa_1, sa_2, info_1, info_2):
        x_1 = sa_1[:, X_UPPER]
        x_2 = sa_2[:, X_UPPER]
        x = np.stack([x_1, x_2], axis=1)
        assert x.shape == (sa_1.shape[0], 2)
        return x


class AcrobotYUpperGaussian(GaussianThreshTeacher):
    def get_point(self, sa_1, sa_2, info_1, info_2):
        x_1 = sa_1[:, Y_UPPER]
        x_2 = sa_2[:, Y_UPPER]
        x = np.stack([x_1, x_2], axis=1)
        assert x.shape == (sa_1.shape[0], 2)
        return x


class AcrobotXYUpperGaussian(GaussianThreshTeacher):
    def get_point(self, sa_1, sa_2, info_1, info_2):
        x_1 = sa_1[:, [X_UPPER, Y_UPPER]]
        x_2 = sa_2[:, [X_UPPER, Y_UPPER]]
        x = np.stack([x_1, x_2], axis=1)
        assert x.shape == (sa_1.shape[0], 4)
        return x


class AcrobotXUpperGaussianTeachers(Gaussian1DThreshTeachers):
    @property
    def bounds(self):
        return X_RANGE

    @property
    def expert_type(self):
        return AcrobotXUpperGaussian


class AcrobotYUpperGaussianTeachers(Gaussian1DThreshTeachers):
    @property
    def bounds(self):
        return Y_RANGE

    @property
    def expert_type(self):
        return AcrobotYUpperGaussian


# TODO: define xy teachers
