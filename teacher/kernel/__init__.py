from abc import ABC, abstractmethod
import numpy as np

from teacher import Teacher


class KernelBetaTeacher(Teacher):
    def __init__(self, ds, da, gamma, eps_mistake, eps_skip, eps_equal):
        super().__init__(ds, da, gamma, eps_mistake, eps_skip, eps_equal)
        self.kernel = self.compute_kernel()

    @abstractmethod
    def compute_kernel(self):
        pass

    def get_beta(self, sa_1, sa_2, info_1, info_2):
        point = self.get_point(sa_1, sa_2, info_1, info_2)
        beta = self.kernel(point).mean()
        return beta

    def get_point(self, sa_1, sa_2, info_1, info_2):
        x_1 = sa_1[:, : self.ds]
        x_2 = sa_2[:, : self.ds]
        x = np.stack([x_1, x_2], axis=1)
        assert x.shape == (sa_1.shape[0], self.ds * 2)
        return x
