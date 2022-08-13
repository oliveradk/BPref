from abc import ABC, abstractproperty

import numpy as np

from teacher import Teachers
from teacher.standard import StandardTeacher
from ..kernel import KernelBetaTeacher
import utils


class GaussianTeacher(KernelBetaTeacher):
    def __init__(
        self, ds, da, centroid, scale, weights, gamma, eps_mistake, eps_skip, eps_equal
    ):
        super().__init__(ds, da, gamma, eps_mistake, eps_skip, eps_equal)
        self.centroid = centroid
        self.scale = scale
        self.weights = weights
        self.kernel = self.compute_kernel()

    def compute_kernel(self):
        return lambda x: self.scale * np.exp(
            (-self.weights * np.square(self.centroid - x)).sum(axis=-1)
        )


class GaussianThreshTeacher(GaussianTeacher):
    def __init__(
        self,
        ds,
        da,
        centroid,
        scale,
        weights,
        thresh,
        thresh_val,
        gamma,
        eps_mistake,
        eps_skip,
        eps_equal,
    ):
        weight_scale = GaussianThreshTeacher.find_weight_scale(
            centroid, scale, thresh, thresh_val, weights
        )
        weights = weight_scale * weights
        super().__init__(
            ds,
            da,
            centroid,
            scale,
            weights,
            gamma,
            eps_mistake,
            eps_skip,
            eps_equal,
        )

    @staticmethod
    def find_weight_scale(centroid, scale, thresh, thresh_val, weights):
        return -np.log(thresh_val / scale) / ((centroid - thresh) ** 2 * weights).sum()


class Gaussian1DThreshTeachers(ABC, Teachers):
    def __init__(
        self,
        ds,
        da,
        n_experts,
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
        self.n_experts = n_experts
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

    @abstractproperty
    def bounds(self):
        raise NotImplementedError

    @abstractproperty
    def expert_type(self):
        raise NotImplementedError

    def define_teachers(self):
        # define experts
        experts = []
        int_size = (self.bounds[1] - self.bounds[0]) / self.n_experts
        cur_int = int_size / 2 + self.bounds[0]
        for i in range(self.n_experts):
            expert = self.expert_type(
                self.ds,
                self.da,
                centroid=np.ones(2) * cur_int,
                scale=self.scale[i],
                thresh=np.ones(2) * cur_int + int_size / 2,
                thresh_val=self.thresh_val[i],
                weights=np.ones(2),
                gamma=self.gamma[i],
                eps_mistake=self.eps_mistake[i],
                eps_skip=self.eps_skip[i],
                eps_equal=self.eps_equal[i],
            )
            experts.append(expert)
            cur_int += int_size

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
