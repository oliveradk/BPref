import numpy as np
import pandas as pd
from .kernel.gaussian import GaussianThreshTeacher, Gaussian1DThreshTeachers
import utils


class GraspInPlaceGaussian(GaussianThreshTeacher):
    def get_point(self, sa_1, sa_2, info_1, info_2):
        keys = ["grasp_reward", "in_place_reward"]
        info_list_1 = utils.get_info_list(info_1, keys)
        info_list_2 = utils.get_info_list(info_2, keys)
        info_arrs_1 = {k: np.array(v) for k, v in info_list_1.items()}
        info_arrs_2 = {k: np.array(v) for k, v in info_list_2.items()}

        x_1 = info_arrs_1["grasp_reward"] / (
            info_arrs_1["grasp_reward"] + info_arrs_1["in_place_reward"]
        )
        x_2 = info_arrs_2["grasp_reward"] / (
            info_arrs_2["grasp_reward"] + info_arrs_2["in_place_reward"]
        )

        x = np.stack([x_1, x_2], axis=1)
        assert x.shape == (sa_1.shape[0], 2)
        return x


class GraspInPlaceTeachers(Gaussian1DThreshTeachers):
    @property
    def bounds(self):
        return [0, 1]

    @property
    def expert_type(self):
        return GraspInPlaceGaussian
