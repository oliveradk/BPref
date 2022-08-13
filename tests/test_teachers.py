import os

import pytest
from omegaconf import OmegaConf
import numpy as np
import pickle as pkl
import ipdb

import utils
from teacher.standard import StandardTeacher, StandardTeachers
from teacher.kernel.gaussian import (
    GaussianTeacher,
    GaussianThreshTeacher,
)
from teacher.acrobot import (
    AcrobotXUpperGaussian,
    AcrobotYUpperGaussian,
    AcrobotXYUpperGaussian,
    AcrobotXUpperGaussianTeachers,
    AcrobotYUpperGaussianTeachers,
)
from teacher.cartpole import CartpoleXGaussian, CartpoleXGaussianTeachers
from teacher.metaworld_reward import GraspInPlaceGaussian, GraspInPlaceTeachers

from tests import TEST_CONFIGS, TEST_OBJS, PATH


def get_config(yaml):
    return OmegaConf.load(os.path.join(PATH, TEST_CONFIGS, yaml))


def get_obj(obj_name):
    return pkl.load(open(os.path.join(PATH, TEST_OBJS, obj_name + ".pkl"), "rb"))


def get_query(env_name):
    sa_t_1 = get_obj(f"sa_t_1_{env_name}")
    sa_t_2 = get_obj(f"sa_t_2_{env_name}")
    r_t_1 = get_obj(f"r_t_1_{env_name}")
    r_t_2 = get_obj(f"r_t_2_{env_name}")
    info_t_1 = get_obj(f"info_t_1_{env_name}")
    info_t_2 = get_obj(f"info_t_2_{env_name}")
    return sa_t_1, sa_t_2, r_t_1, r_t_2, info_t_1, info_t_2


def thresh_gaus_teacher_test(teacher_cls, env, dim):
    teacher = teacher_cls(
        ds=env.observation_space.shape[0],
        da=env.action_space.shape[0],
        centroid=np.array([0, 0]),
        scale=5,
        weights=np.ones(2),
        thresh=np.ones(2),
        thresh_val=0.1,
        gamma=1,
        eps_mistake=0,
        eps_skip=0,
        eps_equal=0,
    )
    sa_1 = np.zeros((50, env.observation_space.shape[0]))
    sa_2 = np.zeros((50, env.observation_space.shape[0]))
    sa_1[:, dim] = 1
    sa_2[:, dim] = 1
    beta = teacher.get_beta(sa_1, sa_2, None, None)
    assert np.isclose(beta, 0.1)

    sa_1 = np.ones((50, env.observation_space.shape[0]))
    sa_2 = np.ones((50, env.observation_space.shape[0]))
    sa_1[:, dim] = 0
    sa_2[:, dim] = 0
    beta = teacher.get_beta(sa_1, sa_2, None, None)
    assert np.isclose(beta, 5)


def thresh_gaus_teachers_test(teacher_cls, env, centroids, edges, dim):
    teachers = teacher_cls(
        ds=env.observation_space.shape[0],
        da=env.observation_space.shape[0],
        n_experts=4,
        n_generalists=0,
        gamma=1,
        eps_mistake=0,
        eps_skip=0,
        eps_equal=0,
        beta_general=1,
        scale=4,
        thresh_val=0.1,
    )

    # kernel
    betas_cent = [
        teachers.teachers[i].kernel(np.array([centroids[i], centroids[i]]))
        for i in range(4)
    ]
    betas_edge = [
        teachers.teachers[i].kernel(np.array([edges[i], edges[i]])) for i in range(4)
    ]
    assert np.allclose(np.array(betas_cent), np.ones(4) * 4)
    assert np.allclose(np.array(betas_edge), np.ones(4) * 0.1)

    # betas
    sa_1 = np.ones((50, env.observation_space.shape[0]))
    sa_2 = np.ones((50, env.observation_space.shape[0]))
    sa_1[:, dim] = centroids[0]
    sa_2[:, dim] = centroids[0]
    beta = teachers.teachers[0].get_beta(sa_1, sa_2, None, None)
    assert np.isclose(beta, 4)

    sa_1 = np.ones((50, env.observation_space.shape[0]))
    sa_2 = np.ones((50, env.observation_space.shape[0]))
    sa_1[:, dim] = edges[-1]
    sa_2[:, dim] = edges[-1]
    beta = teachers.teachers[-1].get_beta(sa_1, sa_2, None, None)
    assert np.isclose(beta, 0.1)


class TestStandardTeachers:
    CFG = get_config("sweep_into.yaml")
    ENV = utils.make_metaworld_env(CFG)
    sa_t_1, sa_t_2, r_t_1, r_t_2, info_t_1, info_t_2 = get_query(
        "metaworld_sweep-into-v2"
    )

    def test_standard_teacher(self):
        teacher = StandardTeacher(
            beta=1,
            ds=self.ENV.observation_space.shape[0],
            da=self.ENV.action_space.shape[0],
            gamma=1,
            eps_mistake=0,
            eps_skip=0,
            eps_equal=0,
        )
        teacher.set_env(self.ENV)
        assert teacher.get_beta(None, None, None, None) == 1

    def test_standard_rational_teachers(self):
        teachers = StandardTeachers(
            beta=["inf", -10000, -200004],
            ds=self.ENV.observation_space.shape[0],
            da=self.ENV.action_space.shape[0],
            gamma=[1, 2, 3],
            eps_mistake=0,
            eps_skip=[0, 1, 0],
            eps_equal=0,
        )
        rat_idx = [5, 10, 15]
        teacher_ids = np.ones(50, dtype=np.int8)
        teacher_ids[rat_idx] = 0
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, teacher_ids, beta = teachers.get_labels(
            teacher_ids,
            self.sa_t_1,
            self.sa_t_2,
            self.r_t_1,
            self.r_t_2,
            self.info_t_1,
            self.info_t_2,
        )
        sum_r_t_1 = self.r_t_1.sum(axis=1)
        sum_r_t_2 = self.r_t_2.sum(axis=1)
        rat_labels = labels[rat_idx]
        perf_labels = np.argmax([sum_r_t_1, sum_r_t_2], axis=0)[rat_idx]
        assert np.array_equal(rat_labels, perf_labels.astype(rat_labels.dtype))

    def test_standard_teachers(self):
        teachers = StandardTeachers(
            beta=[1, 2, 3],
            ds=self.ENV.observation_space.shape[0],
            da=self.ENV.action_space.shape[0],
            gamma=[1, 2, 3],
            eps_mistake=0,
            eps_skip=[0, 1, 0],
            eps_equal=0,
        )
        teachers.set_env(self.ENV)
        betas = [
            teacher.get_beta(None, None, None, None) for teacher in teachers.teachers
        ]
        assert betas == [1, 2, 3]

        gammas = [teacher.gamma for teacher in teachers.teachers]
        assert gammas == [1, 2, 3]

        eps_skips = [teacher.eps_skip for teacher in teachers.teachers]
        assert eps_skips == [0, 1, 0]


class TestGaussianTeachers:
    def test_gaussian_teacher(self):
        # test kernel
        gaus_teacher = GaussianTeacher(
            ds=10,
            da=5,
            centroid=np.array([0, 0]),
            scale=5,
            weights=np.array([1, 2]),
            gamma=1,
            eps_mistake=0,
            eps_skip=0,
            eps_equal=0,
        )
        assert gaus_teacher.kernel(np.array([0, 0])) == 5

        # test vectorized kernel
        xs = np.random.rand(50, 2)
        y_vect = gaus_teacher.kernel(xs)
        y_iter = np.row_stack([gaus_teacher.kernel(x) for x in xs]).squeeze()
        assert np.allclose(y_vect, y_iter)

    def test_gaussian_thresh_teacher(self):
        gaus_thresh_teacher = GaussianThreshTeacher(
            ds=14,
            da=4,
            centroid=np.array([1, 2, 1]),
            scale=4,
            weights=np.ones(3),
            thresh=np.array([0, 0, 0]),
            thresh_val=0.2,
            gamma=1,
            eps_mistake=0,
            eps_skip=0,
            eps_equal=0,
        )
        assert np.isclose(gaus_thresh_teacher.kernel(np.array([1, 2, 1])), 4)
        assert np.isclose(gaus_thresh_teacher.kernel(np.array([0, 0, 0])), 0.2)


class TestAcrobotTeachers:
    CFG = get_config("acrobot.yaml")
    ENV = utils.make_env(CFG)

    sa_t_1, sa_t_2, r_t_1, r_t_2, info_t_1, info_t_2 = get_query("acrobot_swingup")

    def test_acrobot_x_upper_gaussian(self):
        thresh_gaus_teacher_test(AcrobotXUpperGaussian, env=self.ENV, dim=0)

    def test_acrobot_y_upper_gaussian(self):
        thresh_gaus_teacher_test(AcrobotYUpperGaussian, env=self.ENV, dim=2)

    def test_acrobot_xy_upper_gaussian(self):
        pass

    def test_acrobot_x_upper_teachers(self):
        thresh_gaus_teachers_test(
            AcrobotXUpperGaussianTeachers,
            self.ENV,
            centroids=[-0.75, -0.25, 0.25, 0.75],
            edges=[-1, -0.5, 0, 0.5],
            dim=0,
        )

    def test_acrobot_y_upper_teachers(self):
        thresh_gaus_teachers_test(
            AcrobotYUpperGaussianTeachers,
            self.ENV,
            centroids=[-0.75, -0.25, 0.25, 0.75],
            edges=[-1, -0.5, 0, 0.5],
            dim=2,
        )


class TestCartpoleTeachers:
    CFG = get_config("cartpole.yaml")
    ENV = utils.make_env(CFG)

    def test_cartpole_x_gaussian(self):
        thresh_gaus_teacher_test(CartpoleXGaussian, env=self.ENV, dim=0)

    def test_carpole_x_gaussian_teachers(self):
        thresh_gaus_teachers_test(
            CartpoleXGaussianTeachers,
            self.ENV,
            centroids=[-1.35, -0.45, 0.45, 1.35],
            edges=[-1.8, -0.9, 0, 0.9],
            dim=0,
        )


class TestMetaWorldRewardTeachers:
    CFG = get_config("button_press.yaml")
    ENV = utils.make_metaworld_env(CFG)

    sa_t_1, sa_t_2, r_t_1, r_t_2, info_t_1, info_t_2 = get_query(
        "metaworld_button-press-v2"
    )

    def test_grasp_in_place_gaussian(self):
        teacher = GraspInPlaceGaussian(
            ds=self.ENV.observation_space.shape[0],
            da=self.ENV.action_space.shape[0],
            centroid=np.ones(2) * 0.75,
            scale=1,
            weights=np.ones(2),
            thresh=np.ones(2),
            thresh_val=0.1,
            gamma=1,
            eps_mistake=0,
            eps_skip=0,
            eps_equal=0,
        )

        beta = teacher.get_beta(
            self.sa_t_1[0], self.sa_t_2[0], self.info_t_1[0], self.info_t_2[0]
        )
        assert beta.shape == ()

        info_1 = [{"grasp_reward": 0.3, "in_place_reward": 0.1} for _ in range(50)]
        info_2 = [{"grasp_reward": 0.3, "in_place_reward": 0.1} for _ in range(50)]
        beta = teacher.get_beta(self.sa_t_1[0], self.sa_t_2[0], info_1, info_2)
        assert np.isclose(beta, 1)

    def test_grasp_in_place_teachers(self):
        teachers = GraspInPlaceTeachers(
            ds=self.ENV.observation_space.shape[0],
            da=self.ENV.action_space.shape[0],
            n_experts=4,
            n_generalists=1,
            gamma=1,
            eps_mistake=0,
            eps_skip=0,
            eps_equal=0,
            beta_general=1,
            scale=2,
            thresh_val=0.5,
        )
        # teacher 1
        info_1 = [{"grasp_reward": 0.1, "in_place_reward": 0.7} for _ in range(50)]
        info_2 = [{"grasp_reward": 0.1, "in_place_reward": 0.7} for _ in range(50)]
        beta = teachers.teachers[0].get_beta(
            self.sa_t_1[0], self.sa_t_2[0], info_1, info_2
        )
        assert np.isclose(beta, 2)
        info_1 = [{"grasp_reward": 0, "in_place_reward": 0.7} for _ in range(50)]
        info_2 = [{"grasp_reward": 0, "in_place_reward": 0.7} for _ in range(50)]
        beta = teachers.teachers[0].get_beta(
            self.sa_t_1[0], self.sa_t_2[0], info_1, info_2
        )
        assert np.isclose(beta, 0.5)

        # teacher 4
        info_1 = [{"grasp_reward": 0.7, "in_place_reward": 0.1} for _ in range(50)]
        info_2 = [{"grasp_reward": 0.7, "in_place_reward": 0.1} for _ in range(50)]
        beta = teachers.teachers[3].get_beta(
            self.sa_t_1[0], self.sa_t_2[0], info_1, info_2
        )
        assert np.isclose(beta, 2)

        info_1 = [{"grasp_reward": 0.8, "in_place_reward": 0} for _ in range(50)]
        info_2 = [{"grasp_reward": 0.8, "in_place_reward": 0} for _ in range(50)]
        beta = teachers.teachers[3].get_beta(
            self.sa_t_1[0], self.sa_t_2[0], info_1, info_2
        )
        assert np.isclose(beta, 0.5)
