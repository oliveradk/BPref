import os

import pytest
import subprocess

PATH = os.path.abspath(os.path.dirname(__file__))
SCRIPT_DIR = "integ_scripts"


def run_script(script):
    path = os.path.join(PATH, SCRIPT_DIR, script)
    return subprocess.call(path)


def test_pebble_cartpole_gaussian_sim_querries():
    exit_code = run_script("run_PEBBLE_cartpole_gaussian_sim_querries.sh")
    assert exit_code == 0

def test_pebble_acrobot_gaussian_log_queries():
    exit_code = run_script("run_PEBBLE_acrobot_gaussian_log_queries.sh")
    assert exit_code == 0


def test_pebble_acrobot_gaussian_sim_dis_queries():
    exit_code = run_script("run_PEBBLE_acrobot_gaussian_sim_dis_queries.sh")
    assert exit_code == 0



def test_pebble_sweep_into_grasp_inplace_gaussian():
    exit_code = run_script("run_PEBBLE_sweep_into_grasp_inplace_gaussian.sh")
    assert exit_code == 0


def test_ppo_button_press_grasp_inplace_gaussian():
    exit_code = run_script("run_PrefPPO_button_press_grasp_inplace_gaussian.sh")
    assert exit_code == 0


def test_pebble_acrobot_ygaussian_max_beta():
    exit_code = run_script("run_PEBBLE_acrobot_y_gaussian_max_beta.sh")
    assert exit_code == 0


def test_ppo_cartpole_max_beta():
    exit_code = run_script("run_PrefPPO_cartpole_max_beta.sh")
    assert exit_code == 0
