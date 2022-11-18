# Deep Reinforcement Learning from Specialized Feedback

This is code for [The Expertise Problem: Learning from Specialized Feedback](https://arxiv.org/abs/2211.06519). If you use it, please cite:

Oliver Daniels-Koch, Rachel Freedman. “The Expertise Problem: Learning from Specialized Feedback.” In ML Safety Workshop at NeurIPS 2022.


## Install

```
conda env create -f conda_env.yml
pip install -e .[docs,tests,extra]
cd custom_dmcontrol
pip install -e .
cd custom_dmc2gym
pip install -e .
pip install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld
pip install pybullet
```

## Run experiments 

### Teacher Selection


Experiments can be reproduced with the following:
```
bash ./scripts/cartpole_balance/2500/teach_4/run_PEBBLE_uniform_teach_select.sh

bash ./scripts/cartpole_balance/2500/teach_4/run_PEBBLE_max_beta_teach_select.sh

bash ./scripts/walker_walk/5000/teach_4/run_PEBBLE_uniform_teach_select.sh

bash ./scripts/walker_walk/5000/teach_4/run_PEBBLE_max_beta_teach_select.sh
```

### Query Sampling

Experiments can be reproduced with the following:
```
bash ./scripts/cartpole_balance/2500/teach_4/run_PEBBLE_sim_queries.sh

bash ./scripts/cartpole_balance/2500/teach_4/run_PEBBLE_sim_dis_queries.sh

bash ./scripts/walker_walk/5000/teach_4/run_PEBBLE_sim_queries.sh

bash ./scripts/walker_walk/5000/teach_4/run_PEBBLE_sim_dis_queries.sh
```
