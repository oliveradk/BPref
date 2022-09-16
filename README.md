# B-Pref

Official codebase for Deep Reinforcement from Expert Feedback, forked from [B-Pref: Benchmarking Preference-BasedReinforcement Learning](https://openreview.net/forum?id=ps95-mkHF_) contains scripts to reproduce experiments.


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
