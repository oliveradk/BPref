#!/bin/bash
python train_PrefPPO.py --env cartpole_balance --teachers cartpole_x_gaussian --seed 12345  --lr 0.0003 --batch-size 128 --n-envs 8 --ent-coef 0.0 --n-steps 50 --total-timesteps 9000 --num-layer 3 --hidden-dim 256 --clip-init 0.4 --gae-lambda 0.92  --re-feed-type 1 --re-num-interaction 150 --re-segment 25 --unsuper-step 8000 --unsuper-n-epochs 1 --re-max-feed 100 --re-batch 16 --re-update 5 --re_teacher-select max_beta