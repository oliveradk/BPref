for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 6789; do
    for seg in 25 50; do
        for env in 16 64; do
            python train_PrefPPO.py --env metaworld_button-press-v2 --seed $seed  --lr 0.0003 --batch-size 128 --n-envs $env --ent-coef 0.0 --n-steps 250 --total-timesteps 3000000 --num-layer 3 --hidden-dim 256 --clip-init 0.4 --gae-lambda 0.92  --re-feed-type 1 --re-num-interaction $1 --teacher-beta -1 --teacher-gamma 0.9 --teacher-eps-mistake 0 --teacher-eps-skip 0 --teacher-eps-equal 0 --re-segment $seg --unsuper-step 32000 --unsuper-n-epochs 50 --re-max-feed 20000 --re-batch $2 --re-update 50
        done
    done
done