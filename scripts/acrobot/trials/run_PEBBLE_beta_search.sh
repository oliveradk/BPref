for beta in 0.01, 0.03, 0.1, 0.3, 1, 3; do
    python train_PEBBLE.py hydra.exp_name=acrobot_trials teacher.params.beta=[$beta] env=acrobot_swingup seed=12345 agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=250000 num_interact=5000 max_feedback=2000 reward_batch=50 reward_update=50 feed_type=1
done