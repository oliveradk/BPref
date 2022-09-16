for seed in 12345 23415 34512 45123 51234; do
    python train_PEBBLE.py env=cartpole_balance seed=$seed agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=250000 num_interact=5000 max_feedback=2500 reward_batch=50 reward_update=50 feed_type=1
done