python train_PEBBLE.py hydra.exp_name=cartpole_trials teacher=cartpole_x_gaussian env=cartpole_balance seed=12345 teacher.params.scale=1 teacher.params.thresh_val=0.1 agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=250000 num_interact=5000 max_feedback=2000 reward_batch=50 reward_update=50 feed_type=1 teacher_selection=max_beta
