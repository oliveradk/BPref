python train_PEBBLE.py hydra.exp_name=cartpole_trials env=cartpole_balance seed=23451 teacher=cartpole_x_gaussian agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=500000 num_interact=5000 max_feedback=10000 reward_batch=50 reward_update=50 feed_type=7 state_mask=[0] teacher_selection=uniform topk=125
