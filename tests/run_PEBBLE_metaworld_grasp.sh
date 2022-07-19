for seed in 12345; do
    python train_PEBBLE.py hydra.exp_name=tests teacher=grasping env=metaworld_button-press-v2 seed=12345 agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=100 num_seed_steps=100 num_train_steps=501 eval_frequency=500 num_eval_episodes=1 agent.params.batch_size=51 double_q_critic.params.hidden_dim=256  double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 reward_update=1 num_interact=500 max_feedback=10000 reward_batch=50 feed_type=0 
done