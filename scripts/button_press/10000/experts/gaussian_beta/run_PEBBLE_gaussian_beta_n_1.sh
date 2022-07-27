for seed in 45123 51234; do
    python train_PEBBLE.py hydra.exp_name=gaussian_beta teacher=gaussian_beta  env=metaworld_button-press-v2 seed=$seed teacher.params.n_teachers=1 agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=1000000 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3 reward_update=10  num_interact=5000 max_feedback=10000 reward_batch=50 reward_update=10 feed_type=$1
done