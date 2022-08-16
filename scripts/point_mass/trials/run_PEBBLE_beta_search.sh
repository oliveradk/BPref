for beta in 1 0.3 3 0.1 10 0.03; do
    python train_PEBBLE.py hydra.exp_name=point_mass_trials teacher.params.beta=[$beta] env=point_mass_easy seed=12345 agent.params.actor_lr=0.0001 agent.params.critic_lr=0.0001 gradient_update=1 activation=tanh num_unsup_steps=18000 num_train_steps=1000000 num_interact=5000 max_feedback=20000 reward_batch=50 reward_update=50 feed_type=1
done