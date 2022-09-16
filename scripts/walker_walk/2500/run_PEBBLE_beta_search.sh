for beta in 0.01, 0.03 0.1 0.3; do
    python train_PEBBLE.py  env=walker_walk seed=12345 teacher=standard teacher.params.beta=[$beta] agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=500000 num_interact=20000 max_feedback=1000 reward_batch=100 reward_update=50 feed_type=1
done