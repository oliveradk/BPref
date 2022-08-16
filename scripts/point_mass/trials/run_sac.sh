for lr in 1e-3 1e-4 1e-5; do
    python train_SAC.py env=point_mass_easy hydra.exp_name=point_mass_trials seed=12345 agent.params.actor_lr=$lr agent.params.critic_lr=$lr num_train_steps=100000 num_unsup_steps=9000
done