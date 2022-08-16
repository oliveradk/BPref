python train_SAC.py env=cartpole_swingup hydra.exp_name=cartpole_trials seed=12345 agent.params.actor_lr=0.005 agent.params.critic_lr=0.005 num_train_steps=100000 num_unsup_steps=9000
