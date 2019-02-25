# PPO2 with  Action-Specific Focuses

- Implementation of ppo with action-specific focuses in mujoco enviorments.
- `python -m baselines.ppo2.run_mujoco` runs the algorithm for 1M frames on a Mujoco environment, and the "attention" parameter represents the attention type.
    - Attention: action-specific focuses method
    - StateAttention：state attention method
    - NoAttention: pure PPO without attention

