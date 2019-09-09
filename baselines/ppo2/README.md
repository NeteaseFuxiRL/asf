# PPO2 with  Action-Specific Focuses

- Implementation of PPO with action-specific focuses in MuJoCo and Atari environments.
- For MuJoco environments:
    - `python3 -m baselines.ppo2.run_mujoco` runs the algorithm for 1M frames on a Mujoco environment, and the "attention" parameter represents the attention type.
        - `python3 -m baselines.ppo2.run_mujoco --attention=Attention`: action-specific focuses method
        - `python3 -m baselines.ppo2.run_mujoco --attention=StateAttention`: state attention method
        - `python3 -m baselines.ppo2.run_mujoco --attention=NoAttention`: pure PPO without attention
- For Atari environments: 
    - `python3 baselines/ppo2/run_atari.py` runs the algorithm for 80M frames on an Atari environment, and the "param" parameter represents the attention type.
        - `python3 baselines/ppo2/run_atari.py --policy=CnnAttention --param=action`: action-specific focuses method
        - `python3 baselines/ppo2/run_atari.py  --policy=CnnAttention --param=state`: state attention method
        - `python3 baselines/ppo2/run_atari.py  --policy=cnn`: pure PPO without attention
