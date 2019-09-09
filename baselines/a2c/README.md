# A2C with  Action-Specific Focuses

- Implementation of A2C with action-specific focuses inAtari environments.
- For Atari environments: 
    - `python3 baselines/a2c/run_atari.py` runs the algorithm for 80M frames on an Atari environment, and the "param" parameter represents the attention type.
        - `python3 baselines/a2c/run_atari.py --policy=CnnAttention --param=action`: action-specific focuses method
        - `python3 baselines/a2c/run_atari.py  --policy=CnnAttention --param=state`: state attention method
        - `python3 baselines/a2c/run_atari.py  --policy=cnn`: pure PPO without attention