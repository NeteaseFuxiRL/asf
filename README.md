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

<img src="data/logo.jpg" width=25% align="right" />

# Baselines

OpenAI Baselines is a set of high-quality implementations of reinforcement learning algorithms.

These algorithms will make it easier for the research community to replicate, refine, and identify new ideas, and will create good baselines to build research on top of. Our DQN implementation and its variants are roughly on par with the scores in published papers. We expect they will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. 

You can install it by typing:

```bash
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```

- [A2C](baselines/a2c)
- [ACER](baselines/acer)
- [ACKTR](baselines/acktr)
- [DDPG](baselines/ddpg)
- [DQN](baselines/deepq)
- [GAIL](baselines/gail)
- [PPO1](baselines/ppo1) (Multi-CPU using MPI)
- [PPO2](baselines/ppo2) (Optimized for GPU)
- [TRPO](baselines/trpo_mpi)

To cite this repository in publications:

    @misc{baselines,
      author = {Dhariwal, Prafulla and Hesse, Christopher and Klimov, Oleg and Nichol, Alex and Plappert, Matthias and Radford, Alec and Schulman, John and Sidor, Szymon and Wu, Yuhuai},
      title = {OpenAI Baselines},
      year = {2017},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/openai/baselines}},
    }
