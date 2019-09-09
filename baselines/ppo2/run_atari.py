#!/usr/bin/env python3
import sys
import os
path = os.path.abspath(os.path.dirname('../../__file__'))
sys.path.append(os.path.abspath(os.path.dirname('.')))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, CnnAttentionPolicy
import multiprocessing
import tensorflow as tf
import datetime


def train(env_id, num_timesteps, seed, policy, param):

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    tf.Session(config=config).__enter__()
    # change parameter of env to start multi envs
    env = VecFrameStack(make_atari_env(env_id, 1, seed), 4)
    policy = {'cnn': CnnPolicy, 'lstm': LstmPolicy, 'lnlstm': LnLstmPolicy, 'CnnAttention': CnnAttentionPolicy}[policy]
    ppo2.learn(policy=policy, env=env, nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1), param=param)

def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'CnnAttention'], default='cnn')
    parser.add_argument('--param', help='parameters of policy', type=str, default='action')
    parser.add_argument('--nenv', help='num of env', type=int, default=8)
    args = parser.parse_args()
    path = "./trainlog/" + args.env + "/" + "seed_" + str(args.seed) + "_" + args.policy + "_" + args.param + "/" + args.env + "_" + datetime.datetime.now().strftime(
        "openai-%Y-%m-%d-%H-%M-%S-%f")
    logger.configure(path)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, param=args.param)

if __name__ == '__main__':
    main()

    # python3 baselines/ppo2/run_atari.py --policy=CnnAttention --param=action
