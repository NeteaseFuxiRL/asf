#!/usr/bin/env python3
import datetime
import os
import sys

import multiprocessing

path = os.path.abspath(os.path.dirname('../../__file__'))
sys.path.append(os.path.abspath(os.path.dirname('.')))
print(sys.path)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.a2c.a2c import learn
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, CnnAttentionPolicy
import tensorflow as tf


def train(env_id, num_timesteps, seed, policy, lrschedule, num_env, param):
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    elif policy== "CnnAttention":
        policy_fn = CnnAttentionPolicy
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True  # pylint: disable=E1101
    tf.Session(config=config).__enter__()
    # change parameter of env to start multi envs
    env = VecFrameStack(make_atari_env(env_id, num_env, seed), 4)
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule, param=param, nsteps=16)
    env.close()

def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'CnnAttention'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--param', help='parameters of policy', type=str, default='action')
    parser.add_argument('--nenv', help='num of env', type=int, default=16)
    args = parser.parse_args()
    print(args.env)
    path = "./trainlog/" + args.env + "/" + "seed_" + str(args.seed) + "_" + args.policy + "_" + args.param + "/" + args.env + "_" + datetime.datetime.now().strftime(
        "openai-%Y-%m-%d-%H-%M-%S-%f")
    logger.configure(path)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, policy=args.policy, lrschedule=args.lrschedule, num_env=args.nenv, param=args.param)

    # python3 baselines/a2c/run_atari.py --policy=CnnAttention --param=action

if __name__ == '__main__':
    main()
