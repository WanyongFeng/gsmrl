import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import argparse
import logging
import tensorflow as tf
import numpy as np
import pickle
from pprint import pformat, pprint

from dfa_modules.ppo import PPOPolicy
from dfa_modules.cls_env import Env as ClsEnv
from dfa_modules.img_env import Env as ImgEnv
from dfa_modules.img_env_v2 import Env as ImgEnvV2
from dfa_modules.reg_env import Env as RegEnv
from dfa_modules.ts_env import Env as TSEnv
from utils.hparams import HParams

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--env', type=str, default='cls')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--hard', action='store_true')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--env_device', type=int, default=0)
parser.add_argument('--agent_device', type=int, default=1)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()
params = HParams(args.cfg_file)

################################################################
logging.basicConfig(filename=params.exp_dir + f'/{args.mode}.log',
                    filemode='w',
                    level=logging.DEBUG if args.debug else logging.INFO,
                    format='%(asctime)s %(message)s')
logging.info(pformat(params.dict))

np.random.seed(args.seed)
tf.set_random_seed(args.seed)

# Env class
env_dict = {
    'cls': ClsEnv,
    'img': ImgEnv,
    'img_v2': ImgEnvV2,
    'reg': RegEnv,
    'ts': TSEnv
}
Env = env_dict[args.env]

# train
with tf.device(f"/gpu:{args.env_device}"):
    env = Env(params, 'train')

with tf.device(f"/gpu:{args.agent_device}"):
    agent = PPOPolicy(params, env, 'train')

if args.mode == 'resume':
    agent.load()
    agent.train()
if args.mode == 'train':
    agent.train()
train_dict = agent.evaluate(hard=args.hard, max_batches=30)

# test
with tf.device(f"/gpu:{args.env_device}"):
    env = Env(params, 'test')

with tf.device(f"/gpu:{args.agent_device}"):
    agent = PPOPolicy(params, env, 'test')

test_dict = agent.evaluate(hard=args.hard, max_batches=30)

print("train_acflow: " + str(np.mean(train_dict['metrics']['acc_acflow'])))
print("train_policy: " + str(np.mean(train_dict['metrics']['acc_policy'])))
print("test_acflow: " + str(np.mean(test_dict['metrics']['acc_acflow'])))
print("test_policy: " + str(np.mean(test_dict['metrics']['acc_policy'])))

# save
os.makedirs(f'{params.exp_dir}/evaluate', exist_ok=True)
with open(f'{params.exp_dir}/evaluate/train.pkl', 'wb') as f:
    pickle.dump(train_dict, f)
with open(f'{params.exp_dir}/evaluate/test.pkl', 'wb') as f:
    pickle.dump(test_dict, f)
