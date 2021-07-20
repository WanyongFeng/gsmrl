# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 20:35:09 2021

@author: wanyong
"""
import pickle
import numpy as np


file = 'C:/Users/wanyong/Desktop/Archive/exp/gas/ppo2_imputation/evaluate/test.pkl'
with open(file, 'rb') as f:
    data = pickle.load(f)

for i in data['metrics']:
    print(i)

# =============================================================================
# print(data['metrics']['episode_reward'])
# =============================================================================
print(data['metrics']['num_acquisition'])
# =============================================================================
# print(data['metrics']['episode_reward'])
# =============================================================================
# =============================================================================
# print(data['metrics']['reward_acflow'])
# =============================================================================
print(np.mean(data['metrics']['acc_acflow']))
print(np.mean(data['metrics']['acc_policy']))



# =============================================================================
# file = 'C:/Users/wanyong/Desktop/test/ppo_ori_cube/evaluate/test.pkl'
# with open(file, 'rb') as f:
#     data = pickle.load(f)
# 
# print(data['metrics']['episode_reward'])
# print(np.mean(data['metrics']['acc_acflow']))
# print(np.mean(data['metrics']['acc_policy']))
# =============================================================================

# =============================================================================
# a = np.array([2,3])
# m = np.array([[4,5,6,7],[8,9,10,11]])
# print(np.arange(len(a)), a)
# print(m[[0,1],[2,3]])
# print(np.all(m[np.arange(len(a)), a] == 0))
# =============================================================================
# =============================================================================
# np.all(old_m[np.arange(len(a)), a] == 0)
# =============================================================================
