# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 14:33:39 2021

@author: wanyong
"""
import numpy as np
import pickle


dfile = 'C:/Users/wanyong/Desktop/Archive/data/cube_20_0.3_ori.pkl'
with open(dfile, 'rb') as f:
            data = pickle.load(f)
train_x, train_y = data['train']
valid_x, valid_y = data['valid']
train_x = np.concatenate((train_x, valid_x))
train_x = train_x.flatten()
indices = np.random.choice(np.arange(train_x.size), replace=False, size=int(train_x.size * 0.5))
train_x[indices] = 0
train_x = train_x.reshape([15000, 20])
train_y = np.concatenate((train_y, valid_y))
data['train'] = (train_x, train_y)
data.pop('valid', None)

test_x, test_y = data['test']
test_x = test_x.flatten()
indices = np.random.choice(np.arange(test_x.size), replace=False, size=int(test_x.size * 0.5))
test_x[indices] = 0
test_x = test_x.reshape([5000, 20])
data['test'] = (test_x, test_y)

dfile = 'C:/Users/wanyong/Desktop/Archive/data/cube50.pkl'
with open(dfile, 'wb') as f:
            pickle.dump(data, f)



