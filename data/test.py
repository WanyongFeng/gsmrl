# -*- coding: utf-8 -*-
"""
Created on Wed May 26 11:20:17 2021

@author: wanyong
"""

# =============================================================================
# import tensorflow as tf
# =============================================================================

import numpy as np
import math
import random
import pickle
import logging

from pprint import pformat, pprint

import pandas as pd


# =============================================================================
# x = tf.placeholder(tf.float32)
# 
# y = x + x
# z = x * x
# 
# sess = tf.Session()
# 
# print(sess.run([y,z], {x: [[2,3,5,7],[3,4,5,6]]}))
# 
# sess.close()
# 
# =============================================================================
# =============================================================================
# def parse(x, d):
#     true_miss = np.array([])
#     for val in x:
#         if np.isnan(val):
#             true_miss = np.append(true_miss, 0.0)
#         else:
#             true_miss = np.append(true_miss, 1.0)
# 
#     true_miss = true_miss.astype(np.float32)
#     b = np.zeros([d], dtype=np.float32)
#     not_miss = []
#     for idx, val in enumerate(true_miss):
#         if val == 1:
#             not_miss.append(idx)
#     not_miss = np.array(not_miss)
#     no = np.random.choice(len(not_miss))
#     o = np.random.choice(len(not_miss), [no], replace=False)
#     ele = random.choice(not_miss)
#     while ele in not_miss[o]:
#         ele = random.choice(not_miss)
#     b[not_miss[o]] = 1.0
#     m = b.copy()
#     a = not_miss[o]
#     a = np.append(a, ele)
#     m[a] = 1.0
#     for i,val in enumerate(x):
#         if np.isnan(val):
#             x[i] = 0.0
#     x = x.astype(np.float32)
#     return x, b, m
# 
# 
# def _parse(x, d):
#     b = np.zeros([d], dtype=np.float32)
#     no = np.random.choice(d+1)
#     o = np.random.choice(d, [no], replace=False)
#     b[o] = 1.
#     m = b.copy()
#     w = list(np.where(b == 0)[0])
#     w.append(-1)
#     w = np.random.choice(w)
#     if w >= 0:
#         m[w] = 1.
# 
#     return x, b, m
# 
# x = np.array([1,2,float("NaN"),4,5])
# x = x.astype(np.float32)
# 
# x,b,m = _parse(x, 5)
# x,b,m = parse(x, 5)
# 
# =============================================================================


# =============================================================================
# df = pd.read_csv(r'C:/Users/wanyong/Desktop/calibratio2016.csv')
# vals = {}
# for col in df.columns:
#     if col != 'wave':
#         vals[col] = df[col].to_numpy()
# 
# df2 = pd.read_csv(r'C:/Users/wanyong/Desktop/collaboration_all_items_NA_to_zero_2016 (1) (2).csv')
# for col in df2.columns:
#     if '_IND_' in col:
#         temp_vals = df2[col].to_numpy()
#         col = col.split('_IND_')[0]
#         if col == '030EN' or col == '101AD' or col == '023ED' or col == '102EM' or col == '094HA': continue
#         vals[col] = np.concatenate((vals[col], temp_vals))
# 
# 
# delete = []
# for i in vals:
#     if len(vals[i]) == 566:
#        delete.append(i)
# 
# 
# for i in delete:
#     vals.pop(i, None)
#     
# over = []
# for i in vals:
#     if len(vals[i]) == 1262:
#         over.append(i)
#         
# for i in over:
#     vals[i] = vals[i][:914]
# 
# # =============================================================================
# # for i in vals:
# #     for j in vals[i]:
# #         print(j, type(j))
# # =============================================================================
#         
# # =============================================================================
# # num_na = []
# # for i in vals:
# #     count = 0
# #     for j in vals[i]:
# #         if np.isnan(j):
# #             count = count + 1
# #     if count == 309:
# #         print(i)
# # =============================================================================
# # =============================================================================
# #     num_na.append(count)
# # 
# # print(num_na)
# # print(np.amin(num_na))
# # print(np.where(num_na == np.amin(num_na))[0])
# # =============================================================================
# choice = [0.0, 1.0]
# 
# y = vals['013EN'].copy()
# 
# vals.pop('013EN', None)
# 
# for i, val in enumerate(y):
#     if np.isnan(val):
#         y[i] = np.random.choice(choice)
# 
# y = y.astype(np.float32)
# 
# values_2D =[]
# for i in vals:
#     values_2D.append(vals[i].copy())
# 
# values_2D = np.array(values_2D)
# 
# x = []
# for i in range(len(values_2D[0])):
#     x.append(values_2D[:,i])
# 
# for ex in x:
#     for i, val in enumerate(ex):
#         if not np.isnan(val):
#             if val == 0:
#                 ex[i] = 0 + random.uniform(0, 0.5)
#             if val == 1:
#                 ex[i] = 0.5 + random.uniform(0, 0.5)
#         else:
#             ex[i] = 0.0
# 
# x = np.array(x)
# x = x.astype(np.float32)
# 
# c = list(zip(x, y))
# 
# random.shuffle(c)
# 
# x, y = zip(*c)
# 
# x = np.array(x)
# 
# y = np.array(y)
# 
# 
# 
# 
# final = {}
# train = x[0:math.ceil(914 * 0.6)]
# valid = x[math.ceil(914 * 0.6):math.ceil(914 * 0.6) + math.ceil(914 * 0.2)]
# test = x[math.ceil(914 * 0.6) + math.ceil(914 * 0.2):]
# final['train'] = (train, y[0:math.ceil(914 * 0.6)])
# final['valid'] = (valid, y[math.ceil(914 * 0.6):math.ceil(914 * 0.6) + math.ceil(914 * 0.2)])
# final['test'] = (test, y[math.ceil(914 * 0.6) + math.ceil(914 * 0.2):])
# 
# 
# file = 'C:/Users/wanyong/Desktop/baseline/questions_shuffle.pkl'
# 
# with open(file, 'wb') as f:
#     pickle.dump(final, f)
# =============================================================================
# =============================================================================
# file1 = 'C:/Users/wanyong/Desktop/baseline/questions_shuffle.pkl'
# file2 = 'C:/Users/wanyong/Desktop/baseline/questions.pkl'
# 
# with open(file1, 'rb') as f1:
#     data1 = pickle.load(f1)
# 
# with open(file2, 'rb') as f2:
#     data2 = pickle.load(f2)
#     
# print(data1['train'][1])
# print(data2['train'][1])
# =============================================================================





# =============================================================================
# file = 'C:/Users/wanyong/Desktop/Archive/data/questions.pkl'
# 
# with open(file, 'rb') as f:
#     data = pickle.load(f)
# 
# x = data['train'][0]
# 
# print(x[5])
# 
# true_miss = np.array([])
# for val in x[5]:
#     if np.isnan(val):
#         true_miss = np.append(true_miss, 0.0)
#     else:
#         true_miss = np.append(true_miss, 1.0)
# 
# true_miss = true_miss.astype(np.float32)
# 
# b = np.zeros([len(x[5])], dtype=np.float32)
# not_miss = []
# for i, val in enumerate(true_miss):
#     if val == 1:
#         not_miss.append(i)
# not_miss = np.array(not_miss)
# print(not_miss)
# no = np.random.choice(len(not_miss))
# o = np.random.choice(len(not_miss), [no], replace=False)
# print(not_miss[o])
# ele = random.choice(not_miss)
# while ele in not_miss[o]:
#     ele = random.choice(not_miss)
# b[not_miss[o]] = 1
# m = b.copy()
# a = not_miss[o]
# a = np.append(a, ele)
# m[a] = 1
# print(b)
# print(m)
# =============================================================================
# =============================================================================
# b = np.zeros([len(x[5])], dtype=np.float32)
# no = np.random.choice(len(x[5]))
# o = np.random.choice(len(x[5]), [no], replace=False)
# b[o] = 1.
# m = b.copy()
# w = list(np.where(b == 0)[0])
# w = np.random.choice(w)
# m[w] = 1.
# b = b * true_miss
# m = m * true_miss
# =============================================================================








# =============================================================================
# print(x[5], b, m)    
# =============================================================================



# =============================================================================
# print(type(x))
# =============================================================================

    

        
    

dfile = 'C:/Users/wanyong/Desktop/cube_20_0.3.pkl'
with open(dfile, 'rb') as f:
            cube = pickle.load(f)
            
data, label = cube['valid']
print(data)

print(len(data))

# =============================================================================
# print(len(data))
# for i in data:
#     d = len(i)
#     no = np.random.choice(d)
#     o = np.random.choice(d, [no], replace=False)
#     i[o] = 0.
# 
# cube['train'] = (data, label)
# 
# data, label = cube['valid']
# for i in data:
#     d = len(i)
#     no = np.random.choice(d)
#     o = np.random.choice(d, [no], replace=False)
#     i[o] = 0.
# 
# print(len(data))
# cube['valid'] = (data, label)
# 
# data, label = cube['test']
# for i in data:
#     d = len(i)
#     no = np.random.choice(d)
#     o = np.random.choice(d, [no], replace=False)
#     i[o] = 0.
# 
# print(len(data))
# cube['test'] = (data, label)
# =============================================================================


# =============================================================================
# with open(dfile, 'wb') as f:
#     pickle.dump(cube, f)
# =============================================================================
    
   

# =============================================================================
# logging.basicConfig(filename= 'C:/Users/wanyong/Desktop/Archive/data/train.log',
#                     filemode='w',
#                     level=logging.INFO,
#                     format='%(message)s')
# 
# def _parse(i, x, y, d):
#     b = np.zeros([d], dtype=np.float32)
#     no = np.random.choice(d+1)
#     o = np.random.choice(d, [no], replace=False)
#     b[o] = 1.
#     m = b.copy()
#     w = list(np.where(b == 0)[0])
#     w.append(-1)
#     w = np.random.choice(w)
#     if w >= 0:
#         m[w] = 1.
# 
#     return i, x, y, b, m
# 
# 
# 
# dfile = 'C:/Users/wanyong/Desktop/Archive/data/questions.pkl'
# with open(dfile, 'rb') as f:
#             data = pickle.load(f)
# 
# data, label = data['train']
# size = data.shape[0]
# d = data.shape[1]
# num_batches = math.ceil(size / 256)
# ind = tf.range(size, dtype=tf.int64)
# dst = tf.data.Dataset.from_tensor_slices((ind, data, label))
# dst = dst.shuffle(size)
# dst = dst.map(lambda i, x, y: tuple(
#     tf.py_func(_parse, [i, x, y, d], 
#     [tf.int64, tf.float32, tf.float32, tf.float32, tf.float32])),
#     num_parallel_calls=16)
# tensor = dst.make_one_shot_iterator().get_next()
# with tf.Session() as sess:
#     for i in range(10):
#        value = sess.run(tensor)
#        logging.info(value)
# =============================================================================
# =============================================================================
# dst = dst.batch(256)
# dst = dst.prefetch(1)
# dst_it = dst.make_initializable_iterator()
# i, x, y, b,m = dst_it.get_next()
# dst_it = dst_it.initializer
# with tf.Session() as sess:
#     print(sess.run(dst_it))
# =============================================================================
# =============================================================================
# x = np.array([1,2,3,4,5])
# b = np.zeros([5], dtype=np.float32)
# no = np.random.choice(6)
# o = np.random.choice(5, [no], replace=False)
# b[o] = 1.
# b[4] = 0
# m = b.copy()
# w = list(np.where(b == 0)[0])
# w.append(-1)
# w = np.random.choice(w)
# if w >= 0:
#     m[w] = 1.
# 
# print(b,m)
# =============================================================================
# =============================================================================
# old = np.array([[1,0,0],[0,1,0]])
# new = np.array([[1,1,0],[0,1,1]])
# 
# diff = []
# for i, vals in enumerate(old):
#     for j, val in enumerate(vals):
#         if not new[i][j] == val:
#             diff.append(j)
#             
# print(diff)
#     
# x = [[3,0,0],[0,5,0]]
# 
# sam = [[[[80,25,0],[70,36,0]],[[60,40,50],[120,130,140]]]]
# 
# for i, val in enumerate(x):
#     if val[diff[i]] == 0:
#         idx = random.randint(0, 2)
#         val[diff[i]] = sam[0][i][idx][diff[i]]
# 
# print(x)
# =============================================================================
dfile = 'C:/Users/wanyong/Desktop/Archive/data/questions.pkl'
with open(dfile, 'rb') as f:
            data = pickle.load(f)
train_x, train_y = data['train']
valid_x, valid_y = data['valid']
train_x = np.concatenate((train_x, valid_x))
train_y = np.concatenate((train_y, valid_y))
data['train'] = (train_x, train_y)
data.pop('valid', None)
dfile = 'C:/Users/wanyong/Desktop/Archive/data/questions_ppo.pkl'
with open(dfile, 'wb') as f:
            pickle.dump(data, f)
