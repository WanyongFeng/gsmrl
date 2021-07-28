# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 13:58:35 2021

@author: wanyong
"""
import pickle 
import tensorflow as tf
import numpy as np
from models import get_model

dfile = 'C:/Users/wanyong/Desktop/gsmrl_imputation/data/cube5.pkl'
split = 'train'
with open(dfile, 'rb') as f:
            data_dict = pickle.load(f)
data, label = data_dict[split]
b = np.zeros((data.shape[0],data.shape[1]))
for i, vals in enumerate(data):
    for j, val in enumerate(vals):
        if not val == 0:
            b[i][j] = 1
            
g = tf.Graph()
with g.as_default():
    # open a session
    config = tf.ConfigProto()
    config.log_device_placement = True
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config, graph=g)
    # build ACFlow model
    model_hps = HParams(f'{hps.model_dir}/params.json')
    self.model = get_model(self.sess, model_hps)
    # restore weights
    self.saver = tf.train.Saver()
    restore_from = f'{hps.model_dir}/weights/params.ckpt'
    logger.info(f'restore from {restore_from}')
    self.saver.restore(self.sess, restore_from)
    # build dataset
    self.dataset = Dataset(hps.dfile, split, hps.episode_workers)
    self.dataset.initialize(self.sess)