#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:08:23 2016

@author: ly
"""

import os
from shutil import copyfile
from allstate1 import *


dirname='results/'
n_rounds = 400

y_test_pred_list = []
y_train_pred_list = []
mae_list = []
params_list = []
runtime_list = []
for i in range(n_rounds):
    filename = os.path.join(dirname, 'xgbCV'+str(i)+'.pkl')
    if not os.path.exists(filename):
        continue
    y_test_pred, y_train_pred, mae, ntree = \
        read_data(filename)
    y_test_pred_list.append(y_test_pred)
    y_train_pred_list.append(y_train_pred)
    mae_list.append(mae)
    ntree_list.append(ntree)
    
y_test_pred_mean = np.array(y_test_pred_list).mean(axis=0)
y_train_pred_mean = np.array(y_train_pred_list).mean(axis=0)

save_data('xgbCV.pkl', (y_test_pred_list, y_train_pred_list))

save_submission(invlogs(y_test_pred_mean), 'xgbCV_submission.csv')