#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 06:20:55 2016

@author: ly
"""

import os
from shutil import copyfile
from allstate1 import *
from functools import partial

_, _, y_train, _, _, _, _ = process_load()
if not os.path.exists('input_keras.pkl'):
    x_train, id_train, _, x_test, id_test = process_data_keras()
    save_data('input_keras.pkl', (x_train, id_train, y_train, 
                                  x_test, id_test))
else:
    x_train, id_train, _, x_test, id_test = \
        read_data('input_keras.pkl')
        
x_train = pd.read_csv('train.csv')
x_train = x_train[['id', 'loss']]

dirname='results/'
rep = 1
runs = 50
n_rounds = int(round(50/rep))

y_test_pred_list = []
y_train_pred_list = []
mae_list = []
for i in range(n_rounds):
    filename = os.path.join(dirname, 'KF'+str(i)+'.pkl')
    if not os.path.exists(filename):
        print('No file')
        continue
    results = read_data(filename)
    y_test_pred_mean, y_train_pred_mean, y_test_pred, y_train_pred  = results
    y_test_pred_list.append(y_test_pred_mean)
    y_train_pred_list.append(y_train_pred_mean)
    mae_list.append(mae_invlogs(y_train, y_train_pred_mean))
    
y_test_keras = np.array(y_test_pred_list).mean(axis=0)
y_train_keras = np.array(y_train_pred_list).mean(axis=0)
    
mae_train_mean = mae_invlogs(y_train, y_train_pred_mean)

y_test_xgb, y_train_xgb = read_data('xgbCV.pkl')

id_train = list(id_train)
y_train_keras = list(y_train_keras)
y_train_keras = np.array([x for (y,x) in sorted(zip(id_train,y_train_keras))])

id_test = list(id_test)
y_test_keras = list(y_test_keras)
y_test_keras = np.array([x for (y,x) in sorted(zip(id_test,y_test_keras))])

print(mae_invlogs(y_train, y_train_xgb))
print(mae_invlogs(y_train, y_train_keras))

y_train_pred = np.array([y_train_keras, y_train_xgb]).T
y_test_pred = np.array([y_test_keras, y_test_xgb]).T

obj_geo = partial(obj_opt_geo, y_true=y_train, y_pred=y_train_pred)
y_test_geo, w_geo, res_geo = optimize_weights(obj_geo, y_train_pred, y_test_pred, y_train)

obj_lin = partial(obj_opt_lin, y_true=y_train, y_pred=y_train_pred)
y_test_lin, w_lin, res_lin = optimize_weights(obj_lin, y_train_pred, y_test_pred, y_train)


save_submission(invlogs(y_test_geo), 'KF_XGB_submission.csv')