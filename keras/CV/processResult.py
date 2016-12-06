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

#_, _, y_train, _, _, _, _ = process_load()
#if not os.path.exists('input_keras.pkl'):
#    x_train, id_train, _, x_test, id_test = process_data_keras()
#    save_data('input_keras.pkl', (x_train, id_train, y_train, 
#                                  x_test, id_test))
#else:
#    x_train, id_train, _, x_test, id_test = \
#        read_data('input_keras.pkl')
#        
#x_train = pd.read_csv('train.csv')
#x_train = x_train[['id', 'loss']]
#
#save_data('input.pkl', (x_train, id_train, y_train, id_test))

x_train, id_train, y_train, id_test = read_data('input.pkl')

dirname='results/'
rep = 1
runs = 50
n_rounds = int(round(50/rep))

id_test = list(id_test)
id_train = list(id_train)

y_test_keras = []
y_train_keras = []
mae_list = []
for i in range(n_rounds):
    filename = os.path.join(dirname, 'KF'+str(i)+'.pkl')
    if not os.path.exists(filename):
        print('No file')
        continue
    results = read_data(filename)
    y_test_pred, y_train_pred, _, _ = results
    
    y_test_pred = list(y_test_pred)
    y_test_pred = np.array([x for (y,x) in sorted(zip(id_test,y_test_pred))])
    
    y_train_pred = list(y_train_pred)
    y_train_pred = np.array([x for (y,x) in sorted(zip(id_train,y_train_pred))])

    y_test_keras.append(y_test_pred)
    y_train_keras.append(y_train_pred)
    mae_list.append(mae_invlogs(y_train, y_train_pred))
    
y_test_xgb, y_train_xgb = read_data('xgbCV.pkl')

y_test_keras = np.array(y_test_keras).T
y_train_keras = np.array(y_train_keras).T
y_test_xgb = np.array(y_test_xgb).T
y_train_xgb = np.array(y_train_xgb).T
#
y_train_pred = np.hstack((y_train_keras, y_train_xgb))
y_test_pred = np.hstack((y_test_keras, y_test_xgb))
#
ndim = y_train_pred.shape[1]
w0 = 1.0/ndim*np.ones((ndim, ))
                       
#obj_geo = partial(obj_opt_geo, y_true=y_train, y_pred=y_train_pred)
#y_test_geo, w_geo, res_geo = optimize_weights(obj_geo, w0, y_train_pred, y_test_pred, y_train)
#
#obj_lin = partial(obj_opt_lin, y_true=y_train, y_pred=y_train_pred)
#y_test_lin, w_lin, res_lin = optimize_weights(obj_lin, w0, y_train_pred, y_test_pred, y_train)

xgbreg = xgb.XGBRegressor(max_depth=4, min_child_weight=1, learning_rate=0.1,
                          n_estimators=2000, objective=logregobj2, 
                          base_score=7.8, reg_alpha=1, gamma=1, subsample=0.6,
                          colsample_bytree=0.8)
#xgbreg.fit(y_train_pred, y_train), 
y_train_pred = pd.DataFrame(y_train_pred)
y_test_pred = pd.DataFrame(y_test_pred)
y_test_pred, y_train_pred, mae, ntree = \
    cv_predict_xgb(xgbreg, y_train_pred, y_train, y_test_pred, cv=5, 
                   random_state=0, esr=50)






#
#
#save_submission(invlogs(y_test_geo), 'KF_XGB_submission.csv')

