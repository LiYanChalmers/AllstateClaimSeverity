#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 16:14:03 2016

@author: ly
"""

from allstate1 import *

#%% load data
if not os.path.exists('input_keras.pkl'):
    x_train, id_train, y_train, x_test, id_test = process_data_keras()
    save_data('input_keras.pkl', (x_train, id_train, y_train, 
                                  x_test, id_test))
else:
    x_train, id_train, y_train, x_test, id_test = \
        read_data('input_keras.pkl')
        
#%%
y_test_pred_mean, y_train_pred_mean, y_test_pred, y_train_pred = \
    cv_predict_nn_repeat(nn_model, x_train, y_train, x_test, batch_size=128,
                         nepochs=200, cv=10, 
                         rep=1, patience=5, random_state=22) 

save_data('KF22.pkl', 
          (y_test_pred_mean, y_train_pred_mean, y_test_pred, y_train_pred ))