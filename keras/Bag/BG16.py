#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 13:46:39 2016

@author: ly
"""

from allstate1 import *



nbags = 1
nepochs = 100
random_state = 537
filename = 'BG16.pkl'
x_train, y_train, x_test, folds = process_data_keras(nfolds=3, nrows=1000, 
                                                     random_state=random_state)
pred_test, pred_oob = bag_predict_nn(nn_model, x_train, y_train, x_test, 
                                     folds, nbags, nepochs, random_state,
                                     patience=20, verbose=2)
save_data(filename, (pred_test, pred_oob))