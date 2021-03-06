#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 13:46:39 2016

@author: ly
"""

from allstate1 import *



nbags = 10
nepochs = 200
random_state = 2732
filename = 'BG0.pkl'
x_train, y_train, x_test, folds = process_data_keras(nfolds=10, nrows=None, 
                                                     random_state=random_state)
pred_test, pred_oob = bag_predict_nn(nn_model, x_train, y_train, x_test, 
                                     folds, nbags, nepochs, random_state,
                                     patience=5, verbose=2)
save_data(filename, (pred_test, pred_oob))