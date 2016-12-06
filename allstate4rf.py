#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 21:45:37 2016

@author: ly
"""

from allstate1 import *

train_test, x_train, y_train, x_test, cols, cols_cat, cols_num = \
    process_load()
train_test = read_data('train_test_encoded_xgb150_pairs35.pkl')
#x_train = train_test.iloc[:x_train.shape[0],:]
#x_test = train_test.iloc[x_train.shape[0]:,:]
#y_train = y_train.values.ravel()

x_train = train_test.iloc[:10000,:]
y_train = y_train[:10000].values.ravel()

rf = ensemble.RandomForestRegressor(criterion='mae', verbose=10, n_jobs=-1)
et = ensemble.ExtraTreesRegressor(criterion='mae', verbose=10, n_jobs=-1)

rf.fit(x_train, y_train)