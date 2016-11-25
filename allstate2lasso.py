#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 08:02:42 2016

@author: multipos2
"""

from allstate1 import *

#%% baseline
train_test, x_train, y_train, x_test, cols, cols_cat, cols_num = \
    process_load()
#train_test = comb_cat_feat_enc(100, 10)
train_test = read_data('train_test_encoded_xgb100_pairs10.pkl')
x_train = train_test.iloc[:x_train.shape[0],:]
x_test = train_test.iloc[x_train.shape[0]:,:]

del cols, cols_cat, cols_num, train_test
gc.collect()

lasso = linear_model.LassoCV()

lasso.fit(x_train, y_train)
