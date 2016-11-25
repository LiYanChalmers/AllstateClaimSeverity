#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 17:17:29 2016

@author: multipos2
without encoding 5-fold 10-CV, CV=?, LB=1118.00838
with encoding 5-fold 10-CV, 100 xgb features, 15 choose 2 pairs, CV=1128.6921
    LB=1112.50116
"""

from allstate1 import *

#%% baseline
train_test, x_train, y_train, x_test, cols, cols_cat, cols_num = \
    process_load()
train_test = comb_cat_feat_enc(100, 15)
#train_test = read_data('train_test_encoded_xgb100_pairs15.pkl')
x_train = train_test.iloc[:x_train.shape[0],:]
x_test = train_test.iloc[x_train.shape[0]:,:]

del cols, cols_cat, cols_num, train_test
gc.collect()

n_cv = 5
n_rep = 10
regxgb = xgb.XGBRegressor(max_depth=12, learning_rate=0.05, 
                          objective=logregobj2, subsample=0.8,
                          colsample_bytree=0.5, min_child_weight=1,
                          seed=0, n_estimators=7500, base_score=7.8,
                          reg_alpha=1, gamma=1, reg_lambda=1)

y_test_pred, y_train_pred = \
    cv_predict_xgb_repeat(regxgb, x_train, y_train, x_test, cv=n_cv, 
                          random_state=0, rep=n_rep)
y_test_pred = invlogs(y_test_pred)
save_submission(y_test_pred, 'submission_xgbcvEnc1.csv')
save_data('xgbcvEnc1_fold{}_rep{}.pkl'.format(n_cv, n_rep), 
          (y_test_pred, y_train_pred))
