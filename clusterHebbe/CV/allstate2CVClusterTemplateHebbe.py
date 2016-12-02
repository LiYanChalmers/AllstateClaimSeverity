# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 09:21:00 2016

@author: celin

Input files: 
    train_test_encoded.pkl
    parameterList.pkl
    train.csv
    test.csv
    sample_submission.csv (for submission only)
"""
from allstate1 import *

#%% baseline
train_test, x_train, y_train, x_test, cols, cols_cat, cols_num = \
    process_load()
train_test = read_data('train_test_encoded_xgb150_pairs35.pkl')
x_train = train_test.iloc[:x_train.shape[0],:]
x_test = train_test.iloc[x_train.shape[0]:,:]

del cols, cols_cat, cols_num, train_test
gc.collect()

regxgb = xgb.XGBRegressor(max_depth=12, learning_rate=0.01, 
                          objective=logregobj2, subsample=0.8,
                          colsample_bytree=0.2, min_child_weight=3,
                          seed=0, n_estimators=500000, base_score=7.8,
                          reg_alpha=1, gamma=1, nthread=-1)

y_test_pred, y_train_pred, mae, ntree = cv_predict_xgb(regxgb, x_train, y_train, 
                                                       x_test, cv=10, 
                                                       random_state=0, esr=300)
    
save_data('xgbCV0.pkl', (y_test_pred, y_train_pred, mae, ntree))