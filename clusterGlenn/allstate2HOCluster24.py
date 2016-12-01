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

#%% feature encoding
train_test = read_data('train_test_encoded.pkl')
    
#%% parameter list
parameter_list = read_data('parameterList.pkl')

#%% baseline
train_test, x_train, y_train, x_test, cols, cols_cat, cols_num = \
    process_load()
#train_test = comb_cat_feat_enc(150, 35)
train_test = read_data('train_test_encoded_xgb150_pairs35.pkl')
x_train = train_test.iloc[:x_train.shape[0],:]
x_test = train_test.iloc[x_train.shape[0]:,:]

del cols, cols_cat, cols_num, train_test
gc.collect()

regxgb = xgb.XGBRegressor(max_depth=12, learning_rate=0.1, 
                          objective=logregobj2, subsample=0.8,
                          colsample_bytree=0.5, min_child_weight=1,
                          seed=0, n_estimators=50000, base_score=7.8,
                          reg_alpha=1, gamma=1)

param_list = read_data('parameterList.pkl')
params = param_list[24] 

y_test_pred_list, y_train_pred_list, mae_list, ntree_list, param_list = \
    xgb_gridcv(regxgb, params, x_train, y_train, x_test, cv=5, random_state=0)
    
save_data('xgbHyperOpt24.pkl', 
          (y_test_pred_list, y_train_pred_list, mae_list, 
           ntree_list, param_list))