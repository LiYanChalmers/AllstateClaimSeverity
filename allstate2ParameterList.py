# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 23:19:26 2016

@author: li

Generate parameter list, code moved to allstate1.py
"""

from allstate1 import *

#%% baseline
train_test, x_train, y_train, x_test, cols, cols_cat, cols_num = \
    process_load()
train_test = comb_cat_feat_enc(100, 20)
#train_test = read_data('train_test_encoded_xgb10_pairs2.pkl')
x_train = train_test.iloc[:x_train.shape[0],:]
x_test = train_test.iloc[x_train.shape[0]:,:]

del cols, cols_cat, cols_num, train_test
gc.collect()

regxgb = xgb.XGBRegressor(max_depth=12, learning_rate=0.1, 
                          objective=logregobj2, subsample=0.8,
                          colsample_bytree=0.5, min_child_weight=1,
                          seed=0, n_estimators=75000, base_score=7.8,
                          reg_alpha=1, gamma=1)

params = {}
params['max_depth'] = [9, 12, 15, 18, 21, 24]
#params['max_depth'] = [1, 2]
params['learning_rate'] = [0.01]
params['subsample'] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
params['colsample_bytree'] = [0.2, 0.4, 0.6, 0.8]
params['min_child_weight'] = [1, 3, 5]
#params['base_score'] = [1, 2, 4, 8]
params['alpha'] = [1, 2, 5]
params['gamma'] = [1, 3, 5, 10]

parameter_list = list(model_selection.ParameterSampler(params, 100, 0))
save_data('parameterList.pkl', parameter_list)