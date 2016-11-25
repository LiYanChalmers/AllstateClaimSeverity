#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 22:52:10 2016

@author: multipos2
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 17:17:29 2016

@author: multipos2
without encoding 5-fold 10-CV, CV=?, LB=1118.00838
with encoding 5-fold 1-CV, CV=1155.527

max_depth=9, learning_rate=0.1, subsample=0.8, colsample_bytree=0.4, 
min_child_weight=1, alpha=1, gamma=1, CV=1139.1016
max_depth=9, learning_rate=0.1, subsample=0.6, colsample_bytree=0.4, 
min_child_weight=5, alpha=1, gamma=1, CV=1141.4135
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
                          seed=0, n_estimators=7500, base_score=7.8,
                          reg_alpha=1, gamma=1)

params = {}
params['max_depth'] = [9, 12, 15, 18, 21, 24]
#params['max_depth'] = [1, 2]
params['learning_rate'] = [0.01]
params['subsample'] = [0.4, 0.5, 0.6, 0.8]
params['colsample_bytree'] = [0.2, 0.4, 0.6, 0.8]
params['min_child_weight'] = [1, 3]
#params['base_score'] = [1, 2, 4, 8]
params['alpha'] = [1, 2]
params['gamma'] = [1]

y_test_pred_list, y_train_pred_list, mae_list, ntree_list, param_list = \
    xgb_randomcv(regxgb, params, x_train, y_train, x_test, n_iters=30, 
                 cv=5, random_state=0)
    
save_data('xgbRandomSearchCV2.pkl', 
          (y_test_pred_list, y_train_pred_list, mae_list, 
           ntree_list, param_list))