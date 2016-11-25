#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 13:23:02 2016

@author: multipos2

Combined features are better than original features only when logregobj2 is 
used as objective of xgboost.
"""

from allstate1 import *

#%% baseline
#train_test, x_train, y_train, x_test, cols, cols_cat, cols_num = \
#    process_load()
#regxgb = xgb.XGBRegressor(max_depth=6, learning_rate=0.075, 
#                          objective=logregobj2, subsample=0.7,
#                          colsample_bytree=0.2, min_child_weight=1,
#                          seed=0, n_estimators=750, base_score=7.8)
#
#y_test_pred, y_train_pred, mae1, ntree = \
#    cv_predict_xgb(regxgb, x_train, y_train, x_test, cv=2, random_state=0)
#print(mae1.mean())

#y_test_pred, y_train_pred = cv_predict_xgb_repeat(regxgb, x_train, y_train, 
#                                                  cv=5, random_state=0, rep=5)
#print(mae_invlogs(y_train, y_train_pred))    

#%%
#N = 10
#comb_cat_feat(N)
#train_test = read_data('train_test_{}.pkl'.format(N))
#x_train = train_test.iloc[:x_train.shape[0],:]
#x_test = train_test.iloc[x_train.shape[0]:,:]
#
#del cols, cols_cat, cols_num, train_test
#gc.collect()
#
#regxgb = xgb.XGBRegressor(max_depth=6, learning_rate=0.075, 
#                          objective=logregobj2, subsample=0.7,
#                          colsample_bytree=0.7, min_child_weight=1,
#                          seed=0, n_estimators=750, base_score=7.8)
#
#y_test_pred, y_train_pred, mae2, ntree = \
#    cv_predict_xgb(regxgb, x_train, y_train, x_test, cv=2, random_state=0)
#print(mae2.mean())

#%%
#train_test, x_train, y_train, x_test, cols, cols_cat, cols_num = \
#    process_load()
#N = 35
#comb_cat_feat(N)
#train_test = read_data('train_test_{}.pkl'.format(N))
#x_train = train_test.iloc[:x_train.shape[0],:]
#x_test = train_test.iloc[x_train.shape[0]:,:]
#
#del cols, cols_cat, cols_num, train_test
#gc.collect()
#
#xgb_params = {
#'seed': 0,
#'colsample_bytree': 0.7,
#'silent': 1,
#'subsample': 0.7,
#'learning_rate': 0.075,
#'max_depth': 6,
#'min_child_weight': 10,
#'booster': 'gbtree',
#'base_score': 7.8
#}
#dtrain = xgb.DMatrix(x_train, label=y_train)
#regxgb = xgb.train(xgb_params, dtrain, 400, [(dtrain, 'train')], 
#                   obj=logregobj, feval=mae_xgb, verbose_eval=True)
#feature_importance = list(regxgb.get_fscore().items())
#total_score = sum(regxgb.get_fscore().values())
#feature_importance = [(i[0], 1.0*i[1]/total_score) 
#    for i in feature_importance]
#feature_importance = sorted(feature_importance, key=lambda x: x[1], 
#                            reverse=True)
#feats = [i[0] for i in feature_importance[:230]]
#x_train = x_train[feats]
#x_test = x_test[feats]
#save_data('train_test_{}_selected.pkl'.format(N), (x_train, y_train, x_test))

#%% encoding 
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.drop(['id', 'loss'], axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)
train_test = pd.concat((train, test))
values = set()
cols = list(train.columns)
cols_cat = [i for i in cols if 'cat' in i]
for c in cols_cat:
    values = list(np.unique(train_test[c].values))
    to_replace = {v:encode(v) for v in values}
    train_test[c].replace(to_replace, inplace=True)
    print(c, list(np.unique(train_test[c].values)))
    
save_data('train_test_encoded.pkl', train_test)

#for c in cols_cat:
#    values = values.union(set(np.unique(train_test[c].values)))
#for c1, c2 in itertools.combinations(values, 2):
#    values = values.union(set((c1+c2, c2+c1)))
#    
#to_replace = {v:encode(v) for v in values}
#save('cat_encode2col.pkl', to_replace)
