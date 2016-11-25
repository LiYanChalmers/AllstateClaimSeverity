#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 21:13:14 2016

@author: multipos2
"""

import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
import pickle
import itertools
from sys import getsizeof
import gc

import xgboost as xgb
from sklearn import (metrics, cross_validation, model_selection, ensemble, 
                     linear_model)

SHIFT = 200 
SKEW_TH = 0.25
con = 0.7 # for fairobj

def logs(y):
    return np.log(y+SHIFT)
    
def invlogs(y):
    return np.exp(y)-SHIFT

def mae_xgb(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae_xgb', metrics.mean_absolute_error(invlogs(y), invlogs(yhat))
    
def mae_invlogs(ytrue, yhat):
    return metrics.mean_absolute_error(invlogs(ytrue), invlogs(yhat))
    
def xgb_randomcv(reg, params, x_train, y_train, x_test,
                 n_iters=10, cv=3, random_state=0):
    np.random.seed(random_state)
    seed1 = np.random.randint(10000)
    seed2 = np.random.randint(10000)    
    param_list = list(model_selection.ParameterSampler(params, n_iters, seed1))
    y_test_pred_list = []
    y_train_pred_list = []
    mae_list = []
    ntree_list = []
    for p in param_list:
        for k, v in p.items():
            setattr(reg, k, v)
        y_test_pred_, y_train_pred_, mae_, ntree_ = \
            cv_predict_xgb(reg, x_train, y_train, x_test, cv, seed2)
        y_test_pred_list.append(y_test_pred_)
        y_train_pred_list.append(y_train_pred_)
        mae_list.append(mae_)
        ntree_list.append(ntree_)
        
        
    return y_test_pred_list,y_train_pred_list,mae_list,ntree_list,param_list
    
def xgb_gridcv(reg, params, x_train, y_train, x_test, cv=3, random_state=0):
    np.random.seed(random_state)
    seed1 = np.random.randint(10000)
    seed2 = np.random.randint(10000)    
    param_list = list(model_selection.ParameterGrid(params))
    y_test_pred_list = []
    y_train_pred_list = []
    mae_list = []
    ntree_list = []
    for p in param_list:
        for k, v in p.items():
            setattr(reg, k, v)
        y_test_pred_, y_train_pred_, mae_, ntree_ = \
            cv_predict_xgb(reg, x_train, y_train, x_test, cv, seed2)
        y_test_pred_list.append(y_test_pred_)
        y_train_pred_list.append(y_train_pred_)
        mae_list.append(mae_)
        ntree_list.append(ntree_)
        
        
    return y_test_pred_list,y_train_pred_list,mae_list,ntree_list,param_list
    
def cv_predict_xgb(reg, x_train, y_train, x_test, cv=3, random_state=0, esr=300):
    kf = model_selection.KFold(n_splits=cv, shuffle=True, 
                               random_state=random_state)
    mae = []
    ntree = []
    y_test_pred = []
    y_train_pred = np.zeros((y_train.shape[0],))
    for train_index, test_index in kf.split(x_train):
        x_train1 = x_train.iloc[train_index]
        y_train1 = y_train.iloc[train_index]
        x_train2 = x_train.iloc[test_index]
        y_train2 = y_train.iloc[test_index]
        reg.fit(x_train1, y_train1, eval_metric=mae_xgb, verbose=True,
                eval_set=[(x_train1, y_train1), (x_train2, y_train2)], 
                early_stopping_rounds=esr)
        ntree.append(reg.best_ntree_limit)
#        reg.n_estimators = ntree[-1]
#        reg.fit(x_train1, y_train1)
        y_pred2 = reg.predict(x_train2)
        y_train_pred[test_index] = y_pred2
        mae.append(mae_invlogs(y_train2, y_pred2))
        y_test_pred.append(reg.predict(x_test))
        
    mae = np.array(mae)
    print('Mean: ', mae.mean(), ' std: ', mae.std())
    y_test_pred = np.mean(y_test_pred, axis=0)
    
    return y_test_pred, y_train_pred, mae, ntree
    
def cv_predict_xgb_repeat(reg, x_train, y_train, x_test, 
                          cv=3, random_state=0, rep=10):
    y_test_pred = []
    y_train_pred = []
    np.random.seed(random_state)
    
    for i in range(rep):
        tmp_test, tmp_train, _, _ = cv_predict_xgb(reg, x_train, y_train, 
                                                   x_test, cv, 
                                                   np.random.randint(1000))
        y_test_pred.append(tmp_test)
        y_train_pred.append(tmp_train)
        
    y_test_pred_mean = np.mean(y_test_pred, axis=0)
    y_train_pred_mean = np.mean(y_train_pred, axis=0)
    
    return y_test_pred_mean, y_train_pred_mean, y_test_pred, y_train_pred
    
def cv_predict(reg, x_train, y_train, x_test, cv=3, random_state=0):
    kf = model_selection.KFold(n_splits=cv, shuffle=True, 
                               random_state=random_state)
    mae = []
    y_test_pred = []
    y_train_pred = np.zeros((y_train.shape[0],))
    for train_index, test_index in kf.split(x_train):
        x_train1 = x_train.iloc[train_index]
        y_train1 = y_train.iloc[train_index]
        x_train2 = x_train.iloc[test_index]
        y_train2 = y_train.iloc[test_index]
        reg.fit(x_train1, y_train1)
        y_pred2 = reg.predict(x_train2)
        y_train_pred[test_index] = y_pred2
        mae.append(mae_invlogs(y_train2, y_pred2))
        y_test_pred.append(reg.predict(x_test))
        
    mae = np.array(mae)
    print('Mean: ', mae.mean(), ' std: ', mae.std())
    y_test_pred = np.mean(y_test_pred, axis=0)
    
    return y_test_pred, y_train_pred, mae
    
def cv_predict_repeat(reg, x_train, y_train, x_test, 
                      cv=3, random_state=0, rep=10):
    y_test_pred = []
    y_train_pred = []
    np.random.seed(random_state)
    
    for i in range(rep):
        tmp_test, tmp_train, _ = cv_predict(reg, x_train, y_train, x_test, 
                                            cv, np.random.randint(1000))
        y_test_pred.append(tmp_test)
        y_train_pred.append(tmp_train)
        
    y_test_pred_mean = np.mean(y_test_pred, axis=0)
    y_train_pred_mean = np.mean(y_train_pred, axis=0)
    
    return y_test_pred_mean, y_train_pred_mean, y_test_pred, y_train_pred
    
def logregobj(preds, dtrain):
    labels = dtrain.get_label()
#    con = 2
    x = preds-labels
    grad = con*x / (np.abs(x)+con)
    hess = con**2 / (np.abs(x)+con)**2
    return grad, hess
    
def logregobj2(ytrue, ypred):
#    con = 2
    x = ypred-ytrue
    grad = con*x / (np.abs(x)+con)
    hess = con**2 / (np.abs(x)+con)**2
    return grad, hess
    
def save_data(file_name, data):
    """File name must ends with .pkl
    """
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
def read_data(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        
    return data
    
def process_load():
    x_train = pd.read_csv('train.csv')
    y_train = pd.DataFrame({'loss':logs(x_train.loss.copy().values)})
    x_train.drop(['id', 'loss'], axis=1, inplace=True)    
    x_test = pd.read_csv('test.csv')
    x_test.drop('id', axis=1, inplace=True)
    train_test = pd.concat((x_train, x_test))

    cols = list(train_test.columns)
    cols_cat = [i for i in cols if i[:3]=='cat']
    cols_num = [i for i in cols if i[:4]=='cont']
    
    for c in cols_cat:
        train_test[c] = pd.factorize(train_test[c], sort=False)[0]
    
    # boxcox transformation for skewed features
    skewed_feats = [i for i in cols_num if sp.stats.skew(x_train[i])>SKEW_TH]
    for c in skewed_feats:
        train_test[c], lam = sp.stats.boxcox(train_test[c]+0.00001)

    x_train = train_test[:x_train.shape[0]]
    x_test = train_test[x_train.shape[0]:]

    return train_test, x_train, y_train, x_test, cols, cols_cat, cols_num
    
def comb_cat_feat(n_comb_feats):
    x_train = pd.read_csv('train.csv')
    y_train = pd.DataFrame({'loss':logs(x_train.loss.copy().values)})
    x_train.drop(['id', 'loss'], axis=1, inplace=True)    
    x_test = pd.read_csv('test.csv')
    x_test.drop('id', axis=1, inplace=True)
    train_test = pd.concat((x_train, x_test))
        
    selected_xgb,selected_forum,selected_combined, cat_importance = \
        read_data('selected_categorical_features.pkl')
        
    cols_num = [i for i in train_test.columns if 'cont' in i]
    cols_cat = [i for i in train_test.columns if 'cat' in i]
    feats = list(selected_combined.union(set(cols_num)))
    selected_forum = list(selected_forum)
    cat_feats = [i for i in cat_importance if i[0] in selected_forum]
    cat_feats = sorted(cat_feats, key=lambda x: x[1], reverse=True)
    cat_feats = [i[0] for i in cat_feats[:n_comb_feats]]
    selected_xgb = cat_importance[:50]
    selected_xgb = set([i[0] for i in selected_xgb])
    selected_combined = list(selected_xgb.union(set(selected_forum)))
    selected_combined.extend(cols_num)
    train_test = train_test[selected_combined]
        
    for c1, c2 in itertools.combinations(cat_feats, 2):
        col_name = c1+'_'+c2
        print(c1, c2)
        train_test[col_name] = train_test[c1]+train_test[c2]
        train_test[col_name] = pd.factorize(train_test[col_name], sort=True)[0]
        
    for c in train_test:
        if 'cat' in c:
            train_test[c] = pd.factorize(train_test[c], sort=True)[0]
    
    save_data('train_test_{}.pkl'.format(n_comb_feats), train_test)
    
def comb_cat_feat_enc(n_xgb_feats, n_comb_feats):
    '''From xgboost feature importance select the first n_xgb_feats features,
    combine them with the features in kaggle forum. Then select the first 
    n_comb_feats from the kaggle features to form 2nd order pairs.
    '''
    train_test = read_data('train_test_encoded.pkl')
    # figure out what features to use
    selected_xgb,selected_forum,selected_combined, cat_importance = \
        read_data('selected_categorical_features.pkl')
    cols_num = [i for i in train_test.columns if 'cont' in i]
    cols_cat = [i for i in train_test.columns if 'cat' in i]
    # features from kaggle forum
    selected_forum = list(selected_forum)
    # select the most important features from selected_forum
    cat_feats = [i for i in cat_importance if i[0] in selected_forum]
    cat_feats = sorted(cat_feats, key=lambda x: x[1], reverse=True)
    cat_feats = [i[0] for i in cat_feats[:n_comb_feats]]
    # select the most important features in xgboost
    selected_xgb = cat_importance[:n_xgb_feats]
    selected_xgb = set([i[0] for i in selected_xgb])
    # combine the two together
    selected_combined = list(selected_xgb.union(set(selected_forum)))
    selected_combined.extend(cols_num)
    # the final data
    train_test = train_test[selected_combined]

    # 2-column pairs
    for c1, c2 in itertools.combinations(cat_feats, 2):
        col_name = c1+'_'+c2
#        print(c1, c2)
        train_test[col_name] = 26**2*train_test[c1]+train_test[c2]

    save_data('train_test_encoded_xgb{}_pairs{}.pkl'.\
              format(n_xgb_feats, n_comb_feats), train_test)
    
    return train_test
    
def save_submission(y_pred, filename):
    df = pd.read_csv('sample_submission.csv')
    df['loss'] = y_pred
    df.to_csv(filename, index_label=False, index=False)
    
def encode(charcode):
    r = 0
    ln = len(str(charcode))
    for i in range(ln):
        r += (ord(str(charcode)[i]) - ord('A') + 1) * 26 ** (ln - i - 1)
    return r
    
if __name__=='__main__':
#%% load data and processing numeric features
    train_test, x_train, y_train, x_test, cols, cols_cat, cols_num = \
        process_load()
        
#%% feature importance
    xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.2,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.075,
    'objective': 'reg:linear',
    'max_depth': 6,
    'min_child_weight': 1,
    'booster': 'gbtree',
    }
    dtrain = xgb.DMatrix(x_train, label=y_train)
    regxgb = xgb.train(xgb_params, dtrain, 250, [(dtrain, 'train')], 
                       obj=logregobj, feval=mae_xgb, verbose_eval=True)
    feature_importance = list(regxgb.get_fscore().items())
    total_score = sum(regxgb.get_fscore().values())
    feature_importance = [(i[0], 1.0*i[1]/total_score) 
        for i in feature_importance]
    feature_importance = sorted(feature_importance, key=lambda x: x[1], 
                                reverse=True)
    cat_importance = [i for i in feature_importance if 'cat' in i[0]]
    selected_xgb = cat_importance[:35]
    selected_xgb = set([i[0] for i in selected_xgb])
    selected_forum = {'cat103', 'cat72', 'cat82', 'cat1', 'cat10', 'cat11',
                      'cat111', 'cat12', 'cat13', 'cat14', 'cat16', 'cat2',
                      'cat23', 'cat24', 'cat25', 'cat28', 'cat3', 'cat36',
                      'cat38', 'cat4', 'cat40', 'cat5', 'cat50', 'cat57',
                      'cat6', 'cat7', 'cat73', 'cat76', 'cat79', 'cat80',
                       'cat81', 'cat87', 'cat89', 'cat9', 'cat90'}
    selected_combined = selected_xgb.union(selected_forum)
    save_data('selected_categorical_features.pkl', 
              (selected_xgb,selected_forum,selected_combined, cat_importance))
    mae_sk = metrics.make_scorer(mae_invlogs, greater_is_better=False)
    
    
#%% models
#    regxgb = xgb.XGBRegressor(max_depth=6, learning_rate=0.075, 
#                              objective='reg:linear', subsample=0.7,
#                              colsample_bytree=0.2, min_child_weight=1,
#                              seed=0, n_estimators=350)
    #mae = cross_validation.cross_val_score(regxgb, x_train, y_train, 
    #                                       scoring=mae_sk, cv=5, verbose=10)
    
    #y_test_pred, y_train_pred, mae, ntree = \
    #    cv_predict_xgb(regxgb, x_train, y_train, cv=2, random_state=0)
    #print(mae.mean())
    
#    y_test_pred, y_train_pred = cv_predict_xgb_repeat(regxgb, x_train, y_train, 
#                                                      cv=5, random_state=0, rep=5)
#    print(mae_invlogs(y_train, y_train_pred))

#    regrf = ensemble.RandomForestRegressor(n_estimators=100, max_features=0.5,
#                                           max_depth=20, min_samples_leaf=2,
#                                           verbose=10, n_jobs=-1)
#    y_test_pred, y_train_pred, mae = cv_predict(regrf, x_train, y_train, cv=2, 
#                                                random_state=0)
#    y_test_pred, y_train_pred = cv_predict_repeat(regrf, x_train, y_train,  
#                                                  cv=2, random_state=0, 
#                                                  rep=10)
#    print(mae_invlogs(y_train, y_train_pred))
#    comb_cat_feat(5)


#%% test comb_cat_feat_enc(n_xgb_feats, n_comb_feats)
#    train_test = comb_cat_feat_enc(50, 35)

#%% feature encoding
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
    
#%% parameter list
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
