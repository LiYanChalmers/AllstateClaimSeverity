#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 20:45:24 2016

@author: ly
"""

from allstate1 import *
from functools import partial

pred_test0, pred_oob0 = read_data('BG0.pkl')
save_submission(invlogs(pred_test0), 'kerasBag0_submission.csv')
pred_test1, pred_oob1 = read_data('BG1.pkl')
save_submission(invlogs(pred_test1), 'kerasBag1_submission.csv')
pred_test = (pred_test0+pred_test1)/2.0
save_submission(invlogs(pred_test), 'kerasBag3_submission.csv')

test_pred_xgb, train_pred_xgb = read_data('xgbCV.pkl')

y_test_keras = np.array([pred_test0, pred_test1]).T
y_train_keras = np.array([pred_oob0, pred_oob1]).T
y_test_xgb = np.array(test_pred_xgb).T
y_train_xgb = np.array(train_pred_xgb).T
#
y_train_pred = np.hstack((y_train_keras, y_train_xgb))
y_test_pred = np.hstack((y_test_keras, y_test_xgb))
#
ndim = y_train_pred.shape[1]
w0 = 1.0/ndim*np.ones((ndim, ))

_, y_train, _, _ = process_data_keras(nfolds=5, nrows=None, random_state=0)
y_train = pd.DataFrame(y_train)

#w0 = np.ones((y_train_pred.shape[1], ))/y_train_pred.shape[1]
#obj_geo = partial(obj_opt_geo, y_true=y_train, y_pred=y_train_pred)
#y_test_geo, w_geo, res_geo = optimize_weights(obj_geo, w0, y_train_pred, y_test_pred, y_train)
#
#obj_lin = partial(obj_opt_lin, y_true=y_train, y_pred=y_train_pred)
#y_test_lin, w_lin, res_lin = optimize_weights(obj_lin, w0, y_train_pred, y_test_pred, y_train)

xgbreg = xgb.XGBRegressor(max_depth=6, min_child_weight=1, learning_rate=0.1,
                          n_estimators=2000, objective=logregobj2, 
                          base_score=7.8, reg_alpha=1, gamma=1, subsample=0.4,
                          colsample_bytree=0.5, )
#xgbreg.fit(y_train_pred, y_train), 
y_train_pred = pd.DataFrame(y_train_pred)
y_test_pred = pd.DataFrame(y_test_pred)
y_test_pred, y_train_pred, mae, ntree = \
    cv_predict_xgb(xgbreg, y_train_pred, y_train, y_test_pred, cv=5, 
                   random_state=0, esr=50)