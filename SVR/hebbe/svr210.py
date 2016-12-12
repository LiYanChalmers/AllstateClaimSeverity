#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 00:36:33 2016

@author: ly
"""

from allstate1 import *

start_time = time.time()

#x_train, y_train, x_test, _ = process_data_keras(nfolds=5, nrows=None, 
#                                                 random_state=0)
#save_data('input_svr.pkl', (x_train, y_train, x_test))

x_train, y_train, x_test = read_data('input_svr.pkl')


#N = 1000
#x_train = x_train[:N]
#y_train = y_train[:N]
#x_test = x_test[:N]
svr = SVR(C=1, epsilon=0.1, kernel='rbf', degree=3)

params = {}
params['C'] = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 8, 10, 15, 20]
params['epsilon'] = [0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2, 3]
params['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
params['degree'] = [1, 2, 3, 4, 5, 6]
n_iter = 400
param_samples = list(model_selection.ParameterSampler(params, n_iter=n_iter, 
                                                      random_state=0))
save_data('param_samples400.pkl', param_samples)

param_samples = read_data('param_samples400.pkl')

idx = 210
for k,v in param_samples[idx].items():
    print(k, v)
    setattr(svr, k, v)
y_test_pred, y_train_pred, mae = cv_predict_sparse(svr, x_train, y_train, 
                                                   x_test, cv=5, 
                                                   random_state=0)
run_time = time.time()-start_time

save_data('svr_sample{}.pkl'.format(idx), (y_test_pred, y_train_pred, mae, 
          param_samples[idx], run_time))