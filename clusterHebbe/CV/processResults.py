# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 07:34:24 2016

@author: celin
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 14:21:55 2016

@author: celin
"""

import os
from shutil import copyfile
from allstate1 import *

def read_data(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        
    return data

dirname='results/'
cv_per_file = 1
n_rounds = int(round(100/cv_per_file))

y_test_pred_list = []
y_train_pred_list = []
mae_list = []
ntree_list = []
param_list = []
for i in range(n_rounds):
    filename = os.path.join(dirname, 'xgbHyperOpt'+str(i)+'.pkl')
    if not os.path.exists(filename):
        continue
    y_test_pred, y_train_pred, mae, ntree, param = \
        read_data(filename)
    y_test_pred_list.append(y_test_pred[0])
    y_train_pred_list.append(y_train_pred[0])
    mae_list.append(mae[0])
    ntree_list.append(ntree[0])
    param_list.append(param[0])
#    print('Import results', i)
    
mae_list = np.array(mae_list)
ntree_list = np.array(ntree_list)
mae_ave = np.mean(mae_list, axis=1)
ntree_ave = np.mean(ntree_list, axis=1)
mae_sortidx = np.argsort(mae_ave)
mae_sort = [mae_ave[i] for i in mae_sortidx]
ntree_sort = [ntree_ave[i] for i in mae_sortidx]
param_sort = [param_list[i] for i in mae_sortidx]
y_test_pred_sort = [y_test_pred_list[i] for i in mae_sortidx]
y_train_pred_sort = [y_train_pred_list[i] for i in mae_sortidx]
save_submission(invlogs(y_test_pred_sort[0]), 'HO_submission0.csv')
save_submission(invlogs(y_test_pred_sort[1]), 'HO_submission1.csv')
save_submission(invlogs(y_test_pred_sort[2]), 'HO_submission2.csv')
save_submission(invlogs(y_test_pred_sort[3]), 'HO_submission3.csv')