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
for i in range(n_rounds):
    filename = os.path.join(dirname, 'xgbCV'+str(i)+'.pkl')
    if not os.path.exists(filename):
        continue
    y_test_pred, y_train_pred, mae, ntree = \
        read_data(filename)
    y_test_pred_list.append(y_test_pred)
    y_train_pred_list.append(y_train_pred)
    mae_list.append(mae)
    ntree_list.append(ntree)
    
y_test_pred_mean = np.array(y_test_pred_list).mean(axis=0)
y_train_pred_mean = np.array(y_train_pred_list).mean(axis=0)

save_data('xgbCV.pkl', (y_test_pred_mean, y_train_pred_mean))

save_submission(invlogs(y_test_pred_mean), 'xgbCV_submission.csv')