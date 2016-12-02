# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 14:21:55 2016

@author: celin
"""

import os
from shutil import copyfile
import numpy as np

np.random.seed(0)
dirname=''
# .py files
cv_per_file = 1
n_rounds = int(round(100/cv_per_file))
for i in range(n_rounds):
    src = os.path.join(dirname, 'allstate2CVClusterTemplateHebbe.py')
    dst = 'allstate2CVCluster'+str(i)+'.py'
    dst = os.path.join(dirname, dst)
    
    newline33 = \
        "                                                       "+\
        "random_state="+str(np.random.randint(10000))+", esr=300)\n"
    newline35 = "save_data('xgbCV"+str(i)+\
        ".pkl', (y_test_pred, y_train_pred, mae, ntree))"
    destination = open(dst, "w", newline='')
    source = open( src, "r" )
    for l, line in enumerate(source):
        if l==33:
            destination.write(newline33)
        elif l==35:
            destination.write(newline35)
        else:
            destination.write(line)

    source.close()
    destination.close()

# .sh files
for i in range(n_rounds):
    src = os.path.join(dirname, 'allstate2CVClusterTemplateHebbe.sh')
    dst = 'allstate2CVCluster'+str(i)+'.sh'
    dst = os.path.join(dirname, dst)
    
    newline3 = "#SBATCH -J CVC"+str(i)+" \n"
    newline7 = "#SBATCH -o CVC"+str(i)+".stdout \n"
    newline8 = "#SBATCH -e CVC"+str(i)+".stderr \n"
    newline17 = "pdcp allstate2CVCluster"+str(i)+".py $TMPDIR\n"
    newline25 = "python allstate2CVCluster"+str(i)+".py\n"
    destination = open(dst, "w", newline='')
    source = open( src, "r" )
    for l, line in enumerate(source):
        if l==3:
            destination.write(newline3)
        elif l==7:
            destination.write(newline7)
        elif l==8:
            destination.write(newline8)
        elif l==17:
            destination.write(newline17)
        elif l==25:
            destination.write(newline25)
        else:
            destination.write(line)

    source.close()
    destination.close()
    
copyfile('../../allstate1.py', 'allstate1.py')
#copyfile('../../selected_categorical_features.pkl', 
#         'selected_categorical_features.pkl')
#copyfile('../../train_test_encoded_xgb150_pairs35.pkl',
#         'train_test_encoded_xgb150_pairs35.pkl')
#copyfile('../../train_test_encoded.pkl', 'train_test_encoded.pkl')
