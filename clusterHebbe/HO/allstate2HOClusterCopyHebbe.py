# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 14:21:55 2016

@author: celin
"""

import os
from shutil import copyfile

dirname=''
# .py files
cv_per_file = 1
n_rounds = int(round(100/cv_per_file))
for i in range(n_rounds):
    src = os.path.join(dirname, 'allstate2HOClusterTemplateHebbe.py')
    dst = 'allstate2HOCluster'+str(i)+'.py'
    dst = os.path.join(dirname, dst)
    
    newline39 = "params = param_list["+str(cv_per_file*i)+":"+\
                                     str(cv_per_file*(i+1))+"] \n"
    newline44 = "save_data('xgbHyperOpt"+str(i)+".pkl', \n"
    destination = open(dst, "w", newline='')
    source = open( src, "r" )
    for l, line in enumerate(source):
        if l==39:
            destination.write(newline39)
        elif l==44:
            destination.write(newline44)
        else:
            destination.write(line)

    source.close()
    destination.close()

# .sh files
for i in range(n_rounds):
    src = os.path.join(dirname, 'allstate2HOClusterTemplateHebbe.sh')
    dst = 'allstate2HOCluster'+str(i)+'.sh'
    dst = os.path.join(dirname, dst)
    
    newline3 = "#SBATCH -J HOT"+str(i)+" \n"
    newline7 = "#SBATCH -o HOT"+str(i)+".stdout \n"
    newline8 = "#SBATCH -e HOT"+str(i)+".stderr \n"
    newline17 = "pdcp allstate2HOCluster"+str(i)+".py $TMPDIR\n"
    newline25 = "python allstate2HOCluster"+str(i)+".py\n"
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
