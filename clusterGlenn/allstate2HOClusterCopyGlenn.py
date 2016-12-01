# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 14:21:55 2016

@author: celin
"""

import os
from shutil import copyfile

dirname=''
# .py files
for i in range(100):
    src = os.path.join(dirname, 'allstate2HOClusterTemplateGlenn.py')
    dst = 'allstate2HOCluster'+str(i)+'.py'
    dst = os.path.join(dirname, dst)
    
    newline39 = "params = param_list["+str(i)+"] \n"
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
for i in range(100):
    src = os.path.join(dirname, 'allstate2HOClusterTemplateGlenn.sh')
    dst = 'allstate2HOCluster'+str(i)+'.sh'
    dst = os.path.join(dirname, dst)
    
    newline3 = "#SBATCH -J HOT"+str(i)+" \n"
    newline6 = "#SBATCH -o HOT"+str(i)+".stdout \n"
    newline7 = "#SBATCH -e HOT"+str(i)+".stderr \n"
    newline16 = "pdcp allstate2HOCluster"+str(i)+".py $TMPDIR\n"
    newline24 = "python allstate2HOCluster"+str(i)+".py\n"
    destination = open(dst, "w", newline='')
    source = open( src, "r" )
    for l, line in enumerate(source):
        if l==3:
            destination.write(newline3)
        elif l==6:
            destination.write(newline6)
        elif l==7:
            destination.write(newline7)
        elif l==16:
            destination.write(newline16)
        elif l==24:
            destination.write(newline24)
        else:
            destination.write(line)

    source.close()
    destination.close()
    
copyfile('../allstate1.py', 'allstate1.py')