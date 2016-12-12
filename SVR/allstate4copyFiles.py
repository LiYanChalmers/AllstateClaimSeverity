#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 18:43:15 2016

@author: ly
"""

import os
from shutil import copyfile
import numpy as np

np.random.seed(0)

dirname='hebbe'
# .py files
n_rounds = 400
for i in range(n_rounds):
    src = os.path.join('', 'allstate4svr0.py')
    dst = 'svr'+str(i)+'.py'
    dst = os.path.join(dirname, dst)
    
    newline37 = "idx = "+str(i)+"\n"
    destination = open(dst, "w", newline='')
    source = open( src, "r" )
    for l, line in enumerate(source):
        if l==37:
            destination.write(newline37)
        else:
            destination.write(line)

    source.close()
    destination.close()

# .sh files
for i in range(n_rounds):
    src = os.path.join('', 'allstate4bashTemplate.sh')
    dst = 'svr'+str(i)+'.sh'
    dst = os.path.join(dirname, dst)
    
    newline3 = "#SBATCH -J sv"+str(i)+" \n"
    newline8 = "#SBATCH -o sv"+str(i)+".stdout \n"
    newline9 = "#SBATCH -e sv"+str(i)+".stderr \n"
    newline17 = "pdcp svr"+str(i)+".py $TMPDIR\n"
    newline22 = "python svr"+str(i)+".py\n"
    destination = open(dst, "w", newline='')
    source = open( src, "r" )
    for l, line in enumerate(source):
        if l==3:
            destination.write(newline3)
        elif l==8:
            destination.write(newline8)
        elif l==9:
            destination.write(newline9)
        elif l==17:
            destination.write(newline17)
        elif l==22:
            destination.write(newline22)
        else:
            destination.write(line)

    source.close()
    destination.close()
    
copyfile('../allstate1.py', 'hebbe/allstate1.py')