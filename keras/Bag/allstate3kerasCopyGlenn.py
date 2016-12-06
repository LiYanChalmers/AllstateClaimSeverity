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

dirname=''
# .py files
n_rounds = 100
for i in range(n_rounds):
    src = os.path.join(dirname, 'allstate3kerasBagTemplate.py')
    dst = 'BG'+str(i)+'.py'
    dst = os.path.join(dirname, dst)
    
    newline14 = "random_state = "+str(np.random.randint(10000))+"\n"
    newline15 = "filename = 'BG"+str(i)+".pkl'\n"
    destination = open(dst, "w", newline='')
    source = open( src, "r" )
    for l, line in enumerate(source):
        if l==14:
            destination.write(newline14)
        elif l==15:
            destination.write(newline15)
        else:
            destination.write(line)

    source.close()
    destination.close()

# .sh files
for i in range(n_rounds):
    src = os.path.join(dirname, 'allstate3kerasBagTemplate.sh')
    dst = 'BG'+str(i)+'.sh'
    dst = os.path.join(dirname, dst)
    
    newline3 = "#SBATCH -J BG"+str(i)+" \n"
    newline6 = "#SBATCH -o BG"+str(i)+".stdout \n"
    newline7 = "#SBATCH -e BG"+str(i)+".stderr \n"
    newline15 = "pdcp BG"+str(i)+".py $TMPDIR\n"
    newline20 = "python BG"+str(i)+".py\n"
    destination = open(dst, "w", newline='')
    source = open( src, "r" )
    for l, line in enumerate(source):
        if l==3:
            destination.write(newline3)
        elif l==6:
            destination.write(newline6)
        elif l==7:
            destination.write(newline7)
        elif l==15:
            destination.write(newline15)
        elif l==20:
            destination.write(newline20)
        else:
            destination.write(line)

    source.close()
    destination.close()
    
copyfile('../../allstate1.py', './allstate1.py')