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
rep = 1
runs = 50
n_rounds = int(round(50/rep))
for i in range(n_rounds):
    src = os.path.join(dirname, 'kerasStarter2.py')
    dst = 'BG'+str(i)+'.py'
    dst = os.path.join(dirname, dst)
    
    newline16 = "np.random.seed("+str(np.random.randint(1000))+")\n"
    newline171 = "df.to_csv('bag_oob"+str(i)+".csv', index = False)\n"
    newline176 = "df.to_csv('bag_sub"+str(i)+".csv', index = False) \n"
    destination = open(dst, "w", newline='')
    source = open( src, "r" )
    for l, line in enumerate(source):
        if l==16:
            destination.write(newline16)
        elif l==171:
            destination.write(newline171)
        elif l==176:
            destination.write(newline176)
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
    
