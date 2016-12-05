#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 18:43:15 2016

@author: ly
"""

import os
from shutil import copyfile

dirname=''
# .py files
rep = 1
runs = 50
n_rounds = int(round(50/rep))
for i in range(n_rounds):
    src = os.path.join(dirname, 'allstate3kerasTemplateGlenn.py')
    dst = 'allstate3kerasGlenn'+str(i)+'.py'
    dst = os.path.join(dirname, dst)
    
    newline22 = "                         "+\
        "nepochs=200, cv=5, rep="+str(rep)+\
        ", patience=5, random_state="+str(i)+") \n"
    newline24 = "save_data('KF"+str(i)+".pkl', \n"
    destination = open(dst, "w", newline='')
    source = open( src, "r" )
    for l, line in enumerate(source):
        if l==22:
            destination.write(newline22)
        elif l==24:
            destination.write(newline24)
        else:
            destination.write(line)

    source.close()
    destination.close()

# .sh files
for i in range(n_rounds):
    src = os.path.join(dirname, 'allstate3kerasTemplateGlenn.sh')
    dst = 'allstate3kerasGlenn'+str(i)+'.sh'
    dst = os.path.join(dirname, dst)
    
    newline3 = "#SBATCH -J KF"+str(i)+" \n"
    newline6 = "#SBATCH -o KF"+str(i)+".stdout \n"
    newline7 = "#SBATCH -e KF"+str(i)+".stderr \n"
    newline15 = "pdcp allstate3kerasGlenn"+str(i)+".py $TMPDIR\n"
    newline20 = "python allstate3kerasGlenn"+str(i)+".py\n"
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
    
copyfile('../allstate1.py', 'allstate1.py')
