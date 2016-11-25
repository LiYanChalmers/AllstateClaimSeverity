# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 23:28:17 2016

@author: li
"""


import os

dirname=''
for i in range(10):
    src = os.path.join(dirname, 'allstate2HOtemplate.py')
    dst = 'allstate2HO'+str(i)+'.py'
    dst = os.path.join(dirname, dst)
    
    newline60 = "param_slice = param_list["+str(i*10)+":"+str((i+1)*10)+"] \n"
    newline65 = "save_data('xgbHyperOpt"+str(i)+".pkl', \n"
    destination = open( dst, "w" )
    source = open( src, "r" )
    for l, line in enumerate(source):
        if l!=60 and l!=65:
            destination.write(line)
        elif l==60:
            destination.write(newline60)
        elif l==65:
            destination.write(newline65)
    source.close()
    destination.close()

