# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 22:27:12 2016

@author: celin
"""

from subprocess import call

cv_per_file = 1
n_rounds = int(round(100/cv_per_file))
for i in range(n_rounds):
    dst = 'allstate2CVCluster'+str(i)+'.sh'
    call(['sbatch', dst])