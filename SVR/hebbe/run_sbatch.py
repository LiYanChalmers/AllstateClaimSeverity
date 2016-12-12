#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 03:00:49 2016

@author: ly
"""

from subprocess import call

n_rounds = 400
for i in range(n_rounds):
    dst = 'svr'+str(i)+'.sh'
    call(['sbatch', dst])