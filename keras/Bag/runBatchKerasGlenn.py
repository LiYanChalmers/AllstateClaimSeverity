#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 18:53:32 2016

@author: ly
"""

from subprocess import call

n_rounds = 100
for i in range(n_rounds):
    dst = 'BG'+str(i)+'.sh'
    call(['sbatch', dst])