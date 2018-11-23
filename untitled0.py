#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 13:19:12 2018

@author: amaity
"""
import numpy as np
A = np.array([[1,1,1],[100,1,1],[100,16,1]])
B = np.array([32,6400,400])
C = np.matmul(np.linalg.inv(A),B)
#print(C)
#sum = 0
#for c in C:
#    sum = sum + c
#print(sum)