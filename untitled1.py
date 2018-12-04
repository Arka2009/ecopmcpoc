#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:56:23 2018

@author: amaity
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import ptss_utils as ptsl
import ptss_synthpdf as psyn
import heapq
import queue
import timeit
import pandas as pd

# df = pd.read_csv("/home/amaity/Desktop/Nov28-presentation-images/core-requirement.csv")
# x = df['DMR'].values
# y = df['M'].values

# plt.plot(x,y)
# plt.title("Core Requirements for a given DMR (Sample=50)")
# plt.xlabel("-log(DMR)")
# plt.ylabel("Maximum Number of Cores")
# plt.savefig("requirements.pdf")

#print(ptsl.close(0.23,0.18,tol=0.3))
mu  = 66.454092
var = 0.098442

dist = stats.norm(loc=mu,scale=np.sqrt(var))
isf  = 66.665716
sf   = dist.sf(isf)

# Draw 1000 samples
N = 100
sam = dist.rvs(N)
n  = 0
n2 = 0
n3 = 0

# case 1
for el in sam:
    if el > isf :
        n = n+1

# case 2
for iter in range(len(sam)):
    dist = stats.norm(loc=mu,scale=np.sqrt(var))
    el   = dist.rvs(1)[0]
    if el > isf:
        n2 = n2 + 1

# case 3
df    = pd.read_csv("et.csv")
sam2  = df['time'].values
for el in sam2:
    if el > isf:
        n3 = n3 + 1

#plt.hist(sam2)
print(sam2)
print(np.mean(sam2))
print(np.std(sam2)**2)
frac  = n/N
frac2 = n2/N 
frac3 = n3/len(sam2)
print(frac)
print(frac2)
print(frac3)