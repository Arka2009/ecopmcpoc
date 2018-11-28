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

df = pd.read_csv("/home/amaity/Desktop/Nov28-presentation-images/core-requirement.csv")
x = df['DMR'].values
y = df['M'].values

plt.plot(x,y)
plt.title("Core Requirements for a given DMR (Sample=50)")
plt.xlabel("-log(DMR)")
plt.ylabel("Maximum Number of Cores")
plt.savefig("requirements.pdf")