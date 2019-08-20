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


def main():
    fl2  = "locus-2dconv-dist-n8.csv"

    df2  = pd.read_csv(fl2)
    # df2  = pd.read_csv(fl2,header=0, names=['SAMPLE','TIME'])
    srs  = df2['latency(M)'].values
    plt.hist(srs,bins=800,density=True)
    plt.savefig('tmp.pdf')
    plt.xlabel('Execution Time')
    plt.ylabel('Probability Density')
    plt.title('Execution Time Variation of 2D Convolution')
    

if __name__=="__main__":
    main()