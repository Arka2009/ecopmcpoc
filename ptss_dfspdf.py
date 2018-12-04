#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:52:57 2018

@author: amaity

Construct PDFs from
LACE-DFS benchmarks
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import special as sp
import pandas as pd

def get_rgb():
    """
        Generate random RGB value
    """
    r = np.random.uniform()
    g = np.random.uniform()
    b = np.random.uniform()
    return (r,g,b)

def get_pdf(plot=False):
    """
        Creates a mu and var for a
        normal approximation
    """
    dir2 = "/home/amaity/Dropbox/NUS-Datasets/ptss-poc/dfs_all_allocation"
    M    = 32
    mu  = []
    vr  = []

    for m2 in range(1,M+1):
        fl = dir2+"/dataset_ph5_alloc-"+str(m2)+".csv"
        df = pd.read_csv(fl)
        et = (df['time'].values)*1000

        tmu = np.mean(et)
        var = np.std(et)**2
        mu.append(tmu)
        vr.append(var)

        # Create Plots
        if plot:
            if m2 % 5 == 0:
                c    = get_rgb() 
                dist = stats.norm(loc=tmu,scale=np.sqrt(var))
                x    = np.linspace(dist.ppf(0.001),dist.ppf(0.999),1000)
                y    = dist.pdf(x)
                plt.plot(x,y,label="m=%d"%m2,color=c)
                plt.hist(et,bins=800,density=True,color=c)

    
    if plot:
        plt.legend()
        plt.xlabel("Execution Time")
        plt.title("DFS Lace PDFs")
        plt.savefig("generated-dist.pdf")
        plt.close()

    # print(list(range(1,M+1)))
    # print("\n")
    # print(mu)
    # print("\n")
    # print(vr)

    # Export to CSV file
    # f2  = open("mu_vars_dfs.csv","w")
    # f2.write("alloc,mu,var\n")
    # for m in range(1,M+1) :
    #     f2.write("%d,%f,%f\n"%(m,mu[m-1],vr[m-1]))
    # f2.close()

    # Draw The Speed-up curve and execution Time Characteristics
    # s = mu[0] / mu
    # plt.plot(range(1,M+1),s)
    # plt.title("Execution Time Speedup")
    # plt.xlabel("Allocation")
    # plt.ylabel("Speed-up")
    # plt.savefig("speed-up-dfs.pdf")
    # plt.close()

    # s2 = np.sqrt(vr) * 100 / mu
    # plt.plot(range(1,M+1),s2)
    # plt.title("Coeffecient of Varition of Execution Time")
    # plt.xlabel("Allocation")
    # plt.ylabel("Coeffecient of Varition (%)")
    # plt.savefig("coeffvar-dfs.pdf")
    # plt.close()

    return (mu,vr)

def gen_hist(numbins):
    """
        Generate a histogram (pdf) for
        a given dataset
    """
    dir2  = "/home/amaity/Dropbox/NUS-Datasets/ptss-poc/dfs_all_allocation"
    M     = 32
    dist  = []

    for m2 in range(1,M+1):
        fl = dir2+"/dataset_ph5_alloc-"+str(m2)+".csv"
        df = pd.read_csv(fl)
        et = (df['time'].values)*1000

        hist = np.histogram(et,bins=numbins)
        t3   = stats.rv_histogram(hist)
        dist.append(t3)
    
    return dist

def main_test2():
#    mu,var = get_pdf(False)
#    plt.plot(range(1,33),mu)
#    
#    print(mu)
#    print(var)
    # dista = gen_hist(100)
    # print(dista)
    get_pdf(plot=False)

if __name__=="__main__":
    main_test2()
    