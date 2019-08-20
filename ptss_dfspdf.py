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
from matplotlib import rc

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def get_rgb():
    """
        Generate random RGB value
    """
    r = np.random.uniform()
    g = r #np.random.uniform()
    b = r #np.random.uniform()
    rgb = rgb-0.3
    return (r,g,b)

def get_pdf(plot=False,write=True):
    """
        Creates a mu and var for a
        normal approximation
    """
    dir2 = "/home/amaity/Dropbox/NUS-Research/ptss_risk_model/ptss-poc/dfs_all_allocation_power"
    M    = 32
    mu  = []
    vr  = []
    ppgpower = []
    
    if write:
        fl2 = open(dir2+"/mu_vars_enrg.csv",mode='w')
        fl2.write("alloc,mu-time,var-time,mu-enrg,var-enrg,ppg-power\n")

    r = 0.01
    for m2 in range(1,M+1):
        fl = dir2+"/dataset_dfs_"+str(m2)+".csv"
        df = pd.read_csv(fl)
        et = (df['TIME'].values)
        ppg2 = (df['PKG-ENRG'].values)
        
        # Cleanup Values in MSR readings
        mdn = np.median(ppg2)
        #ppg = np.array(filter(lambda u : u > 0 and u < abs(1000*mdn), ppg2))
        ppg = [u for u in ppg2 if (u > 0 and u < abs(1000*mdn))]

        tmu = np.mean(et)
        var = np.std(et)**2
        tppgavg = np.mean(ppg)
        tppgvar = np.std(ppg)**2
        
        mu.append(tmu)
        vr.append(var)
        ppgpower.append(tppgavg*1000/tmu)

        # Create Plots
        if plot:
            if m2 % 10 == 0:
                c    = (r,r,r)
                r    = r + 0.20
                dist = stats.norm(loc=tmu,scale=np.sqrt(var))
                x    = np.linspace(dist.ppf(0.001),dist.ppf(0.999),1000)
                y    = dist.pdf(x)
                plt.plot(x,y,label="m=%d"%m2,color=c)
                plt.hist(et,bins=800,density=True,color=c)
        
        # Dump the values to a file
        if write:
            fl2.write("%d,%f,%f,%f,%f,%f\n"%(m2,tmu,var,tppgavg,tppgvar,tppgavg*1000/tmu))

    
    if plot:
        plt.legend()
        plt.xlabel("Execution Time (ms)")
        plt.ylabel("Probability Density")
        plt.title("Distribution of Execution Time of a phase")
        plt.savefig("generated-dist.pdf",bbox_inches='tight')
        plt.close()

    if write:
        fl2.close()
    
    # Plot The characteristics
    # plt.plot(range(1,M+1),ppgpower)
    # plt.xlabel("Allocation")
    # plt.ylabel("Package Power Consumed (Watts)")
    # plt.savefig(dir2+"/power.pdf")
    # plt.close()
    
    # plt.plot(range(1,M+1),mu)
    # plt.xlabel("Allocation")
    # plt.ylabel("Latency (ms)")
    # plt.savefig(dir2+"/latency.pdf")
    # plt.close()
    
    
    return (mu,vr,ppgpower)

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
    get_pdf(plot=True,write=False)

if __name__=="__main__":
    main_test2()
    