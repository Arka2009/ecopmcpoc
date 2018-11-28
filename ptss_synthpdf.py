#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:52:57 2018

@author: amaity

Generate a synthetic
PDF distribution that
is dependent upon the number
of core allocations,

you might ignore the workload
for now
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import special as sp

def construct_pdf(D,M,var,plot=False):
    """
        Creates a mu and var for a
        normal distribution
    """
    synth_risk = np.linspace(1e-5,0.999999,M)
    mu  = []
    vr  = []
    m   = M
    for s in synth_risk:
        tmu = estimate_mu(D,s,var)
        mu.append(tmu)
        vr.append(var)

        # Create Plots
        if plot:
            if m % 16 == 0:
                dist = stats.norm(loc=tmu,scale=np.sqrt(var))
                x    = np.linspace(dist.ppf(0.001),dist.ppf(0.999),1000)
                y    = dist.pdf(x)
                plt.plot(x,y,label="m=%d"%m)
        
        m = m-1
    
    if plot:
        plt.legend()
        plt.xlabel("Execution Time")
        plt.title("Synthetically Constructed Execution Time Distrbution")
        plt.savefig("constructed-dist.pdf")
        plt.close()

    mu.reverse()
    vr.reverse()
    return (mu,vr)

def estimate_mu(D,s,var):
    """
        Given the deadline, risk (probability)
        and variance of the distribution

        Returns the value of mu
    """
    mu = D - (np.sqrt(2*var) * sp.erfinv(1 - 2*s))
    return mu

def main_test2():
    D  = 2500
    var = 4.00
    s  = np.linspace(0.01,0.99,100)
    mu = estimate_mu(D,s,var)
    plt.plot(s,mu)

if __name__=="__main__":
    #main_test2()
    construct_pdf(2500,4,4.00,True)