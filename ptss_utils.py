#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:10:50 2018

@author: amaity
"""
import numpy as np
import matplotlib.pyplot as plt

def cdf(dataset,bins=1000):
    """ 
        Compute the cumulative distribution of dataset.
        The optional parameter bins, specifies the number
        of the bins used for the histogram plot.
    """
    pdf,bin_edges   = np.histogram(dataset,bins=bins,density=True)
    cdf_vals        = np.cumsum(pdf)
    xp              = bin_edges[1:]
    Fp              = cdf_vals/cdf_vals[-1]
    return (xp,Fp)

def pdf(dataset,bins=2000):
    pdf,bin_edges = np.histogram(dataset,bins=bins,density=True)
    return (pdf,bin_edges)

if __name__=="__main__":
    x = [np.random.normal() for u in range(0,1000000)]
    y,u = pdf(x)
    plt.plot(u[1:],y)