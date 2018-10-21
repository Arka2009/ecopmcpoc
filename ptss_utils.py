#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:10:50 2018

@author: amaity
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import timeit
import scipy.interpolate as interpolate

# Global Constants
NUMBINS   = 2000                    # Used Internally by CDF and PDFs functions
T         = 1000                    # In us
D         = 2500                    # Deadline
M         = 26                      # Total number of Cores available in the system
W         = 100                     # Total number of PRBs 

# TODO : Uncomment the following global variables when you want to create the pdf database file
##(Check 16000/12000/10000)
#bins      = list(range(0,12000))     # Bin-edges of the histogram/pdf (Bin-width = 1us) of each phase execution times
#
#shape     = (W,M,len(bins)-1)
#ph1_table = np.full(shape,-1.2)       # Table to store the distribution of Phase 1 Execution Times
#ph2_table = np.full(shape,-1.2)       # Table to store the distribution of Phase 2 Execution Times
#ph3_table = np.full(shape,-1.2)       # Table to store the distribution of Phase 3 Execution Times
#ph4_table = np.full(shape,-1.2)       # Table to store the distribution of Phase 4 Execution Times
#
## 
## Table to store the
## Execution Time Distribution of the remaining computations
## of a subframe, applicable only with a bin
## size of 12000 (So don't change the number of bins above)
##
#waiting_remcomp_db = np.full((W,M,83987),-1.2)
#ph1s1_remcomp_db   = np.full((W,M,71989),-1.2)
#ph2s1_remcomp_db   = np.full((W,M,59991),-1.2)
#ph3s1_remcomp_db   = np.full((W,M,47993),-1.2)
#ph1s2_remcomp_db   = np.full((W,M,35995),-1.2)
#ph2s2_rempcom_db   = np.full((W,M,23997),-1.2)
#ph3s2_remcomp_db   = np.full((W,M,11999),-1.2)

def cdf(dataset,bins=NUMBINS):
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

def cdf2(pdf,bins):
    """
        Compute the CDFs of the given pdf
        Always use bins = len(pdf)
    """
    cdf_vals        = np.cumsum(pdf)
    xp              = bins
    Fp              = cdf_vals/cdf_vals[-1]
    return (xp,Fp)

def inv_transform_sampling(cdf,bins):
    """ 
        The inverse transform samping is used to obtain
        random samples from an RV whose cdf is given
    """
    inv_cdf = interpolate.interp1d(cdf,bins)
    r = np.random.rand(1)
    y = inv_cdf(r)
    return y[0]


def p_geq_r(dataset,value,bins=NUMBINS):
    """ Return the probability of the "value" greater than the dataset's underlying distribution """
    xp,Fp = cdf(dataset,NUMBINS)
    risk  = (1-np.interp(value,xp,Fp))*100
    return risk

def pdf(dataset,bins=NUMBINS):
    pdf,bin_edges = np.histogram(dataset,bins=bins,density=True)
    return (pdf,bin_edges)

# Workload,Allocation vs Distribution Tables
def init_workload_risk_alloc_table():
    
    for w in range(0,100):
        for c in range(1,M):
            start_time     = timeit.default_timer()
            
            mfile          = "model-data/ecolab-knl/"+"alloc_prbs-"+str(w+1)+"_cores-"+str(c+1)

            # Phase 1
            tmp              = pd.read_csv(mfile+"/dataset_ph1.csv")
            tmp1             = tmp[tmp['t2'] > 0]
            tmp1['sum']      = tmp1.apply(lambda row : (row.t1 + row.t2 + row.t3)/1000.0,axis=1)
            pdf1,u1          = pdf(tmp1['sum'].values,bins)
            ph1_table[w,c,:] = pdf1
            #plt.plot(u1[1:],pdf1,color="black")

            # Phase 2
            tmp              = pd.read_csv(mfile+"/dataset_ph2.csv")
            tmp1             = tmp[tmp['t2'] > 0]
            tmp1['sum']      = tmp1.apply(lambda row : (row.t1 + row.t2 + row.t3)/1000.0,axis=1)
            pdf2,u2          = pdf(tmp1['sum'].values,bins)
            ph2_table[w,c,:] = pdf2
            #plt.plot(u2[1:],pdf2,color="green")

            # Phase 3
            tmp              = pd.read_csv(mfile+"/dataset_ph3.csv")
            tmp1             = tmp[tmp['t2'] > 0]
            tmp1['sum']      = tmp1.apply(lambda row : (row.t1 + row.t2 + row.t3)/1000.0,axis=1)
            pdf3,u3          = pdf(tmp1['sum'].values,bins)
            ph3_table[w,c,:] = pdf3
            #plt.plot(u3[1:],pdf3,color="blue")

            # Phase 4
            tmp              = pd.read_csv(mfile+"/dataset_ph4.csv")
            tmp1             = tmp[tmp['t2'] > 0]
            tmp1['sum']      = tmp1.apply(lambda row : (row.t1 + row.t2 + row.t3)/1000.0,axis=1)
            pdf4,u4          = pdf(tmp1['sum'].values,bins)
            ph4_table[w,c,:] = pdf4
            #plt.plot(u4[1:],pdf4,color="red")
            
            elapsed = timeit.default_timer() - start_time
            print("Finished PDF computation of %d-prb and %d-core case in %f seconds " % (w+1,c,elapsed))

    # Save the numpy objects in a npy file
    np.save("ph1db.npy",ph1_table)
    np.save("ph2db.npy",ph2_table)
    np.save("ph3db.npy",ph3_table)
    np.save("ph4db.npy",ph4_table)

def compute_cumulative_risk():
    ph1tbl = np.load("analyses/ph1db.npy")
    ph2tbl = np.load("analyses/ph2db.npy")
    ph3tbl = np.load("analyses/ph3db.npy")
    ph4tbl = np.load("analyses/ph4db.npy")
    
    for w in range(0,W):
        for c in range(1,M):
            start_time     = timeit.default_timer()
            pdf1   = ph1tbl[w,c,:]
            pdf2   = ph2tbl[w,c,:]
            pdf3   = ph3tbl[w,c,:]
            pdf4   = ph4tbl[w,c,:]
            
            # Compute waiting_remcomp_pdf
            t0                  = np.convolve(pdf1,pdf2)
            t1                  = np.convolve(t0,pdf3)
            t2                  = np.convolve(t1,t1)
            waiting_remcomp_pdf = np.convolve(t2,pdf4)
            u                   = list(range(0,len(waiting_remcomp_pdf)))
            xp, Fp              = cdf2(waiting_remcomp_pdf,u)
            waiting_remcomp_db[w,c,:] = Fp
            #print(len(Fp))
            
            # Compute ph1s1_remcomp_pdf
            t1                  = np.convolve(pdf2,pdf3)
            t2                  = np.convolve(t1,t1)
            t3                  = np.convolve(pdf1,t2)
            ph1s1_remcomp_pdf   = np.convolve(t3,pdf4)
            u                   = list(range(0,len(ph1s1_remcomp_pdf)))
            xp, Fp              = cdf2(ph1s1_remcomp_pdf,u) 
            ph1s1_remcomp_db[w,c,:] = Fp
            #print(len(Fp))
            
            # Compute ph2s1_remcomp_pdf
            t1                  = np.convolve(np.convolve(pdf1,pdf2),pdf4)
            t2                  = np.convolve(pdf3,pdf3)
            ph2s1_remcomp_pdf   = np.convolve(t1,t2)
            u                   = list(range(0,len(ph2s1_remcomp_pdf)))
            xp, Fp              = cdf2(ph2s1_remcomp_pdf,u)
            ph2s1_remcomp_db[w,c,:] = Fp
            #print(len(Fp))
            
            # Compute ph3s1_remcomp_pdf
            t1                  = np.convolve(pdf1,pdf2)
            t2                  = np.convolve(pdf3,pdf4)
            ph3s1_remcomp_pdf   = np.convolve(t1,t2)
            u                   = list(range(0,len(ph3s1_remcomp_pdf)))
            xp, Fp              = cdf2(ph3s1_remcomp_pdf,u)
            ph3s1_remcomp_db[w,c,:] = Fp
            #print(len(Fp))
            
            # Compute ph1s2_remcomp_pdf
            t1                  = np.convolve(pdf2,pdf3)
            ph1s2_remcomp_pdf   = np.convolve(t1,pdf4)
            u                   = list(range(0,len(ph1s2_remcomp_pdf)))
            xp, Fp              = cdf2(ph1s2_remcomp_pdf,u)
            ph1s2_remcomp_db[w,c,:] = Fp
            #print(len(Fp))
            
            # Compute ph2s2_remcomp_pdf
            ph2s2_remcomp_pdf   = np.convolve(pdf3,pdf4)
            u                   = list(range(0,len(ph2s2_remcomp_pdf)))
            xp, Fp              = cdf2(ph2s2_remcomp_pdf,u)
            ph2s2_rempcom_db[w,c,:] = Fp
            #print(len(Fp))
            
            # Compute ph3s2_remcomp_pdf
            ph3s2_remcomp_pdf   = pdf4
            u                   = list(range(0,len(ph3s2_remcomp_pdf)))
            xp, Fp              = cdf2(ph3s2_remcomp_pdf,u)
            ph3s2_remcomp_db[w,c,:] = Fp
            #print(len(Fp))
            
            elapsed = timeit.default_timer() - start_time
            print("Finshed Cumulative risk computation of %d-prb and %d-core case in %f seconds " % (w+1,c,elapsed))
    
    np.save("ph1231234db_cum.npy",waiting_remcomp_db)
    np.save("ph231234db_cum.npy",ph1s1_remcomp_db)
    np.save("ph31234db_cum.npy",ph2s1_remcomp_db)
    np.save("ph1234db_cum.npy",ph3s1_remcomp_db)
    np.save("ph234db_cum.npy",ph1s2_remcomp_db)
    np.save("ph34db_cum.npy",ph2s2_rempcom_db)
    np.save("ph4db_cum.npy",ph3s2_remcomp_db)

def get_occupied_cores(alloc,time):
    """ return the number of occupied cores in a time step """
    total = 0
    for (start,end,m) in alloc:
        if time < end and time > start:
            total = total + m
    return total
 
if __name__=="__main__":
    # x = [np.random.normal() for u in range(0,1000000)]
    # y,u = pdf(x)
    # plt.plot(u[1:],y)
    #init_workload_risk_alloc_table()
    #compute_cumulative_risk()
    print("Blah")