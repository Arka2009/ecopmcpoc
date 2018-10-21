#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:23:03 2018

@author: amaity
"""
import os                                                                       
import numpy as np                                                              
import pandas as pd                                                             
import matplotlib.pyplot as plt                                                 
from scipy import stats                                                      
from enum import Enum 
import ptss_utils as ptsl

def main():
    ph1f = "alloc_prbs-32_cores-8/dataset_ph1.csv"
    ph2f = "alloc_prbs-32_cores-8/dataset_ph2.csv"
    ph3f = "alloc_prbs-32_cores-8/dataset_ph3.csv"
    ph4f = "alloc_prbs-32_cores-8/dataset_ph4.csv"
    bins = list(range(0,5000))
    
    tmp            = pd.read_csv(ph1f)
    tmp1           = tmp[tmp['t2'] > 0]
    tmp1['sum']    = tmp1.apply(lambda row : (row.t1 + row.t2 + row.t3)/1000.0,axis=1)
    ph1v           = tmp1['sum'].values
    pdf1,u1        = ptsl.pdf(ph1v,bins)
    
    tmp            = pd.read_csv(ph2f)
    tmp1           = tmp[tmp['t2'] > 0]
    tmp1['sum']    = tmp1.apply(lambda row : (row.t1 + row.t2 + row.t3)/1000.0,axis=1)
    ph2v           = tmp1['sum'].values
    pdf2,u2        = ptsl.pdf(ph2v,bins)
    
    tmp            = pd.read_csv(ph3f)
    tmp1           = tmp[tmp['t2'] > 0]
    tmp1['sum']    = tmp1.apply(lambda row : (row.t1 + row.t2 + row.t3)/1000.0,axis=1)
    ph3v           = tmp1['sum'].values
    pdf3,u3        = ptsl.pdf(ph3v,bins)
    
    tmp            = pd.read_csv(ph4f)
    tmp1           = tmp[tmp['t2'] > 0]
    tmp1['sum']    = tmp1.apply(lambda row : (row.t1 + row.t2 + row.t3)/1000.0,axis=1)
    ph4v           = tmp1['sum'].values
    pdf4,u4        = ptsl.pdf(ph4v,bins)

    
    # Perform Convilution pf pd1,pdf2,pdf3,pd
    print(len(u1))
    print(len(u2))
    print(len(u3))
    print(len(u4))
#    pdf123 = np.convolve(np.convolve(pdf1,pdf2),pdf3)
#    c      = np.convolve(np.convolve(pdf123,pdf123),pdf4)
#    u      = list(range(0,len(c)))
#    u5     = [x/1000.0 for x in u]
#    
#    tmp     = pd.read_csv("alloc_prbs-32_cores-8/dataset_sf.csv")
#    c2      = tmp['ExecutionTime']
#    pdfc,u2 = ptsl.pdf(c2,u5)
#    plt.plot(u2[1:],pdfc,label="Actual Distribution",marker='o')
#    plt.plot(u5,c,color="blue",label="Convoluted Distribution")
#    plt.legend()
#    plt.savefig("dump/pdf-compare.pdf")
    


if __name__=="__main__":
    #bins = list(range(0,5000))
    #print(u)
    main()