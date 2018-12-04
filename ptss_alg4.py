#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import ptss_utils as ptsl
import ptss_synthpdf as psyn
import ptss_dfspdf as pgen
import ptss_risksteepness as porc
import heapq
import queue
import timeit
from pprint import pprint
import itertools
from pylab import meshgrid
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats


mua, vra = pgen.get_pdf()
def alg4():
    """
        Find an allocation using 
        greedy state-space search
        as discussed on 03/12/2018
    """
    alloc = [1,1,1,1,1]
    dmrexp     = 0.3
    
    while True :
        r     = porc.compute_risk(mua,vra,alloc,ptsl.D)
        
        if (r < dmrexp) :
            break
        
        # Find the idx with maximum possible decrease in diff_et
        diff_et = []
        for ph in range(ptsl.NPH) :
            tmp =  mua[alloc[ph]]/(alloc[ph])
            diff_et.append(tmp)
        
        idx = np.argmax(diff_et)
        alloc[idx] = alloc[idx]+1
    
    print(r)
    print(alloc)


    # Compute the estimated utilization
    util = 0
    for ph in alloc :
        util = util + ph * mua[ph-1]
    
    print(util)
    return alloc



if __name__=="__main__":
    alg4()