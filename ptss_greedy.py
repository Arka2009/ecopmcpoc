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
import pandas as pd

mua, vra,_ = pgen.get_pdf()
def alg4():
    """
        Find an allocation using 
        greedy state-space search
        as discussed on 03/12/2018
    """
    greedyalloc     = [1,1]
    dmrexp          = 0.30
    deadline        = ptsl.D

    greedyutil      = 1.0
    minutil         = 1.0

    # print(ptsl.close(minutil,greedyutil))

    while (ptsl.close(minutil,greedyutil)) :
        # Vizualize the trajectory
        start   = timeit.default_timer()
        x2           = [greedyalloc[0]]
        y2           = [greedyalloc[1]]

        # Evaluate and compute the optimal point
        all_points   = list(itertools.product(range(1,17),repeat=ptsl.NPH))
        valid_points = []
        util_list    = []

        for a in all_points:
            risk = porc.compute_risk(mua,vra,a,deadline)
            if (risk <= dmrexp) :
                x,y  = a
                util = x*mua[x-1] + y*mua[y-1]
                valid_points.append(a)
                util_list.append(util)

        idx        = np.argmin(util_list)
        optx, opty = valid_points[idx]
        minutil    = util_list[idx]
        ascent_completed = False # Flag to indicate the completion of ascent
        hsteps           = 0     #

        # Greedy algorithm
        while True :
            r     = porc.compute_risk(mua,vra,greedyalloc,deadline)
            u     = porc.compute_util(mua,greedyalloc)

            if (hsteps >= 1):
                break
            
            if (r < dmrexp) :
                ascent_completed = True
                break

            if not ascent_completed :
                # Find the idx with maximum possible decrease in diff_et
                diff_et = []
                for ph in range(ptsl.NPH) :
                    tmp =  mua[greedyalloc[ph]]/(greedyalloc[ph])
                    diff_et.append(tmp)

                idx = np.argmax(diff_et)
                greedyalloc[idx] = greedyalloc[idx]+1

                x2.append(greedyalloc[0])
                y2.append(greedyalloc[1])
            else :
                tmpx = greedyalloc[0]
                tmpy = greedyalloc[1]

                if tmpx > 1:
                    tmpx = tmpx + 1
                if tmpy < 16:
                    tmpy = tmpy - 1
                
                greedyalloc = [tmpx,tmpy]

                x2.append(greedyalloc[0])
                y2.append(greedyalloc[1])

                hsteps = hsteps + 1


        # Plots
        a1    = range(1,17)
        a2    = range(1,17)
        shape = (len(a1),len(a2))
        X,Y   = meshgrid(a1,a2)
        Z1     = np.full(shape,-1.0)
        Z2     = np.full(shape,-1.0)
        for x in a1 :
            for y in a2 :
                Z1[x-1,y-1] = porc.compute_risk(mua,vra,[x,y],deadline)
                Z2[x-1,y-1] = porc.compute_util(mua,[x,y])

                

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ris2 = porc.compute_risk(mua,vra,[5,7],deadline)
        ut2  = porc.compute_util(mua,[5,7])
        ut3  = porc.compute_util(mua,greedyalloc)
        levels1=[ris2]
        CS1 = plt.contour(X,Y,Z1,levels=levels1,colors="white",linewidths=[0.5])
        # plt.clabel(CS1,fontsize=9, inline=1)
        #levels2=[minutil*0.7,minutil*1.4,minutil*2]
        levels2 = list(np.linspace(480,1200,100))
        levels2.append(ut2)
        levels2.append(ut3)
        levels2.sort()
        CS2 = plt.contourf(X,Y,Z2,levels=levels2)
        # plt.clabel(CS2,fontsize=5, manual=[(8.5,3)])
        # print(qset.levels)
        # plt.plot([5,7],[7,5],marker="r*")
        ax.set_xlabel("Alloc ph1")
        ax.set_ylabel("Alloc ph2")
        ax.set_title("Contour plots for 2-phase allocations")
        plt.plot(x2,y2,label="Search Trajectory",linewidth=0.2,color="magenta")
        plt.plot([optx],[opty],marker="*",color="cyan",markersize=0.8)
        plt.plot([7],[5],marker="*",label="Optimal Point",color="cyan",markersize=0.8)
        # plt.xlim(6,10)
        # plt.ylim(6,10)
        plt.legend()
        plt.savefig("greedy-trajectory.pdf")
        plt.close()

        # Compute the estimated utilization
        greedyutil = porc.compute_util(mua,greedyalloc)

        print("Deadline : %f, DMR : %f"%(deadline,dmrexp))
        #print(deadline)
        print("Optimal Allocation")
        print([optx,opty])
        print(minutil)
        print("Greedy Allocation")
        print(greedyalloc)
        print(greedyutil)
        elapsed = timeit.default_timer() - start

        deadline    = np.random.uniform(low=ptsl.D-8,high=ptsl.D+4)
        greedyalloc = [1,1]
        print("Time elapse : %f"%elapsed) 
        print("\n\n")

        # alloc2 = [8,9]
        # util2  = porc.compute_util(mua,alloc2)
        # print("Utilization with (8,9) : %f"%util2)
        break
    return greedyalloc


def algs4_sniper():
    """
        Using Sniper Profile and Power
        Data, and constraints in deadline
    """ 
    deadline        = 30 # Change it

    minutil         = 1.0

    # Read the Profile Data
    df  = pd.read_csv("/home/amaity/Dropbox/NUS-Research/ptss_risk_model/ptss-sniper/profile-lace-dfs-000.csv")
    lat = df['latency(M)'].values
    pkg = df['PKG'].values


    # Evaluate and compute the optimal point
    all_points   = list(itertools.product(range(1,64),repeat=ptsl.NPH))
    valid_points = []
    util_list    = []
    for a in all_points:
        x,y  = a
        risk = lat[x-1]+lat[y-1]
        if (risk <= deadline) :
            util = np.max([pkg[x-1],pkg[y-1]])
            # util = ((pkg[x-1]*lat[x-1]) + (lat[y-1]*pkg[y-1]))/(lat[x-1]+lat[y-1])
            valid_points.append(a)
            util_list.append(util)
    idx        = np.argmin(util_list)
    optx, opty = valid_points[idx]
    minutil    = util_list[idx]


    # Plots
    a1    = range(1,ptsl.M+1)
    a2    = range(1,ptsl.M+1)
    shape = (len(a1),len(a2))
    X,Y   = meshgrid(a1,a2)
    Z1     = np.full(shape,-1.0)
    Z2     = np.full(shape,-1.0)
    for x in a1 :
        for y in a2 :
            Z1[x-1,y-1] = lat[x-1]+lat[y-1]
            Z2[x-1,y-1] = np.max([pkg[x-1],pkg[y-1]])
            # Z2[x-1,y-1] = ((pkg[x-1]*lat[x-1]) + (lat[y-1]*pkg[y-1]))/(lat[x-1]+lat[y-1])
            
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ris2 = deadline
    # ut2  = np.max([pkg[x-1],pkg[y-1]])
    # ut3  = np.max([pkg[x-1],pkg[y-1]])
    levels1=[ris2]
    CS1 = plt.contour(X,Y,Z1,levels=levels1,colors="white",linewidths=[0.5])
#    CS1 = plt.contour(X,Y,Z1,colors="white",linewidths=[0.5])
    # plt.clabel(CS1,fontsize=9, inline=1)
    #levels2=[minutil*0.7,minutil*1.4,minutil*2]
    levels2 = list(np.linspace(12,230,10))
    # levels2.append(ut2)
    # levels2.append(ut3)
    levels2.sort()
    CS2 = plt.contourf(X,Y,Z2,levels=levels2)
    # plt.clabel(CS2,fontsize=5, manual=[(8.5,3)])
    # print(qset.levels)
    # plt.plot([5,7],[7,5],marker="r*")
    ax.set_xlabel("Alloc ph1")
    ax.set_ylabel("Alloc ph2")
    ax.set_title("Contour plots for 2-phase allocations")
    # plt.plot(x2,y2,label="Search Trajectory",linewidth=0.2,color="magenta")
    plt.plot([optx],[opty],marker="*",color="cyan",markersize=0.8)
    plt.grid(b=True, which='major', color='#666666', linewidth=0.1)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    # plt.plot([7],[5],marker="*",label="Optimal Point",color="cyan",markersize=0.8)
    # plt.xlim(6,10)
    # plt.ylim(6,10)
    plt.legend()
    plt.savefig("greedy-trajectory.pdf")
    plt.close()
    
    print("Optimal Allocation")
    print([optx,opty])
    print(minutil)





if __name__=="__main__":
    algs4_sniper()