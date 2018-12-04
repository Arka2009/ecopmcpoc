#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:01:38 2018

@author: amaity
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import ptss_utils as ptsl
import ptss_synthpdf as psyn
import ptss_dfspdf as pgen
import heapq
import queue
import timeit
from pprint import pprint
import itertools
from pylab import meshgrid
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

def main():
    # Load the Source Distribution
    mua, vra = pgen.get_pdf()

    # Creating the allocation vector
    alloc = list(itertools.product(range(1,17),repeat=ptsl.NPH))
    slack = ptsl.D
    #print(alloc)

    risk = []
    n    = 1
    alloc2 = alloc
    alloc2.reverse()
    for a in alloc2 :
        mu  = 0
        var = 0
        for sa in a :
            mu  = mu + mua[sa-1]
            var = var + vra[sa-1]
        dist = stats.norm(loc=mu,scale=np.sqrt(var))
        r    = dist.sf(slack)


        # Store only if risk is below a threshold
        if r < 1-1e-5:
            print("complete allocation %d,"%n+" alloc : "+str(a)+",risk %f"%r)
            print("mu : %f, std : %f"%(mu,np.sqrt(var)))
            print("\n")
            risk.append(r)
        # else :
        #     print("complete allocation %d,"%n+" alloc : "+str(a))

        n = n+1
        
    #risk2 = [r2 for r2 in risk if r2 < 0.999]
    # Histogram
    plt.hist(risk)
    np.save("dmr-saved",risk)
    plt.savefig("dmr-hist.pdf")


def plot_slack():
    """
        plot a two dimensional slack surface
    """
    mua, vra = pgen.get_pdf()
    deadline = ptsl.D
    dmrexp   = 0.25

    a1    = range(1,ptsl.M+1)
    a2    = range(1,ptsl.M+1)
    shape = (len(a1),len(a2))
    X,Y   = meshgrid(a1,a2)
    Z     = np.full(shape,-1.0)

    # print(mua)
    # print(vra)
    f2 = open()
    for x in a1 :
        for y in a2 :
            mu = mua[x-1] + mua[y-1]
            vr = vra[x-1] + vra[y-1]

            dist   = stats.norm(loc=mu,scale=np.sqrt(vr))
            risk   = dist.sf(deadline)


            not_risky = (risk <= dmrexp) or (ptsl.close(dmrexp,risk,tol=1e-8))
            ins_slack = 0
        
            if  not_risky :
                if not ptsl.close(dmrexp,risk,tol=0.005):
                    new_slack      = dist.isf(dmrexp)
                    ins_slack      = deadline - new_slack
                    # print("UE(%d,%d,%d) - Slack Inserted : %f, hypo-demand : %d"%(self.subframe,self.crnti,self.state,ins_slack,demanded_cores))
                    # demanded_cores = 0

            Z[x-1,y-1] = ins_slack

            if (ins_slack > 0) :
                x2.append(x)
                y2.append(y)

    # print(Z)
    # for 
    # plt.scatter()
    # fig  = plt.figure(figsize=(10,8))
    # ax   = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
    #                        cmap=cm.RdBu,linewidth=0, antialiased=False)
    # ax.set_xlabel("alloc ph1")
    # ax.set_ylabel("alloc ph2")
    # ax.set_zlabel("Inserted Slack")
    # ax.set_title("Slack Surface")
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig("Slack.pdf")
    # plt.close()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # levels=[Z[14,14],Z[20,20],Z[24,24],Z[28,28],Z[31,31]]
    # print(levels)
    # levels.sort()
    # print(levels)
    # qset = plt.contour(X,Y,Z,levels=levels)
    # plt.plot(a1,a2,marker="*")
    # print(qset.levels)
    # for l in qset.collections : 
    #     print(l.get_array())
    # ax.annotate("(%s,%s,%s)"% (14,14,1.82), xy=(14,14), textcoords='data')
    # ax.annotate("(%s,%s,%s)"% (17,17,4.63), xy=(17,17), textcoords='data')
    # ax.annotate("(%s,%s,%s)"% (19,19,6.75), xy=(19,19), textcoords='data')
    # ax.annotate("(%s,%s,%s)"% (20,20,7.36), xy=(20,20), textcoords='data')
    # ax.set_xlabel("alloc ph1")
    # ax.set_ylabel("alloc ph2")
    # ax.set_title("Contour plots for 2-phase allocations")
    # plt.savefig("Slack2.pdf")
    # plt.close()

def compute_risk(mua,vra,alloc,slack):
    """
        Accept an allocation 
        as a list
    """
    mu  = 0
    var = 0
    for sa in alloc :
        mu  = mu + mua[sa-1]
        var = var + vra[sa-1]
    dist = stats.norm(loc=mu,scale=np.sqrt(var))
    r    = dist.sf(slack)
    return r

def main3_alg():
    mua, vra = pgen.get_pdf()
    slack    = ptsl.D
    #dmrexp   = 0.25

    all_alloc = list(itertools.product(range(1,17),repeat=ptsl.NPH))
    riska     = []
    f2        = open("risk-file-D216-NPH5.csv","w")
    f2.write("alloc1,alloc2,alloc3,alloc4,alloc5,risk,util\n")
    for a in all_alloc :
        a1, a2, a3, a4, a5 = a
        r = compute_risk(mua, vra, a, slack)
    
        if r > 0.00001 and r < 1 - 0.00001 :
            riska.append(r)
            # print("alloc ")
            # print(a)
            # print(" Finished")
            util = a1 * mua[a1-1] + a2 * mua[a2-1] + a3 * mua[a3-1] + a4 * mua[a4-1] + a5 * mua[a5-1]
            f2.write("%d,%d,%d,%d,%d,%f,%f\n"%(a1,a2,a3,a4,a5,r,util))
    f2.close()
    np.save("stored_risk",riska)



if __name__=="__main__":
    #main()
    #print()
    # for u in (16, 16, 16, 14, 9) :
    #     print(u)
    # risk = np.load("dmr-saved.npy")
    # plt.hist(risk,bins=np.arange(0.1,0.9,0.1))
    # plt.savefig("dmr-hist2.pdf")
    # plot_slack()
    main3_alg()
