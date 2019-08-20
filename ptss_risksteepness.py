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
from matplotlib import rc

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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


def plot_risk():
    """
        plot a two dimensional slack surface
    """
    mua,vra,pra = pgen.get_pdf(plot=False,write=False)
    deadline = ptsl.D

    a1    = range(1,17)
    a2    = range(1,17)
    shape = (len(a1),len(a2))
    X,Y   = meshgrid(a1,a2)
    Z1     = np.full(shape,-1.0) # Risk Power Surface
    Z2     = np.full(shape,-1.0) # Peak Power Surface

    for x in a1 :
        for y in a2 :
            mu = mua[x-1] + mua[y-1]
            vr = vra[x-1] + vra[y-1]

            dist   = stats.norm(loc=mu,scale=np.sqrt(vr))
            risk   = dist.sf(deadline)

            Z1[x-1,y-1] = risk
            Z2[x-1,y-1] = np.max([pra[x-1],pra[y-1]])


    fig  = plt.figure()
    ax   = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, 
                           cmap=cm.Greys,linewidth=0, antialiased=False)
    ax.set_xlabel("alloc ph1")
    ax.set_ylabel("alloc ph2")
    ax.set_zlabel("Risk")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig("risk-surface.pdf",bbox_inches='tight')
    plt.close()

    fig  = plt.figure()
    ax   = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, 
                           cmap=cm.Greys_r,linewidth=0, antialiased=False)
    ax.set_xlabel("alloc ph1")
    ax.set_ylabel("alloc ph2")
    ax.set_zlabel("peak power")
    # print(ax.azim)
    ax.view_init(azim=-30)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig("pkp-surface.pdf",bbox_inches='tight')
    plt.close()



    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # levels=[Z3[7,7]]
    # print(levels)
    # # levels.sort()
    # # print(levels)
    # qset = plt.contour(X,Y,Z3,levels=levels)
    # # plt.plot(a1,a2,marker="*")
    # # print(qset.levels)
    # # for l in qset.collections : 
    # #     print(l.get_array())
    # # ax.annotate("(%s,%s,%s)"% (14,14,1.82), xy=(14,14), textcoords='data')
    # # ax.annotate("(%s,%s,%s)"% (17,17,4.63), xy=(17,17), textcoords='data')
    # # ax.annotate("(%s,%s,%s)"% (19,19,6.75), xy=(19,19), textcoords='data')
    # # ax.annotate("(%s,%s,%s)"% (20,20,7.36), xy=(20,20), textcoords='data')
    # ax.set_xlabel("alloc ph1")
    # ax.set_ylabel("alloc ph2")
    # ax.set_title("Contour plots for 2-phase allocations")
    # plt.savefig("Util2.pdf")
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

def compute_util(mua,alloc):
    util = 0
    for sa in alloc :
        util = util + sa*mua[sa-1]
    return util

def evaluate_all_points():
    """
        Assuming NPH = 5
    """
    start_time  = timeit.default_timer()
    mua, vra = pgen.get_pdf()
    slack    = ptsl.D

    all_alloc = list(itertools.product(range(1,ptsl.M+1),repeat=ptsl.NPH))
    riska     = []
    f2        = open("risk-file-D216-NPH5.csv","w")
    f2.write("alloc1,alloc2,alloc3,alloc4,alloc5,risk,util\n")
    count = 0
    for a in all_alloc :
        a1, a2, a3, a4, a5 = a
        r = compute_risk(mua, vra, a, slack)
    
        if r > 0.00001 and r < 1 - 0.00001 :
            riska.append(r)
            util = a1 * mua[a1-1] + a2 * mua[a2-1] + a3 * mua[a3-1] + a4 * mua[a4-1] + a5 * mua[a5-1]
            f2.write("%d,%d,%d,%d,%d,%f,%f\n"%(a1,a2,a3,a4,a5,r,util))
            count = count + 1
    f2.close()
    np.save("stored_risk",riska)
    elapsed = timeit.default_timer() - start_time
    print("Brute Force Evaluation Time for %d points : %fs"%(count,elapsed))



if __name__=="__main__":
    #main()
    #print()
    # for u in (16, 16, 16, 14, 9) :
    #     print(u)
    # risk = np.load("dmr-saved.npy")
    # plt.hist(risk,bins=np.arange(0.1,0.9,0.1))
    # plt.savefig("dmr-hist2.pdf")
    # plot_slack()
    # evaluate_all_points()
    plot_risk()


