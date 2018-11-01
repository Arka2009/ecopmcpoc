#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 16:19:11 2018

@author: amaity
"""
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
#import ptss_poc as ps
import heapq
import queue
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

#ph1   = stats.gamma(k,theta)
def numpy_load_test():
    #print(x)
    A = np.load("analyses/ph4db.npy")
    #b = A[24,2,:]
    #print(A)
    c = np.array(range(0,12000))
    plt.plot(c[1:],A[99,8,:])
    #plt.xlim()

def alloc_q_test():
    a = queue.Queue(1000)
    for i in range(0,5):
        a.put(np.random.randint(7,890))
    
    print(a.queue)
    print(a.queue[0])
    a.get()
    print(a.queue)


def surface_plot_test():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Grab some test data.
    X, Y, Z = axes3d.get_test_data(0.05)

    #print(Y.shape)
    # Plot a basic wireframe.
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

    plt.show()


if __name__=="__main__":
    #numpy_load_test()
    alloc_q_test()

 # # Phase 1
            # tmp              = pd.read_csv(mfile+"/dataset_ph1.csv")
            # tmp1             = tmp[tmp['t2'] > 0]
            # tmp1['sum']      = tmp1.apply(lambda row : (row.t1 + row.t2 + row.t3)/1000.0,axis=1)
            # pdf1,u1          = pdf(tmp1['sum'].values,bins)
            # ph1_table[w,c,:] = pdf1
            # #plt.plot(u1[1:],pdf1,color="black")

            # # Phase 2
            # tmp              = pd.read_csv(mfile+"/dataset_ph2.csv")
            # tmp1             = tmp[tmp['t2'] > 0]
            # tmp1['sum']      = tmp1.apply(lambda row : (row.t1 + row.t2 + row.t3)/1000.0,axis=1)
            # pdf2,u2          = pdf(tmp1['sum'].values,bins)
            # ph2_table[w,c,:] = pdf2
            # #plt.plot(u2[1:],pdf2,color="green")

            # # Phase 3
            # tmp              = pd.read_csv(mfile+"/dataset_ph3.csv")
            # tmp1             = tmp[tmp['t2'] > 0]
            # tmp1['sum']      = tmp1.apply(lambda row : (row.t1 + row.t2 + row.t3)/1000.0,axis=1)
            # pdf3,u3          = pdf(tmp1['sum'].values,bins)
            # ph3_table[w,c,:] = pdf3
            # #plt.plot(u3[1:],pdf3,color="blue")

            # # Phase 4
            # tmp              = pd.read_csv(mfile+"/dataset_ph4.csv")
            # tmp1             = tmp[tmp['t2'] > 0]
            # tmp1['sum']      = tmp1.apply(lambda row : (row.t1 + row.t2 + row.t3)/1000.0,axis=1)
            # pdf4,u4          = pdf(tmp1['sum'].values,bins)
            # ph4_table[w,c,:] = pdf4
            # #plt.plot(u4[1:],pdf4,color="red")

# Plots
    fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    
    # Phase 1
    axes[0,0].set_title("Phase1 PDF")
    axes[0,0].plot(u1[1:],pdf1,color='red')
    axes[0,0].set_ylabel("Probabilty Distribution")
    axes[0,0].set_xlabel("Execution Time (us)")
    axes[0,0].set_xlim(10,200)
    
    # Phase 2
    axes[0,1].set_title("Phase2 PDF")
    axes[0,1].plot(u2[1:],pdf2,color='red')
    axes[0,1].set_ylabel("Probabilty Distribution")
    axes[0,1].set_xlabel("Execution Time (us)")
    axes[0,1].set_xlim(10,800)
    
    # Phase 3
    axes[1,0].set_title("Phase3 PDF")
    axes[1,0].plot(u3[1:],pdf3,color='red')
    axes[1,0].set_ylabel("Probabilty Distribution")
    axes[1,0].set_xlabel("Execution Time (us)")
    axes[1,0].set_xlim(10,200)
    
    # Phase 4
    axes[1,1].set_title("Phase4 PDF")
    axes[1,1].plot(u4[1:],pdf4,color='red')
    axes[1,1].set_ylabel("Probabilty Distribution")
    axes[1,1].set_xlabel("Execution Time (us)")
    axes[1,1].set_xlim(300,600)
    
    # build a rectangle in axes coords
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height

    # Actual (PDF)
    axes[0].set_title("Actual PDF")
    axes[0].plot(u6[1:],pdfc,color='blue')
    axes[0].set_ylabel("Probabilty Distribution")
    axes[0].set_xlabel("Execution Time (ms)")
    axes[0].set_xlim(1,3)
    axes[0].text(1.8,8.4,"Mean : "+str(1.52)+"ms, stddev : "+str(0.09)+"ms")
    #axes[0].legend()
    
    # Estimated (PDF)
    axes[1].set_title("Estimated PDF")
    axes[1].plot(u5,c,color='black')
    axes[1].set_ylabel("Probabilty Distribution")
    axes[1].set_xlabel("Execution Time (ms)")
    axes[1].set_xlim(1,3)
    axes[1].text(1.8,0.0053,"Mean : "+str(1.4)+"ms, stddev : "+str(0.1)+"ms")
    
    fig.tight_layout()
    plt.savefig("test.pdf")