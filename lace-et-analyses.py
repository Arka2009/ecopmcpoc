#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:22:34 2018

@author: amaity
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import scipy
import scipy.signal
from scipy.stats import norm
import timeit
import io
from enum import Enum
import ptss_utils as ptsl


def plot_ph_dist():
    """
        Plot the Distribution of Each
        phase.
    """
    phases = range(1,4)
    ph     = []
    phapp  = [] # Normal Approximation
    inf    = 0
    sup    = 200.0
    T      = 200000 # Number of Bins-Edges
    bine   = np.linspace(inf,sup,T)
    fl2    = "/home/amaity/Dropbox/Datasets-Analyses/ptss-poc/lace_case23"
    bine   = np.linspace(inf,sup,T)
    TOH    = 0.008622 # OH due to timing instrumentation
    #fig,axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 7))
    for i in phases :
        start_time = timeit.default_timer()

        phx   = pd.read_csv(fl2+"/dataset_ph"+str(i)+".csv")
        # if (i == 1) :
        #     phx = pd.read_csv("/home/amaity/Dropbox/Datasets-Analyses/ptss-poc/lace_case21"+"/dataset_ph"+str(i)+".csv")
        # else :
        #     phx = pd.read_csv("/home/amaity/Dropbox/Datasets-Analyses/ptss-poc/lace_case21"+"/dataset_ph"+str(i)+".csv")
        t1    = phx['time'].values
        t2    = t1*1000.0
        ulim  = np.max(t2)
        llim  = np.min(t2)
        print("ph-%d, llim : %d"%(i,llim))
        print("ph-%d, ulim : %d"%(i,ulim))

        # Store the distribution
        phet = ptsl.etPDF(t2,bine)
        ph.append(phet)
        phapp.append((np.mean(t2),np.std(t2)))

        # Evaluate the Distribution at selected points
        alpha  = 0.95
        beta   = 1
        a      = llim*alpha
        b      = ulim*beta
        x      = np.linspace(a,b,num=1000) #np.arange(a,b,(b-a)/1000)
        y      = phet.pdf(x)
        pdfapp = norm(loc=np.mean(t2),scale=np.std(t2))
        y2     = pdfapp.pdf(x)

        # Plot 
        plt.hist(t2,bins=1000,density=True,label="Actual Distribution")
        plt.plot(x,y2,label="Normal (mean:%04.2fms,std:%04.2fms)"%(np.mean(t2),np.std(t2)),color="black")
        plt.xlabel("Execution Time (ms)")
        plt.title("Ph"+str(i)+" Execution Time Distribution")
        #plt.xlim(llim*alpha,ulim*beta)
        plt.legend()
        plt.savefig(fl2+"/black-ph"+str(i)+".pdf")
        plt.close()

        # # Compact plot
        # axes[i-1].set_title("Ph"+str(i)+" Execution Time Distribution")
        # axes[i-1].hist(t2,bins=1000,density=True,label="Actual Distribution")
        # axes[i-1].plot(x,y2,label="Normal Approx",color="black")
        # if i == 1 :
        #     axes[i-1].set_ylabel("Probabilty Distribution")
        # axes[i-1].set_xlabel("Execution Time (s)")
        
        elapsed_time = timeit.default_timer() - start_time
        print("Finished phase"+str(i)+" in %.2fs"%elapsed_time)

    # Convolution
    elem     = ph[0]
    elem3_mu,tmp_std = phapp[0]
    elem3_var = tmp_std**2
    print(ph[0])
    for i in range(1,len(ph)):
        #print(ph[i])
        mu, std = phapp[i]
        elem3_mu = elem3_mu + mu
        elem3_var = elem3_var + std**2
        elem = elem + ph[i]
    elem3_std = np.sqrt(elem3_var) 
    #print(elem)
    

    # Original distribution
    sfx = pd.read_csv(fl2+"/dataset_sf2.csv")
    t3  = sfx['time'].values
    t4  = (t3 + TOH*(len(phases)-1))*1000.0
    
    # Generate Estimated Distribution
    ulim    = np.max(t4)
    llim    = np.min(t4)
    alpha   = 0.97
    beta    = 1.1
    a       = llim*alpha
    b       = ulim*beta
    x2      = np.linspace(a,b,num=1000)
    elem3   = norm(loc=elem3_mu,scale=elem3_std) # Normal Approximation
    y2      = elem.pdf(x2)
    y4      = elem3.pdf(x2)
    
    # Error in STD and Mean
    print("Convolution Error")
    err_mean1 = abs(elem.mean()-np.mean(t4))*100/abs(np.mean(t4))
    err_std1  = abs(elem.std()-np.std(t4))*100/abs(np.std(t4))
    print(err_mean1)
    print(err_std1)
    err_mean2 = abs(elem3.mean()-np.mean(t4))*100/abs(np.mean(t4))
    err_std2  = abs(elem3.std()-np.std(t4))*100/abs(np.std(t4))
    print("Normal Approximation Error")
    print(err_mean2)
    print(err_std2)

    # Plot
    plt.hist(t4,bins=800,density=True,label="Actual Distribution")
    plt.plot(x2,y2,label="Conv Error (mean:%04.2f%%,std:%04.2f%%)"%(err_mean2,err_std2),color="red")
    plt.plot(x2,y4,label="Normal Error (mean:%04.2f%%,std:%04.2f%%)"%(err_mean2,err_std2),color="green")
    plt.xlabel("Execution Time (ms)")
    plt.xlim(a,b)
    plt.title("Actual Distribution vs Approximation")
    plt.legend()
    plt.savefig(fl2+"/black-total.pdf")
    plt.close()

    # # Compact plot
    # axes[2].set_title("Actual Distribution vs Approximation")
    # axes[2].hist(t4,bins=800,density=True,label="Actual Distribution")
    # axes[2].plot(x2,y2,label="Conv Error (mean:%04.2f%%,std:%04.2f%%)"%(err_mean2,err_std2),color="red")
    # axes[2].plot(x2,y4,label="Normal Error (mean:%04.2f%%,std:%04.2f%%)"%(err_mean2,err_std2),color="green")
    # #axes[2].set_ylabel("Probabilty Distribution")
    # axes[2].set_xlabel("Execution Time (s)")
    # axes[2].set_xlim(a,b)
    # axes[2].legend()
    # fig.savefig("Nov20-lace-results.pdf")
    return ph


def plot_freq_effects():
    """ Plot the effect of frequency """
    phx1 = pd.read_csv("/home/amaity/Dropbox/Datasets-Analyses/ptss-poc/lace_case9/dataset_ph1.csv")
    phx2 = pd.read_csv("/home/amaity/Dropbox/Datasets-Analyses/ptss-poc/lace_case10/dataset_ph1.csv")
    t1   = np.array(phx1['time'].values)
    t2   = np.array(phx2['time'].values)
    # print(t1.shape)
    # print(t2.shape)
    # t3   = np.concatenate(t1,t2)
    # print(len(t3))
    ulim  = np.max([np.max(t2),np.max(t1)])
    llim  = np.min([np.min(t2),np.max(t1)])
    plt.hist(t1,bins=1000,density=True,label="1000MHz Freq",color="blue")
    plt.hist(t2,bins=1000,density=True,label="1200MHz Freq",color="green")
    alpha  = 0.95
    beta   = 1
    plt.xlabel("Execution Time (s)")
    plt.title(" Execution Time Distribution (Different Frequency)")
    plt.xlim(llim*alpha,ulim*beta)
    plt.legend()
    plt.savefig("freq-compare.pdf")


def plot_hist(ph):
    """
        Plot the Histograms
        and Means and Variances
        of lace benchmarks

        Input the phase number
    """
    #ph  = 3
    fl2 = "/home/amaity/Dropbox/Datasets-Analyses/ptss-poc/lace_find_normal/dfs-8000-roi-instrumentation-d8w5"
    phx = pd.read_csv(fl2+"/dataset_ph"+str(ph)+".csv")
    #phx = pd.read_csv(fl2+"/dataset_sf.csv")
    t1  = phx['time'].values
    t2  = t1*1000.0
    ulim = np.max(t2)
    llim = np.min(t2)
    print("ph-%d, llim : %d"%(ph,llim))
    print("ph-%d, ulim : %d"%(ph,ulim))

    # Evaluate the Distribution at selected points
    alpha  = 1
    beta   = 1
    a      = 31 #llim*alpha
    b      = 33 #ulim*beta
    x      = np.linspace(a,b,num=1000) #np.arange(a,b,(b-a)/1000)
    pdfapp = norm(loc=np.mean(t2),scale=np.std(t2))
    y      = pdfapp.pdf(x)

    # Plot 
    plt.hist(t2,bins=1000,density=True,label="Actual Distribution")
    plt.plot(x,y,label="Normal (mean:%04.2fms,std:%04.2fms)"%(np.mean(t2),np.std(t2)),color="black")
    plt.xlabel("Execution Time (ms)")
    plt.title("Execution Time Distribution")
    plt.xlim(a,b)
    plt.legend()
    plt.savefig(fl2+"/et-sf"+str(ph)+".pdf")
    plt.close()

def plot_interference():
    phx1 = pd.read_csv("/home/amaity/Dropbox/NUS-Datasets/ptss-poc/lace_case24/dataset_ph1.csv")
    phx2 = pd.read_csv("/home/amaity/Dropbox/NUS-Datasets/ptss-poc/lace_case24/dataset_ph2.csv")
    phx3 = pd.read_csv("/home/amaity/Dropbox/NUS-Datasets/ptss-poc/lace_case24/dataset_ph5.csv")
    t1   = (phx1['time'].values)*1000.0
    t2   = np.append(t1,(phx2['time'].values)*1000.0)
    t3   = (phx3['time'].values)*1000.0
    plt.hist(t2,bins=1000,density=True,label="Isolated(mean:%04.2f,std:%04.2f)"%(np.mean(t2),np.std(t2)),color="green")
    plt.hist(t3,bins=1000,density=True,label="Interfered(mean:%04.2f,std:%04.2f)"%(np.mean(t3),np.std(t3)),color="red")
    plt.legend()
    plt.xlim(47,55)
    plt.title("Interference Study")
    plt.xlabel("Execution Time (in ms)")
    plt.ylabel("Probability Distribution")
    plt.savefig("/home/amaity/Dropbox/NUS-Datasets/ptss-poc/lace_case24/tmp.pdf")


if __name__=="__main__":
    #plot_ph_dist()
    #plot_freq_effects()
    #plot_hist(1)
    plot_interference()