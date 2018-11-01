#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:23:03 2018

@author: amaity


This module primarily is used to validate/compare the 
the estimated PDF with the actual PDF for each
phases as well as for the total execution
"""
                                                                  
import numpy as np                                                              
import pandas as pd                                                             
import matplotlib.pyplot as plt                                                 
import ptss_utils as ptsl
import timeit
from pylab import meshgrid
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

def compute_pdf_distance_v1():
    """
        An improved version of compute_mean_diff1

        Computes the difference mean (and std) of estimated 
        subframe execution time distribution
        and actual subframe execution time distribution

        PDFs of subframe execution time are computed 
        using method-1
    """
    shape          = (ptsl.W,ptsl.M,2)    # Workload, Cores and Mean estimated, Mean actual and Number of Cores
    ret_mean       = np.full(shape,-1.7)  # for storing the mean
    ret_std        = np.full(shape,-1.7)  # for storing the std 

    # The pdfs are already computed, load them from the pdf db file
    ph1_table = np.load("pdf-risk-db3/ph1db.npy")
    ph2_table = np.load("pdf-risk-db3/ph2db.npy")
    ph3_table = np.load("pdf-risk-db3/ph3db.npy")
    ph4_table = np.load("pdf-risk-db3/ph4db.npy")
    i1_table  = np.load("pdf-risk-db3/i1db.npy")
    i2_table  = np.load("pdf-risk-db3/i2db.npy")
    crc_table = np.load("pdf-risk-db3/crcdb.npy")
    
    for w in range(89,100):
        i1  = ptsl.etPDF(i1_table[w,:],ptsl.BINS)
        i2  = ptsl.etPDF(i2_table[w,:],ptsl.BINS)
        crc = ptsl.etPDF(crc_table[w,:],ptsl.BINS)
        #print(pd.DataFrame(i2_table[w,:]))
        #return
        for m in range(1,ptsl.M):
            start_time     = timeit.default_timer()

            # Retrieve the PDFs of all the phases
            pdf1 = ptsl.etPDF(ph1_table[w,m,:],ptsl.BINS)
            pdf2 = ptsl.etPDF(ph2_table[w,m,:],ptsl.BINS)
            pdf3 = ptsl.etPDF(ph3_table[w,m,:],ptsl.BINS)
            pdf4 = ptsl.etPDF(ph4_table[w,m,:],ptsl.BINS)
                
            # Compose the execution time distribution
            sfet            = pdf1 + pdf1 + pdf2 + pdf2 + i1 + i1 + pdf3 + pdf3 + i2 + pdf4 + crc
            ret_mean[w,m,0] = sfet.mean()
            ret_std[w,m,0]  = sfet.std()
            print(sfet)

            # (Actual Distribution)
            tmp              = pd.read_csv("/home/amaity/Desktop/Datasets/ptss-raw-execution-data/ecolab-knl-2018-10-28/alloc_prbs-"+str(w+1)+"_cores-"+str(m+1)+"/dataset_sf.csv")
            c2               = (tmp['ExecutionTime'].values) * 1000.0
            # Median Filtering
            m2               = np.median(c2)
            c4               = list(filter((lambda x : abs(x-m2) < 5*m2),c2))
            ret_mean[w,m,1]  = np.mean(c4)
            ret_std[w,m,1]   = np.std(c4)
            print("Actual Distribution Mean : %f, std %f" % (np.mean(c4),np.std(c4)))

            # Also compute the error
            err_mean = abs(ret_mean[w,m,0] - ret_mean[w,m,1])*100/ret_mean[w,m,0]
            err_std  = abs(ret_std[w,m,0] - ret_std[w,m,1])*100/ret_std[w,m,0]
            elapsed = timeit.default_timer() - start_time
            print("Error mean : %.2f, std : %.2f"%(err_mean,err_std))
            print("Computed discrepancy for %d prbs on %d cores in %f seconds\n\n"%(w+1,m,elapsed))
            
    np.save("pdf-discrepancy-mean.npy",ret_mean)
    np.save("pdf-discrepancy-std.npy",ret_std)
    #return ret

def compute_pdf_distance_v2():
    """
        The PDFs of subframe
        from each phase using method-2
    """
    shape          = (ptsl.W,ptsl.M,2)    # Workload, Cores and Mean estimated, Mean actual and Number of Cores
    ret_mean       = np.full(shape,-1.7)  # for storing the mean
    ret_std        = np.full(shape,-1.7)  # for storing the std

    # The pdfs are already computed, load them from the pdf db file
    ph1s1_table = np.load("pdf-db3-v2/ph1s1db.npy")
    ph2s1_table = np.load("pdf-db3-v2/ph2s1db.npy")
    i1s1_table  = np.load("pdf-db3-v2/i1s1db.npy")
    ph3s1_table = np.load("pdf-db3-v2/ph3s1db.npy")
    ph1s2_table = np.load("pdf-db3-v2/ph1s2db.npy")
    ph2s2_table = np.load("pdf-db3-v2/ph2s2db.npy")
    ph3s2_table = np.load("pdf-db3-v2/ph3s2db.npy")
    i1s2_table  = np.load("pdf-db3-v2/i1s2db.npy")
    i2_table    = np.load("pdf-db3-v2/i2db.npy")
    ph4_table   = np.load("pdf-db3-v2/ph4db.npy")
    crc_table   = np.load("pdf-db3-v2/crcdb.npy")
    
    for w in range(89,100):
        for m in range(1,ptsl.M):
            start_time     = timeit.default_timer()

            # Retrieve the PDFs of all the phases
            pdf1s1 = ptsl.etPDF(ph1s1_table[w,m,:],ptsl.BINS)
            pdf2s1 = ptsl.etPDF(ph2s1_table[w,m,:],ptsl.BINS)
            i1s1   = ptsl.etPDF(i1s1_table[w,m,:],ptsl.BINS)
            pdf3s1 = ptsl.etPDF(ph3s1_table[w,m,:],ptsl.BINS)
            pdf1s2 = ptsl.etPDF(ph1s2_table[w,m,:],ptsl.BINS)
            pdf2s2 = ptsl.etPDF(ph2s2_table[w,m,:],ptsl.BINS)
            i1s2   = ptsl.etPDF(i1s2_table[w,m,:],ptsl.BINS)
            pdf3s2 = ptsl.etPDF(ph3s2_table[w,m,:],ptsl.BINS)
            
            i2     = ptsl.etPDF(i2_table[w,m,:],ptsl.BINS)
            pdf4   = ptsl.etPDF(ph4_table[w,m,:],ptsl.BINS)
            crc    = ptsl.etPDF(crc_table[w,m,:],ptsl.BINS)
                
            # Compose the execution time distribution
            sfet            = pdf1s1 + pdf1s2 + pdf2s1 + pdf2s2 + i1s1 + i1s2 + pdf3s1 + pdf3s2 + i2 + pdf4 + crc
            ret_mean[w,m,0] = sfet.mean()
            ret_std[w,m,0]  = sfet.std()
            #print(sfet)

            # (Actual Distribution)
            tmp              = pd.read_csv("/home/amaity/Desktop/Datasets/ptss-raw-execution-data/ecolab-knl-2018-10-28/alloc_prbs-"+str(w+1)+"_cores-"+str(m+1)+"/dataset_sf.csv")
            c2               = (tmp['ExecutionTime'].values) * 1000.0
            # Median Filtering
            m2               = np.median(c2)
            c4               = list(filter((lambda x : abs(x-m2) < 5*m2),c2))
            ret_mean[w,m,1]  = np.mean(c4)
            ret_std[w,m,1]   = np.std(c4)
            #print("Actual Distribution Mean : %f, std %f" % (np.mean(c4),np.std(c4)))

            # # Plot the pdf
            # fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            # # Actual (PDF)
            # axes[0].set_title("Actual PDF")
            # axes[0].hist(c4,bins=ptsl.BINS,color='blue')
            # axes[0].set_ylabel("Probabilty Distribution")
            # axes[0].set_xlabel("Execution Time (us)")
            # axes[0].set_xlim(500,5000)

            # # Estimated (PDF)
            # axes[1].set_title("Estimated PDF")
            # axes[1].plot(sfet.xp,sfet.pdf(sfet.xp),color='black')
            # axes[1].set_ylabel("Probabilty Distribution")
            # axes[1].set_xlabel("Execution Time (us)")
            # axes[1].set_xlim(500,5000)
            # #axes[1].set_xlim(0.7,3)

            # Also compute the error
            err_mean = abs(ret_mean[w,m,0] - ret_mean[w,m,1])*100/ret_mean[w,m,0]
            err_std  = abs(ret_std[w,m,0] - ret_std[w,m,1])*100/ret_std[w,m,0]
            elapsed = timeit.default_timer() - start_time
            print("Error mean : %.2f, std : %.2f"%(err_mean,err_std))
            print("Computed discrepancy for %d prbs on %d cores in %f seconds\n\n"%(w+1,m,elapsed))
            
    np.save("pdf-discrepancy-mean.npy",ret_mean)
    np.save("pdf-discrepancy-std.npy",ret_std)
    #return ret

def plot_err(file2,ext):
    """ 
        Plot The error between the
        actual subframe execution time
        and estimated subframe execution
        time
    """
    prb   = np.array(range(1,ptsl.W+1))  # 1-100 (100 values)
    alloc = np.array(range(2,ptsl.M+1))  # 2-26  (25 values)
    X,Y   = meshgrid(prb,alloc)
    
    stat      = np.load(file2)
    shape     = (len(alloc),len(prb))
    Z         = np.full(shape,-1.0)
    
    for m in alloc:
        for w in prb:
            # Index Shifting
            # Indices for Z
            w2 = w - 1
            m2 = m - 2
            # Indices for stat
            w3 = w - 1
            m3 = m - 1

            # print("w2 : %d,m2 : %d"%(w2,m2))
            # print("w3 : %d,m3 : %d"%(w3,m3))
            # print("Computed : %d, Actual : %d"%(stat[w3,m3,0],stat[w3,m3,1]))
            Z[m2,w2] = abs(stat[w3,m3,0]-stat[w3,m3,1])*100/stat[w3,m3,0]
    
    # Find the indices where Z is maximum
    m2,w2 = np.unravel_index(np.argmax(Z,axis=None),Z.shape)
    m = m2+2
    w = w2+1
    #np.savetxt("foo.csv", Z, delimiter=",")
    ds = pd.DataFrame(Z)
    #print(ds)
    ds.to_csv("Error-"+ext+".csv")
    #print("m : %.2f,w : %.2f"%(m,w))
    #print("Average ; %.2f\n" % (np.mean(Z)))
    #return
    fig  = plt.figure(figsize=(10,8))
    ax   = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                           cmap=cm.RdBu,linewidth=0, antialiased=False)
    ax.set_xlabel("PRBs")
    ax.set_ylabel("Cores")
    ax.set_zlabel("Error Percent")
    ax.set_title("Comparison of Actual and Estimated PDF (Max = "+str(np.max(Z))+"%)")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig("Error-"+ext+".pdf")

def expansion_pdf_comp():
    """
        Compute the PDF when the
        different phases are allocated 
        different cores. Check if
        the superposition of PDF holds.
    """

    # The pdfs are already computed, load them from the pdf db file
    ph1_table = np.load("pdf-risk-db3/ph1db.npy")
    ph2_table = np.load("pdf-risk-db3/ph2db.npy")
    ph3_table = np.load("pdf-risk-db3/ph3db.npy")
    ph4_table = np.load("pdf-risk-db3/ph4db.npy")
    i1_table  = np.load("pdf-risk-db3/i1db.npy")
    i2_table  = np.load("pdf-risk-db3/i2db.npy")
    crc_table = np.load("pdf-risk-db3/crcdb.npy")

    # Retrieve the PDFs of all the phases
    i1  = ptsl.etPDF(i1_table[89,:],ptsl.BINS)
    i2  = ptsl.etPDF(i2_table[89,:],ptsl.BINS)
    crc = ptsl.etPDF(crc_table[89,:],ptsl.BINS)
    pdf1 = ptsl.etPDF(ph1_table[89,8,:],ptsl.BINS)
    pdf2 = ptsl.etPDF(ph2_table[89,16,:],ptsl.BINS)
    pdf3 = ptsl.etPDF(ph3_table[89,20,:],ptsl.BINS)
    pdf4 = ptsl.etPDF(ph4_table[89,24,:],ptsl.BINS)

    sfet   = pdf1 + pdf1 + pdf2 + pdf2 + i1 + i1 + pdf3 + pdf3 + i2 + pdf4 + crc
    print(sfet)

    # (Actual Distribution)
    tmp        = pd.read_csv("/home/amaity/Dropbox/Datasets-Analyses/ptss-poc/Validation-Datasets/alloc_prbs-90_cores-8-16-20-24/dataset_sf.csv")
    c2         = tmp['ExecutionTime'] * 1000.0
    #print(np.mean(c2))
    print("Actual Ditribution Mean : %f, std %f" % (np.mean(c2),np.std(c2)))

    # Plot the pdf
    plt.plot(sfet.xp,sfet.pdf(sfet.xp),label="Estimated")
    plt.hist(c2,bins=len(ptsl.BINS),density=True,label="Oct-28")
    plt.xlim(2200,5000)
    plt.xlabel("Execution Time (in ms)")
    plt.ylabel("PDF")
    plt.legend()
    plt.title("Execution Time Distribution")
    #plt.savefig("comparison-prb100-cores2.pdf")
    # fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # # Actual (PDF)
    # axes[0].set_title("Actual PDF")
    # axes[0].hist(c2,bins=ptsl.BINS,color='blue')
    # axes[0].set_ylabel("Probabilty Distribution")
    # axes[0].set_xlabel("Execution Time (us)")
    # axes[0].set_xlim(500,5000)
    # # Estimated (PDF)
    # axes[1].set_title("Estimated PDF")
    # axes[1].plot(sfet.xp,sfet.pdf(sfet.xp),color='black')
    # axes[1].set_ylabel("Probabilty Distribution")
    # axes[1].set_xlabel("Execution Time (us)")
    # axes[1].set_xlim(500,5000)
    # #axes[1].set_xlim(0.7,3)

def compute_correlation():
    """ 
        Compute the correlation coefficient
        amongst different phases 
    """
    w = 100
    m = 22

    N     = 5000
    ds    = np.full((7,N),-1.7)
    r     = np.full((7,7),-10.8)
    # Collect ph1s1 dataset
    mfile = "raw-execution-data/ecolab-knl-2018-10-19/alloc_prbs-"+str(w)+"_cores-"+str(m)
    
    tmp              = pd.read_csv(mfile+"/dataset_ph1.csv")
    tmp1             = tmp[tmp['t2'] > 0]
    tmp1['sum']      = tmp1.apply(lambda row : (row.t1 + row.t2 + row.t3)/1000.0,axis=1)
    tmp2             = tmp1['sum'].values
    ph1s1            = [tmp2[idx] for idx,val in enumerate(tmp2) if (idx//16) % 2 == 0]
    ds[0,:]          = ptsl.inv_transform_sampling_dataset(ph1s1,N)
    ph1s2            = [tmp2[idx] for idx,val in enumerate(tmp2) if (idx//16) % 2 == 1]
    ds[3,:]          = ptsl.inv_transform_sampling_dataset(ph1s2,N)
    
    tmp              = pd.read_csv(mfile+"/dataset_ph2.csv")
    tmp1             = tmp[tmp['t2'] > 0]
    tmp1['sum']      = tmp1.apply(lambda row : (row.t1 + row.t2 + row.t3)/1000.0,axis=1)
    tmp2             = tmp1['sum'].values
    ph2s1            = [tmp2[idx] for idx,val in enumerate(tmp2) if (idx//16) % 2 == 0]
    ds[1,:]          = ptsl.inv_transform_sampling_dataset(ph2s1,N)
    ph2s2            = [tmp2[idx] for idx,val in enumerate(tmp2) if (idx//16) % 2 == 1]
    ds[4,:]          = ptsl.inv_transform_sampling_dataset(ph2s2,N)

    tmp              = pd.read_csv(mfile+"/dataset_ph3.csv")
    tmp1             = tmp[tmp['t2'] > 0]
    tmp1['sum']      = tmp1.apply(lambda row : (row.t1 + row.t2 + row.t3)/1000.0,axis=1)
    tmp2             = tmp1['sum'].values
    ph3s1            = [tmp2[idx] for idx,val in enumerate(tmp2) if (idx//16) % 2 == 0]
    ds[2,:]          = ptsl.inv_transform_sampling_dataset(ph3s1,N)
    ph3s2            = [tmp2[idx] for idx,val in enumerate(tmp2) if (idx//16) % 2 == 1]
    ds[5,:]          = ptsl.inv_transform_sampling_dataset(ph3s2,N)

    tmp              = pd.read_csv(mfile+"/dataset_ph4.csv")
    tmp1             = tmp[tmp['t2'] > 0]
    tmp1['sum']      = tmp1.apply(lambda row : (row.t1 + row.t2 + row.t3)/1000.0,axis=1)
    ph4              = tmp1['sum'].values
    ds[6,:]          = ptsl.inv_transform_sampling_dataset(ph4,N)


    # Compute the pearson correaltion coefficient
    for i in range(0,7):
        for j in range(0,7):
            r[i,j],_ = stats.pearsonr(ds[i,:],ds[j,:])
    df = pd.DataFrame(r)
    xfile = "r-w"+str(w)+"-m"+str(m)+".xlsx"
    df.to_excel(xfile,index=False)
    print(df)
    
        
if __name__=="__main__":
    #compute_correlation()
    #compute_pdf_distance_v2()
    plot_err("pdf-discrepancy-mean.npy","mean")
    plot_err("pdf-discrepancy-std.npy","std")
    #expansion_pdf_comp()
    #expansion_pdf_comp()