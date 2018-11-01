#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:10:50 2018

@author: amaity
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import scipy.signal
import timeit

# Global Constants
NUMBINS   = 2000                    # Used Internally by CDF and PDFs functions
T         = 1000                    # In us
D         = 2500                    # Deadline
M         = 26                      # Total number of Cores available in the system
W         = 100                     # Total number of PRBs 

##(Check 16000/12000/10000)
DELTA     = 1.0                         # Execution Time Granularity (1us)
BINS      = np.arange(0,12000,DELTA)    # Bin-edges of the histogram/pdf (Bin-width = 1us) of each phase execution times
Xp        = np.array([(BINS[i]+BINS[i+1])/2 for i in range(0,len(BINS)-1)]) # Mid Point of Each Bins


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

class etPDF(object):
    """ 
        What is the Data-Stucture used to represent
        the distribution (a probability density function) ?

        This is a tuple represented by (bin_value,bin_edges).
        sizes of bin_value and bin_edges differ exactly by 1.

        For ptss the bin_edges are equally spaced,
        1us apart. This simplifies the convolution operation.

        Wherever possible the method names are similar to
        the one used by scipy.stats.rv_histogram class.
    """
    def __init__(self,bins,bin_edges):
        assert len(bins) + 1 == len(bin_edges),"Incompatible sizes of bins(%d) and bin_edges(%d)" % (len(bins),len(bin_edges))
        self.bins      = bins
        self.bin_edges = bin_edges
        self.delta     = 1           # Don't Touch bin-edges (always 1 us)

        # CDF values at selected x
        cdf_vals       = np.cumsum(self.bins)
        self.Fp        = cdf_vals/cdf_vals[-1]
        self.xp        = np.array([(bin_edges[i]+bin_edges[i+1])/2 for i in range(0,len(bin_edges)-1)]) # The x coordinates where pdf and cdf functions are given 

    def pdf(self,x):
         """ Evaluate the cdf at x """
         return (np.interp(x,self.xp,self.bins))

    def cdf(self,x):
        """ Evaluate the cdf at x """
        return (np.interp(x,self.xp,self.Fp))

    def sf(self,x):
        """ Survival function (1-F(x)) """
        return (1-self.cdf(x))

    def rvs(self,N):
        """ Generate N random samples from the above distribution """
        inv_cdf = scipy.interpolate.interp1d(self.Fp,self.xp)
        r = np.random.rand(N)
        y = inv_cdf(r)
        return y
    
    def mean(self):
        """ 
            Compute the mean of the distribution.
            Obtained by randomly generating a sample
            of N values and then computing the mean
            of it. Exploits the law of large numbers
        """
        # N = 1000000
        # s = self.rvs(N)
        # return (np.mean(s))
        """
            Compute mean for first principles
        """
        yp   = (self.xp) * self.pdf((self.xp)) * self.delta # f(x)
        sum2 = 0.0
        for v in yp :
            sum2 = sum2 + v
        return sum2

    
    def std(self):
        """ Compute standard deviation of the distribution """
        # N = 1000000
        # s = self.rvs(N)
        # return (np.std(s))
        """
            Compute std for first principles
        """
        mu = self.mean()
        yp = ((self.xp - mu)**2) * self.pdf((self.xp)) * self.delta
        sum2 = 0.0
        for v in yp :
            sum2 = sum2 + v
        return np.sqrt(sum2)

    def __add__(self,other):
        """ 
            Add two RV with a given distribution, 
            Aka perform convolution of its distribution.

            Check how to perform numerical convolution
            in scipy
        """
        conv_pdf   = np.convolve(self.bins,other.bins)
        conv_edges = np.arange(0,len(conv_pdf)+1)

        ret = etPDF(conv_pdf,conv_edges) # Bin edges are assumed to be same

        return ret
    
    def __radd__(self,other):
        """ 
            Add two RV with a given distribution, 
            Aka perform convolution of its distribution.

            Check how to perform numerical convolution
            in scipy
        """
        conv_pdf   = np.convolve(self.bins,other.bins)
        conv_edges = np.arange(0,len(conv_pdf)+1)

        ret = etPDF(conv_pdf,conv_edges) # Bin edges are assumed to be same

        return ret
    
    def __str__(self):
        """ Display Summary Statistics """
        return "Estimated Ditribution Mean : %f, std %f" % (self.mean(),self.std())

    def viz(self):
         """ Plot """
         start_time = timeit.default_timer()
         #print(self.std())
         plt.plot(self.xp,self.pdf(self.xp))
         #plt.hist(self.cdf(self.xp),density=True)
         #plt.xlim(1000,4000)
         plt.xlabel("Execution Time (in us)")
         plt.ylabel("PDF")
         plt.title("Execution Time Distribution")
        
         elapsed_time = timeit.default_timer() - start_time
         print("elapsed time %f " % elapsed_time)
        
    

def test_etPDF():
    # Create (Random) Datasets
    N = 500000
    x = np.random.normal(6000,1000,N)
    y = np.random.normal(6500,1300,N)
    z = x + y
    print("True Sample Mean")
    print(np.mean(x))
    print(np.mean(y))
    print(np.mean(z))
    #bins2 = np.arange(0,2,0.01)

    # Create the PDF objects
    x1, b1 = np.histogram(x,BINS,density=True)
    y1, b1 = np.histogram(y,BINS,density=True)
    xp = etPDF(x1,b1)
    yp = etPDF(y1,b1)
    print("PDF Object mean")
    print(xp.mean())
    print(yp.mean())  
    zp = yp + xp
    print(zp.mean())
    print(zp)
    #plt.plot(b1[1:],x1)
    #plt.hist(x)
    #plt.xlim()
    #z = x + y
    
def testCONV():
    """
        Study the convolution of two PDFs
    """
    x = scipy.stats.uniform(loc=2,scale=3)
    y = scipy.stats.norm(loc=0,scale=0.25)
    # print(x)
    # print(y)
    
    #NB step-size determines precision of approximation
    delta = 1e-4
    big_grid = np.arange(-10,10,delta)
    pmf1 = x.pdf(big_grid)*delta
    pmf2 = y.pdf(big_grid)*delta
    # print(len(pmf1))
    # print(len(pmf2))
    conv_pmf  = scipy.signal.fftconvolve(pmf1,pmf2,'same')
    conv_pmf  = conv_pmf/sum(conv_pmf)

    #big_grid2 = np.linspace(-10,10,len(conv_pmf))
    plt.plot(big_grid,pmf1, label='Tophat')
    plt.plot(big_grid,pmf2, label='Gaussian error')
    plt.plot(big_grid,conv_pmf, label='Sum')
    plt.xlim(-3,max(big_grid))
    plt.legend(loc='best'), plt.suptitle('PMFs')

def get_dataset_phase(prefixf):
    """ 
        Return the execution time dataset 
        from a folder "prefixf" and return the
        execution time for the phases
        Also filter out the zero values.
    """
    #start_time     = timeit.default_timer()
    tmp            = pd.read_csv(prefixf+"/dataset_ph1.csv")
    tmp1           = tmp[tmp['t2'] > 0]
    tmp1['sum']    = tmp1.apply(lambda row : (row.t1 + row.t2 + row.t3)/1000.0,axis=1)
    ph1v           = tmp1['sum'].values

    tmp            = pd.read_csv(prefixf+"/dataset_ph2.csv")
    tmp1           = tmp[tmp['t2'] > 0]
    tmp1['sum']    = tmp1.apply(lambda row : (row.t1 + row.t2 + row.t3)/1000.0,axis=1)
    ph2v           = tmp1['sum'].values

    tmp            = pd.read_csv(prefixf+"/dataset_ph3.csv")
    tmp1           = tmp[tmp['t2'] > 0]
    tmp1['sum']    = tmp1.apply(lambda row : (row.t1 + row.t2 + row.t3)/1000.0,axis=1)
    ph3v           = tmp1['sum'].values

    tmp            = pd.read_csv(prefixf+"/dataset_ph4.csv")
    tmp1           = tmp[tmp['t2'] > 0]
    tmp1['sum']    = tmp1.apply(lambda row : (row.t1 + row.t2 + row.t3)/1000.0,axis=1)
    ph4v           = tmp1['sum'].values

    tmp            = pd.read_csv(prefixf+"/dataset_intrlv1.csv")
    tmp1           = tmp[tmp['time'] > 0]
    intrlv1        = (tmp1['time'].values)/1000.0
    
    tmp            = pd.read_csv(prefixf+"/dataset_intrlv2.csv")
    tmp1           = tmp[tmp['time'] > 0]
    intrlv2        = (tmp1['time'].values)/1000.0

    tmp            = pd.read_csv(prefixf+"/dataset_crcturbo.csv")
    tmp1           = tmp[tmp['time'] > 0]
    crcturbo       = (tmp1['time'].values)/1000.0
    #elapsed = timeit.default_timer() - start_time
    #print("Dataset retrieved in %f seconds"%(elapsed))

    return (ph1v,ph2v,intrlv1,ph3v,intrlv2,ph4v,crcturbo)

def test_above():
    prefixf = "Validation-Datasets/alloc_prbs-90_cores-16"
    (ph1v,ph2v,intrlv1,ph3v,intrlv2,ph4v,crcturbo) = get_dataset_phase(prefixf)
    total = 2*(np.mean(ph1v)+np.mean(ph2v)+np.mean(ph3v)+np.mean(intrlv1))+np.mean(intrlv2)+np.mean(ph4v)+np.mean(crcturbo)
    print(total)
    ##plt.hist(ph1v,bins=BINS,density=True)
    #plt.hist(ph4v,bins=BINS,density=True)
    #plt.xlim(300,1250)
    #plt.ylim(0,0.025)
    #plt.title("Mean : "+str(np.mean(ph4v))+", Std : "+str(np.std(ph4v)))
    #plt.savefig("test-ph4-pdf4.pdf")
    #tmp = pd.read_csv(prefixf+"/dataset_sf.csv")
    #t2  = tmp["EcecutionTime"].values
    #print(t2)
    #print(np.mean(ph1v))
    #print(np.std(ph1v))
    #,bins=BINS)

def compare_pdfs():
    """ 
        Compare the estimated and 
        actual probability distribution 
        of the subframe execution time.

        This is done for a particular workload
        and allocation case.
    """
    # The pdfs are already computed, load them from the pdf db file
    ph1_table = np.load("pdf-risk-db3/ph1db.npy")
    ph2_table = np.load("pdf-risk-db3/ph2db.npy")
    ph3_table = np.load("pdf-risk-db3/ph3db.npy")
    ph4_table = np.load("pdf-risk-db3/ph4db.npy")
    i1_table  = np.load("pdf-risk-db3/i1db.npy")
    i2_table  = np.load("pdf-risk-db3/i2db.npy")
    crc_table = np.load("pdf-risk-db3/crcdb.npy")
    (w,m)     = (99,1) # Indices for the PDF table
    i1        = etPDF(i1_table[w,:],BINS)
    i2        = etPDF(i2_table[w,:],BINS)
    crc       = etPDF(crc_table[w,:],BINS)

    # Retrieve the PDFs of all the phases
    pdf1 = etPDF(ph1_table[w,m,:],BINS)
    pdf2 = etPDF(ph2_table[w,m,:],BINS)
    pdf3 = etPDF(ph3_table[w,m,:],BINS)
    pdf4 = etPDF(ph4_table[w,m,:],BINS)

    # Combine the PDFs using convolution ()
    sfet = pdf1 + pdf1 + pdf2 + pdf2 + i1 + i1 + pdf3 + pdf3 + i2 + pdf4 + crc

    # Use another method to estimate the PDF
    # --------------------------------------
    # How is different from the previous one ?
    # 1. I do not use the precomputed PDF tables
    # 2. All the tasks of a phase are drawn only once. 
    #    Basically I am more interested in the phase 
    #    level profiling.
    (ph1v,ph2v,intrlv1,ph3v,intrlv2,ph4v,crcturbo) = \
    get_dataset_phase("/home/amaity/Desktop/Datasets/ptss-raw-execution-data/ecolab-knl-2018-10-28/alloc_prbs-100_cores-2/")
    ph1t   = [ph1v[i] for i in range(0,len(ph1v),16)]
    ph1s1  = np.array([ph1t[i] for i in range(0,8000) if i%2 == 0])
    ph1s2  = np.array([ph1t[i] for i in range(0,8000) if i%2 == 1])
    ph2t   = [ph2v[i] for i in range(0,len(ph2v),12)]
    ph2s1  = np.array([ph2t[i] for i in range(0,8000) if i%2 == 0])
    ph2s2  = np.array([ph2t[i] for i in range(0,8000) if i%2 == 1])
    intrlv1s1   = np.array([intrlv1[i] for i in range(0,8000) if i%2 == 0])
    intrlv1s2   = np.array([intrlv1[i] for i in range(0,8000) if i%2 == 0])
    ph3t   = [ph3v[i] for i in range(0,len(ph3v),24)]
    ph3s1  = np.array([ph3t[i] for i in range(0,8000) if i%2 == 0])
    ph3s2  = np.array([ph3t[i] for i in range(0,8000) if i%2 == 1])
    ph4    = np.array([ph4v[i] for i in range(0,len(ph4v),24)])

    ph1s1_pdf,_ = np.histogram(ph1s1,bins=BINS,density=True)
    ph1s2_pdf,_ = np.histogram(ph1s2,bins=BINS,density=True)
    ph2s1_pdf,_ = np.histogram(ph2s1,bins=BINS,density=True)
    ph2s2_pdf,_ = np.histogram(ph2s2,bins=BINS,density=True)
    i1s1_pdf,_  = np.histogram(intrlv1s1,bins=BINS,density=True)
    i1s2_pdf,_  = np.histogram(intrlv1s2,bins=BINS,density=True)
    ph3s1_pdf,_ = np.histogram(ph3s1,bins=BINS,density=True)
    ph3s2_pdf,_ = np.histogram(ph3s2,bins=BINS,density=True)
    i2_pdf,_    = np.histogram(intrlv2,bins=BINS,density=True)
    ph4_pdf,_   = np.histogram(ph4,bins=BINS,density=True)
    crc_pdf,_   = np.histogram(crcturbo,bins=BINS,density=True)

    ph1s1pdf    = etPDF(ph1s1_pdf,BINS)
    ph1s2pdf    = etPDF(ph1s2_pdf,BINS)
    ph2s1pdf    = etPDF(ph2s1_pdf,BINS)
    ph2s2pdf    = etPDF(ph2s2_pdf,BINS)
    i1s1pdf     = etPDF(i1s1_pdf,BINS)
    i1s2pdf     = etPDF(i1s2_pdf,BINS)
    ph3s1pdf    = etPDF(ph3s1_pdf,BINS)
    ph3s2pdf    = etPDF(ph3s2_pdf,BINS)
    i2pdf       = etPDF(i2_pdf,BINS)
    ph4pdf      = etPDF(ph4_pdf,BINS)
    crcpdf      = etPDF(crc_pdf,BINS)

    sfet2       = ph1s1pdf + ph1s2pdf + ph2s1pdf + ph2s2pdf + i1s1pdf + i1s2pdf + ph3s1pdf + ph3s2pdf + i2pdf + ph4pdf + crcpdf

    # File
    pf1   = "/home/amaity/Desktop/Datasets/ptss-raw-execution-data/ecolab-knl-2018-10-28/alloc_prbs-100_cores-2/dataset_sf.csv"
    pf2   = "/home/amaity/Desktop/Datasets/ptss-raw-execution-data/ecolab-knl-2018-10-19/alloc_prbs-100_cores-2/dataset_sf.csv"
    vals1 = pd.read_csv(pf1)
    vals2 = pd.read_csv(pf2)
    tmp1  = (vals1["ExecutionTime"].values)*1000
    tmp2  = (vals2["ExecutionTime"].values)*1000
    plt.hist(tmp1,bins=len(BINS),density=True,label="Oct-28")
    plt.hist(tmp2,bins=len(BINS),density=True,label="Oct-19")

    # Plot the estimated
    plt.plot(sfet.xp,sfet.pdf(sfet.xp),label="Estimated")
    plt.plot(sfet2.xp,sfet2.pdf(sfet.xp),label="Estimated-2")
    plt.xlim(22000,23500)
    plt.xlabel("Execution Time (in ms)")
    plt.ylabel("PDF")
    plt.legend()
    plt.title("Execution Time Distribution")
    plt.savefig("comparison-prb100-cores2.pdf")
    
# Workload,Allocation vs Distribution Tables
def dump_pdf_table():
    shape         = (W,M,len(BINS)-1)
    ph1_table     = np.full(shape,-1.2)       # Table to store the distribution of Phase 1 Execution Times
    ph2_table     = np.full(shape,-1.2)       # Table to store the distribution of Phase 2 Execution Times
    ph3_table     = np.full(shape,-1.2)       # Table to store the distribution of Phase 3 Execution Times
    ph4_table     = np.full(shape,-1.2)       # Table to store the distribution of Phase 4 Execution Times
    intrlv1_table = np.full((W,len(BINS)-1),-1.2)
    intrlv2_table = np.full((W,len(BINS)-1),-1.2)
    crc_table     = np.full((W,len(BINS)-1),-1.2)

    prefix        = "/home/amaity/Desktop/Datasets/ptss-raw-execution-data/ecolab-knl-2018-10-28"
    
    for w in range(89,100):
        for m in range(1,M):
            start_time     = timeit.default_timer()
            
            mfile          = prefix+"/alloc_prbs-"+str(w+1)+"_cores-"+str(m+1)
            
            ph1v,ph2v,intrlv1,ph3v,intrlv2,ph4v,crcturbo = \
            get_dataset_phase(mfile)

            # Compute and Store the PDFs
            ph1_table[w,m,:],_ = np.histogram(ph1v,bins=BINS,density=True)
            ph2_table[w,m,:],_ = np.histogram(ph2v,bins=BINS,density=True)
            ph3_table[w,m,:],_ = np.histogram(ph3v,bins=BINS,density=True)
            ph4_table[w,m,:],_ = np.histogram(ph4v,bins=BINS,density=True)
            if (m == M-1): #  Compute for a particular core allocation
                intrlv1_table[w,:],_ = np.histogram(intrlv1,bins=BINS,density=True)
                intrlv2_table[w,:],_ = np.histogram(intrlv2,bins=BINS,density=True)
                crc_table[w,:],_     = np.histogram(crcturbo,bins=BINS,density=True)
                #print(intrlv1)
                #plt.hist(crcturbo)
                #print(np.max(crc_table))
                # plt.hist(intrlv1,bins=BINS,density=True)
                # plt.xlim(-100,100)
                # plt.plot(intrlv1_table[w,:],BINS[1:])
                #return
            
            elapsed = timeit.default_timer() - start_time
            print("Finished PDF computation of %d-prb and %d-core case in %f seconds " % (w+1,m,elapsed))

    # Save the numpy objects in a npy file
    np.save("ph1db.npy",ph1_table)
    np.save("ph2db.npy",ph2_table)
    np.save("ph3db.npy",ph3_table)
    np.save("ph4db.npy",ph4_table)
    np.save("i1db.npy",intrlv1_table)
    np.save("i2db.npy",intrlv2_table)
    np.save("crcdb.npy",crc_table)

    #print(np.max(crc_table))

def dump_pdf_table2():
    """
        Uses method2 to dump the PDF tables,
        In this
        1. The computation time
           for all the serial phases for 
           different allocation are also stored
        2. The two slots are treated differently.
    """
    shape           = (W,M,len(BINS)-1)
    ph1s1_table     = np.full(shape,-1.2)       # Table to store the distribution of Phase 1 Execution Times
    ph2s1_table     = np.full(shape,-1.2)       # Table to store the distribution of Phase 2 Execution Times
    intrlv1s1_table = np.full(shape,-1.2)
    ph3s1_table     = np.full(shape,-1.2)       # Table to store the distribution of Phase 3 Execution Times
    ph1s2_table     = np.full(shape,-1.2)       # Table to store the distribution of Phase 1 Execution Times
    ph2s2_table     = np.full(shape,-1.2)       # Table to store the distribution of Phase 2 Execution Times
    intrlv1s2_table = np.full(shape,-1.2)
    ph3s2_table     = np.full(shape,-1.2)       # Table to store the distribution of Phase 3 Execution Times
    intrlv2_table   = np.full(shape,-1.2)
    ph4_table       = np.full(shape,-1.2)       # Table to store the distribution of Phase 4 Execution Times
    crc_table       = np.full(shape,-1.2)

    prefix        = "/home/amaity/Desktop/Datasets/ptss-raw-execution-data/ecolab-knl-2018-10-28"
    
    for w in range(89,100):
        for m in range(1,M):
            start_time     = timeit.default_timer()
            
            mfile          = prefix+"/alloc_prbs-"+str(w+1)+"_cores-"+str(m+1)
            
            ph1v,ph2v,intrlv1,ph3v,intrlv2,ph4v,crcturbo = \
            get_dataset_phase(mfile)

            # Seprate out the phase and slot boundaries
            ph1t   = [ph1v[i] for i in range(0,len(ph1v),16)]
            ph1s1  = np.array([ph1t[i] for i in range(0,8000) if i%2 == 0])
            ph1s2  = np.array([ph1t[i] for i in range(0,8000) if i%2 == 1])
            ph2t   = [ph2v[i] for i in range(0,len(ph2v),12)]
            ph2s1  = np.array([ph2t[i] for i in range(0,8000) if i%2 == 0])
            ph2s2  = np.array([ph2t[i] for i in range(0,8000) if i%2 == 1])
            intrlv1s1   = np.array([intrlv1[i] for i in range(0,8000) if i%2 == 0])
            intrlv1s2   = np.array([intrlv1[i] for i in range(0,8000) if i%2 == 0])
            ph3t   = [ph3v[i] for i in range(0,len(ph3v),24)]
            ph3s1  = np.array([ph3t[i] for i in range(0,8000) if i%2 == 0])
            ph3s2  = np.array([ph3t[i] for i in range(0,8000) if i%2 == 1])
            ph4    = np.array([ph4v[i] for i in range(0,len(ph4v),24)])

            # Compute and Store the PDFs
            ph1s1_table[w,m,:],_   = np.histogram(ph1s1,bins=BINS,density=True)
            ph2s1_table[w,m,:],_   = np.histogram(ph2s1,bins=BINS,density=True)
            intrlv1s1_table[w,:],_ = np.histogram(intrlv1s1,bins=BINS,density=True)
            ph3s1_table[w,m,:],_   = np.histogram(ph3s1,bins=BINS,density=True)
            ph1s2_table[w,m,:],_   = np.histogram(ph1s2,bins=BINS,density=True)
            ph2s2_table[w,m,:],_   = np.histogram(ph2s2,bins=BINS,density=True)
            intrlv1s2_table[w,:],_ = np.histogram(intrlv1s2,bins=BINS,density=True)
            ph3s2_table[w,m,:],_   = np.histogram(ph3s2,bins=BINS,density=True)
            intrlv2_table[w,m,:],_ = np.histogram(intrlv2,bins=BINS,density=True)
            ph4_table[w,m,:],_     = np.histogram(ph4,bins=BINS,density=True)
            crc_table[w,m,:],_     = np.histogram(crcturbo,bins=BINS,density=True)
            
            elapsed = timeit.default_timer() - start_time
            print("Finished PDF computation of %d-prb and %d-core case in %f seconds " % (w+1,m,elapsed))

    # Save the numpy objects in a npy file
    np.save("ph1s1db.npy",ph1s1_table)
    np.save("ph2s1db.npy",ph2s1_table)
    np.save("i1s1db.npy",intrlv1s1_table)
    np.save("ph3s1db.npy",ph3s1_table)
    np.save("ph1s2db.npy",ph1s2_table)
    np.save("ph2s2db.npy",ph2s2_table)
    np.save("i1s2db.npy",intrlv1s2_table)
    np.save("ph3s2db.npy",ph3s2_table)
    np.save("ph4db.npy",ph4_table)
    np.save("i2db.npy",intrlv2_table)
    np.save("crcdb.npy",crc_table)

    #print(np.max(crc_table))

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
    # prefix = "/home/amaity/Desktop/Datasets/ptss-raw-execution-data/ecolab-knl-2018-10-28/alloc_prbs-90_cores-2"
    # _,_,i1,_,i2,_,i3 = get_dataset_phase(prefix)
    # print(i1)
    #dump_pdf_table2()
    #test_etPDF()
    #testCONV()
    #print(len(BINS))
    #test_above()
    compare_pdfs()
    print("Blah")