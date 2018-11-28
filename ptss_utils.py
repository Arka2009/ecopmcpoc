#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:10:50 2018

@author: amaity
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import scipy
import scipy.signal
import timeit
import io
from enum import Enum

# Global Constants
NUMBINS   = 2000                    # Used Internally by CDF and PDFs functions
T         = 8000                    # Unit unspecified
D         = 2550                    # Deadline (relative to arrival time)
D2        = 2500                    # Used for synthetically generating the execution time distribution
M         = 64                      # Total number of Cores available in the system
W         = 100                     # Total number of PRBs 
TD        = np.arange(0,D,0.1)      # Discretized Time Steps
NPH       = 5                       # Total number of phaes (assumed to be same for all the UE)

##(Check 16000/12000/10000)
DELTA     = 1.0                         # Execution Time Granularity (1us)
BINS      = np.arange(0,25000,DELTA)    # Bin-edges of the histogram/pdf (Bin-width = 1us) of each phase execution times
Xp        = np.array([(BINS[i]+BINS[i+1])/2 for i in range(0,len(BINS)-1)]) # Mid Point of Each Bins

class UEExecState(Enum):
    """ 
        The execution is broken down into multiple states.
        A state may either denote a phase where multiple 
        parallel control flow might exist concurrently or
        it might also denote a waiting state, when the UE
        has arrived but still waiting to be assigned to be 
        assigned a 
        set of cores.

        A state like PH1S1 means the UE has finished executing
        PH1S1 and waiting for next states.

        NOTE : Please Generalize it to N-parallel phases 
    """
    INVALID  = 15        # Invalid State, Neither arrived nor is its workload known
    FUTURE   = 12        # The UE has not arrived but its workload is known.
    WAITING  = 0
    PH1S1    = 1
    PH2S1    = 2
    PH3S1    = 3
    PH1S2    = 5
    PH2S2    = 6
    PH3S2    = 7
    FINISHED = 9
    DROPPED  = 10

def next_state(curr_state):
    """ 
        Returns the next state.
        Try to make it a class method
    """
    ns = {
        UEExecState.WAITING  : UEExecState.PH1S1,    # Transition Marks the end of PH1S1 
        UEExecState.PH1S1    : UEExecState.PH2S1,    # Transition Marks the end of PH2S1 
        UEExecState.PH2S1    : UEExecState.PH3S1,    # Transition Marks the end of PH3S1 
        UEExecState.PH3S1    : UEExecState.PH1S2,    # Transition Marks the end of PH1S2
        UEExecState.PH1S2    : UEExecState.PH2S2,    # Transition Marks the end of PH2S2 
        UEExecState.PH2S2    : UEExecState.PH3S2,    # Transition Marks the end of PH3S2 
        UEExecState.PH3S2    : UEExecState.FINISHED, # Transition Marks the end of PH4
        UEExecState.FINISHED : UEExecState.FINISHED,
        UEExecState.DROPPED  : UEExecState.DROPPED
        }

    return ns[curr_state]

def is_final(curr_state):
    """
        Check if the final state is 
        reached or not
    """
    if curr_state == UEExecState.FINISHED or curr_state == UEExecState.DROPPED :
        return True
    else:
        return False

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

def close(a,b,tol=1e-3):
    """ 
        Check if two values are nearly close 
        to each other.
    """
    if abs(a) > 0 and abs(b) > 0:
        err = abs(a-b)/np.min([abs(a),abs(b)])
        if err < tol :
            return True
        else :
            return False
    else :
        raise ValueError("Both values must be non-zero")

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
    def __init__(self,ds,support,density=False):
        if not density:
            # Compute the support
            self.inf     = np.min(support)
            self.sup     = np.max(support)
            self.support = support

            # Compute the histogram
            self.bins, self.bin_edges = np.histogram(ds,bins=support,density=True)
            self.delta                = self.bin_edges[1]-self.bin_edges[0]       # Bin spacing is constant

            # CDF values at selected x
            cdf_vals       = np.cumsum(self.bins)
            self.Fp        = cdf_vals/cdf_vals[-1]
            self.xp        = np.array([(self.bin_edges[i]+self.bin_edges[i+1])/2 for i in range(0,len(self.bin_edges)-1)]) # The x coordinates where pdf and cdf functions are given 
        else :
            # Treat the ds a the pdf with the given support
            if (len(ds) + 1 == len(support)):
                # Compute the support
                self.inf     = np.min(support)
                self.sup     = np.max(support)
                self.support = support
                
                self.bins      = ds
                self.bin_edges = support
                self.delta     = self.bin_edges[1]-self.bin_edges[0]       # Bin spacing is constant

                # CDF values at selected x
                cdf_vals       = np.cumsum(self.bins)
                self.Fp        = cdf_vals/cdf_vals[-1]
                self.xp        = np.array([(self.bin_edges[i]+self.bin_edges[i+1])/2 for i in range(0,len(self.bin_edges)-1)]) # The x coordinates where pdf and cdf functions are given 
            else:
                print(len(ds) + 1)
                print(len(support))
                raise ValueError("Support and PDF vector mismatch !!!")
    




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
        if (close(self.delta,other.delta)):
            conv_pdf     = np.convolve(self.bins,other.bins)*self.delta
            conv_inf     = self.inf + other.inf
            conv_sup     = self.sup + other.sup
            N            = len(conv_pdf) + 1
            conv_support = np.linspace(conv_inf,conv_sup,N)
        else :
            print(self.delta)
            print(other.delta)
            raise ValueError("Delta of both the distributions must be same")
        ret = etPDF(conv_pdf,conv_support,density=True) # Bin edges are assumed to be same
        return ret
    
    def __radd__(self,other):
        """ 
            Add two RV with a given distribution, 
            Aka perform convolution of its distribution.

            Check how to perform numerical convolution
            in scipy
        """
        if (close(self.delta,other.delta)):
            conv_pdf     = np.convolve(self.bins,other.bins)*self.delta
            conv_inf     = self.inf + other.inf
            conv_sup     = self.sup + other.sup
            N            = len(conv_pdf) + 1
            conv_support = np.linspace(conv_inf,conv_sup,N)
        else :
            print(self.delta)
            print(other.delta)
            raise ValueError("Delta of both the distributions must be same")
        ret = etPDF(conv_pdf,conv_support,density=True) # Bin edges are assumed to be same
        return ret
    
    def __str__(self):
        """ Display Summary Statistics """
        return "Estimated Distribution Mean : %f, std : %f, delta : %f" % (self.mean(),self.std(),self.delta)

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


def get_dataset_phase_filtered(prefixf):
    """ 
        Filter the dataset to remove the
        repeated values.

        Also treat the different slots
        separately
    """
    ph1v,ph2v,intrlv1,ph3v,intrlv2,ph4v,crcturbo = \
    get_dataset_phase(prefixf)

    # Separate out the phase and slot boundaries
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

    return (ph1s1,ph1s2,ph2s1,ph2s2,intrlv1s1,intrlv1s2,ph3s1,ph3s2,intrlv2,ph4,crcturbo)

    
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

def plot_time_division():
    """
        Plot the different phases 
        in a timed systems.
    """
    prefix        = "/home/amaity/Desktop/Datasets/ptss-raw-execution-data/ecolab-knl-2018-10-28"
    for w in range(96,100):
        for m in range(16,M-4):
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
            i1s1   = np.array([intrlv1[i] for i in range(0,8000) if i%2 == 0])
            i1s2   = np.array([intrlv1[i] for i in range(0,8000) if i%2 == 0])
            ph3t   = [ph3v[i] for i in range(0,len(ph3v),24)]
            ph3s1  = np.array([ph3t[i] for i in range(0,8000) if i%2 == 0])
            ph3s2  = np.array([ph3t[i] for i in range(0,8000) if i%2 == 1])
            ph4    = np.array([ph4v[i] for i in range(0,len(ph4v),24)])

            sfet   = np.mean(ph1s1) + \
                     np.mean(ph2s1) + \
                     np.mean(i1s1) + \
                     np.mean(ph3s1) + \
                     np.mean(ph1s2) + \
                     np.mean(ph2s2) + \
                     np.mean(i1s2) + \
                     np.mean(ph3s2) + \
                     np.mean(intrlv2) + \
                     np.mean(ph4) + \
                     np.mean(crcturbo)

            alloc_dist = u"ttype,start,end\n"

            start = 0.0
            end   = 0.0

            # Phase1 s1
            start = 0.0
            end   = start + np.mean(ph1s1)/sfet
            alloc_dist = alloc_dist + "PH1S1," + str(start) + "," + str(end) + "\n"

            start = end
            end   = start + np.mean(ph2s1)/sfet
            alloc_dist = alloc_dist + "PH2S1," + str(start) + "," + str(end) + "\n"

            start = end
            end   = start + np.mean(i1s1)/sfet
            alloc_dist = alloc_dist + "I1S1," + str(start) + "," + str(end) + "\n"

            start = end
            end   = start + np.mean(ph3s1)/sfet
            alloc_dist = alloc_dist + "PH3S1," + str(start) + "," + str(end) + "\n"

            start = end
            end   = start + np.mean(ph1s2)/sfet
            alloc_dist = alloc_dist + "PH1S2," + str(start) + "," + str(end) + "\n"

            start = end
            end   = start + np.mean(ph2s2)/sfet
            alloc_dist = alloc_dist + "PH2S2," + str(start) + "," + str(end) + "\n"

            start = end
            end   = start + np.mean(i1s2)/sfet
            alloc_dist = alloc_dist + "I1S2," + str(start) + "," + str(end) + "\n"

            start = end
            end   = start + np.mean(ph3s2)/sfet
            alloc_dist = alloc_dist + "PH3S2," + str(start) + "," + str(end) + "\n"

            start = end
            end   = start + np.mean(intrlv2)/sfet
            alloc_dist = alloc_dist + "I2," + str(start) + "," + str(end) + "\n"

            start = end
            end   = start + np.mean(ph4)/sfet
            alloc_dist = alloc_dist + "PH4," + str(start) + "," + str(end) + "\n"

            start = end
            end   = start + np.mean(crcturbo)/sfet
            alloc_dist = alloc_dist + "CRCTURBO," + str(start) + "," + str(end) + "\n"

            df = pd.read_csv(io.StringIO(alloc_dist))
            df["diff"] = df["end"] - df["start"]
            df.to_csv("task_test.csv")
            # Start Printing
            color = {"PH1S1":"black",\
                     "PH2S1":"crimson",\
                     "I1S1":"blue",\
                     "PH3S1":"green",\
                     "PH1S2":"black",\
                     "PH2S2" : "crimson", \
                     "I1S2":"blue",\
                     "PH3S2":"green",\
                     "I2" : "brown",\
                     "PH4" : "olive",\
                     "CRCTURBO" : "cyan",\
                     }
            fig,ax=plt.subplots(figsize=(8,2))

            labels=[]
            #print(df.groupby("ttype"))
            # for i, task in enumerate(df.groupby("core")):
            #     labels.append(task[0])
            for r in df.groupby("ttype"):
                #print(r)
                data = r[1][["start", "diff"]]
                #print(data)
                #ax.broken_barh(data.values, (i-0.4,0.8), color=color[r[0]] )
                # print(color[r[0]])
                # return
                ax.broken_barh(data.values, yrange = (0.4,0.6), color=color[r[0]] )

            # ax.set_yticks(range(len(labels)))
            # ax.set_yticklabels(labels) 
            ax.set_xlabel("time [Normalized]")
            ax.set_title("Execution Time Distribution")
            plt.tight_layout()
            plt.savefig("frac/test-prb"+str(w)+"-alloc"+str(m)+".pdf")     
            #plt.show()
            #else :
            #    raise IOError("Subframe Not Yet Scheduled")
            elapsed = timeit.default_timer() - start_time
            print("Broken Barh computation of %d-prb and %d-core case in %f seconds " % (w+1,m,elapsed))

def gen_risk_table():
    """
        Compute the risk of 
        finishing the remaining computation time
        when your at state 's' and you have 't'
        time remaining to finish the deadline
    """
    prefix        = "/home/amaity/Desktop/Datasets/ptss-raw-execution-data/ecolab-knl-2018-10-28"

    W2            = range(90,101)
    M2            = range(2,M)
    shape         = (len(W2),len(TD),len(M2))

    start_time        = timeit.default_timer()
    risktable_waiting = np.full(shape,-1.2)
    risktable_ph1s1   = np.full(shape,-1.2)
    risktable_ph2s1   = np.full(shape,-1.2)
    risktable_i1s1    = np.full(shape,-1.2)
    risktable_ph3s1   = np.full(shape,-1.2)
    risktable_ph1s2   = np.full(shape,-1.2)
    risktable_ph2s2   = np.full(shape,-1.2)
    risktable_i1s2    = np.full(shape,-1.2)
    risktable_ph3s2   = np.full(shape,-1.2)
    risktable_i2      = np.full(shape,-1.2)
    risktable_ph4     = np.full(shape,-1.2)
    #risktable_crct    = np.full(shape,-1.2)
    elapsed = timeit.default_timer() - start_time
    print("Finished Initializing Risk Tables in %f seconds " % (elapsed))


    for w in W2:
        for m in M2:
            start_time     = timeit.default_timer()
            
            mfile          = prefix+"/alloc_prbs-"+str(w)+"_cores-"+str(m)

            ph1s1,ph1s2,ph2s1,ph2s2,i1s1,i1s2,ph3s1,ph3s2,i2,ph4,crcturbo = \
            get_dataset_phase_filtered(mfile)

            # Done with Waiting about to start PH1S1
            remcomp = ph1s1 + ph2s1 + \
                      ph3s1 + i1s1  + \
                      ph1s2 + ph2s2 + \
                      ph3s2 + i1s2  + \
                      i2    + ph4   + \
                      crcturbo
            (bins2,bin_edges) = np.histogram(remcomp,bins=BINS) 
            dist = scipy.stats.rv_histogram((bins2,bin_edges))
            # Re-adjust the idices
            w2 = w-100
            m2 = m-2
            risktable_waiting[w2,:,m2] = dist.sf(TD)

            # Done PH1S1
            remcomp = ph2s1 + \
                      ph3s1 + i1s1  + \
                      ph1s2 + ph2s2 + \
                      ph3s2 + i1s2  + \
                      i2    + ph4   + \
                      crcturbo
            (bins2,bin_edges) = np.histogram(remcomp,bins=BINS) 
            dist = scipy.stats.rv_histogram((bins2,bin_edges))
            # Re-adjust the idices
            w2 = w-90
            m2 = m-2
            risktable_ph1s1[w2,:,m2] = dist.sf(TD)
            #print(np.mean(risktable_ph1s1[w,:,m]))

            # Done PH2S1
            remcomp = \
                      ph3s1 + i1s1  + \
                      ph1s2 + ph2s2 + \
                      ph3s2 + i1s2  + \
                      i2    + ph4   + \
                      crcturbo
            (bins2,bin_edges) = np.histogram(remcomp,bins=BINS) 
            dist = scipy.stats.rv_histogram((bins2,bin_edges))
            # Re-adjust the idices
            w2 = w-90
            m2 = m-2
            risktable_ph2s1[w2,:,m2] = dist.sf(TD)
            #print(np.mean(risktable_ph2s1[w,:,m]))

            # Done I1S1
            remcomp = \
                      ph3s1 + \
                      ph1s2 + ph2s2 + \
                      ph3s2 + i1s2  + \
                      i2    + ph4   + \
                      crcturbo
            (bins2,bin_edges) = np.histogram(remcomp,bins=BINS) 
            dist = scipy.stats.rv_histogram((bins2,bin_edges))
            # Re-adjust the idices
            w2 = w-90
            m2 = m-2
            risktable_i1s1[w2,:,m2] = dist.sf(TD)
            #print(np.mean(risktable_i1s1[w,:,m]))

            # Done PH3S1
            remcomp = \
                      ph1s2 + ph2s2 + \
                      ph3s2 + i1s2  + \
                      i2    + ph4   + \
                      crcturbo
            (bins2,bin_edges) = np.histogram(remcomp,bins=BINS) 
            dist = scipy.stats.rv_histogram((bins2,bin_edges))
            # Re-adjust the idices
            w2 = w-90
            m2 = m-2
            risktable_ph3s1[w2,:,m2] = dist.sf(TD)
            #print(np.mean(risktable_ph3s1[w,:,m]))

            # Done PH1S2
            remcomp = ph2s2 + \
                      ph3s2 + i1s2  + \
                      i2    + ph4   + \
                      crcturbo
            (bins2,bin_edges) = np.histogram(remcomp,bins=BINS) 
            dist = scipy.stats.rv_histogram((bins2,bin_edges))
            # Re-adjust the idices
            w2 = w-90
            m2 = m-2
            risktable_ph1s2[w2,:,m2] = dist.sf(TD)
            #print(np.mean(risktable_ph1s2[w,:,m]))

            # Done PH2S2
            remcomp = \
                      ph3s2 + i1s2  + \
                      i2    + ph4   + \
                      crcturbo
            (bins2,bin_edges) = np.histogram(remcomp,bins=BINS) 
            dist = scipy.stats.rv_histogram((bins2,bin_edges))
            # Re-adjust the idices
            w2 = w-90
            m2 = m-2
            risktable_ph2s2[w2,:,m2] = dist.sf(TD)
            #print(np.mean(risktable_ph2s2[w,:,m]))

            # Done I2S2
            remcomp = \
                      ph3s2 + \
                      i2    + ph4   + \
                      crcturbo
            (bins2,bin_edges) = np.histogram(remcomp,bins=BINS) 
            dist = scipy.stats.rv_histogram((bins2,bin_edges))
            # Re-adjust the idices
            w2 = w-90
            m2 = m-2
            risktable_i1s2[w2,:,m2] = dist.sf(TD)
            #print(np.mean(risktable_i1s2[w,:,m]))

            # Done PH3S2
            remcomp = \
                      i2    + ph4   + \
                      crcturbo
            (bins2,bin_edges) = np.histogram(remcomp,bins=BINS) 
            dist = scipy.stats.rv_histogram((bins2,bin_edges))
            # Re-adjust the idices
            w2 = w-90
            m2 = m-2
            risktable_ph3s2[w2,:,m2] = dist.sf(TD)
            #print(np.mean(risktable_ph3s2[w,:,m]))

            # Done I2
            remcomp = \
                      ph4   + \
                      crcturbo
            (bins2,bin_edges) = np.histogram(remcomp,bins=BINS) 
            dist = scipy.stats.rv_histogram((bins2,bin_edges))
            # Re-adjust the idices
            w2 = w-90
            m2 = m-2
            risktable_i2[w2,:,m2] = dist.sf(TD)
            #print(np.mean(risktable_i2[w,:,m]))

            # Done PH4
            remcomp = crcturbo
            (bins2,bin_edges) = np.histogram(remcomp,bins=BINS) 
            dist = scipy.stats.rv_histogram((bins2,bin_edges))
            # Re-adjust the idices
            w2 = w-90
            m2 = m-2
            risktable_ph4[w2,:,m2] = dist.sf(TD)
            #print(np.mean(risktable_ph4[w,:,m]))
            

            elapsed = timeit.default_timer() - start_time
            print("Computed Remaining Risk of %d-prb and %d-core case in %f seconds " % (w,m,elapsed))
    
    #print(risktable_waiting)
    np.save("rtbl-db/rtbl_wait.npy",risktable_waiting)
    np.save("rtbl-db/rtbl_ph1s1.npy",risktable_ph1s1)
    np.save("rtbl-db/rtbl_ph2s1.npy",risktable_ph2s1)
    np.save("rtbl-db/rtbl_i1s1.npy",risktable_i1s1)
    np.save("rtbl-db/rtbl_ph3s1.npy",risktable_ph3s1)
    np.save("rtbl-db/rtbl_ph1s2.npy",risktable_ph1s2)
    np.save("rtbl-db/rtbl_ph2s2.npy",risktable_ph2s2)
    np.save("rtbl-db/rtbl_i1s2.npy",risktable_i1s2)
    np.save("rtbl-db/rtbl_ph3s2.npy",risktable_ph3s2)
    np.save("rtbl-db/rtbl_i2.npy",risktable_i2)
    np.save("rtbl-db/rtbl_ph4.npy",risktable_ph4)



def plot_risk_surface(filem):
    """
        1. Plot the risk surface
            (for 4,8 and 16) core allocations
        2. Estimate a closed
            form expression of risk
            interms of w and m.
    """
    prb   = np.array(range(90,101,1))
    td    = np.array(range(0,len(TD),1))
    alloc = np.array(range(2,26,1))
    w     = prb[0]-90      # For 90 PRBs

    y     = alloc          # Y-axis is either alloc/prb
    X,Y   = np.meshgrid(y,td,indexing='ij')
    shape = (len(y),len(td))
    Z     = np.full(shape,-1.0)
    stat  = np.load(filem)

    for t in td :
        for y2 in y :
            # Access correct index of stat
            yidx = y2 - 2
            tidx = t

            # Access correct index of Z
            yidx2 = y2-2
            tidx2 = t
            # print(widx)
            # print(tidx)
            # print(m)
            Z[yidx2,tidx2] = stat[w,tidx,yidx]

    #print(Z)
    # print(stat[w,1,:]) 
    # print(stat[w,1,:].shape)
    #Plot
    fig  = plt.figure(figsize=(8,6))
    ax   = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y/10000, Z, rstride=1, cstride=1, 
                           cmap=cm.RdBu,linewidth=0, antialiased=False)
    ax.set_xlabel("Cores")
    ax.set_ylabel("Slack")
    ax.set_zlabel("Risk")
    ax.set_title("Risk Surface for (w = %s)"%str(w+90))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig("risk-w90.pdf")























if __name__=="__main__":
    np.set_printoptions(precision=3)
    # x = [np.random.normal() for u in range(0,1000000)]
    # y,u = pdf(x)
    # plt.plot(u[1:],y)
    #init_workload_risk_alloc_table()
    #compute_cumulative_risk()
    #prefix = "/home/amaity/Desktop/Datasets/ptss-raw-execution-data/ecolab-knl-2018-10-28/alloc_prbs-90_cores-2"
    # _,_,i1,_,i2,_,i3 = get_dataset_phase(prefix)
    # print(i1)
    #dump_pdf_table2()
    #test_etPDF()
    #testCONV()
    #print(len(BINS))
    #gen_risk_table()
    #print(dist)
    #plt.hist(dist,bins=BINS)
    filem = "rtbl-db/rtbl_ph1s2.npy"
    plot_risk_surface(filem)
    #compare_pdfs()
    #plot_time_division()
    print("Blah")
