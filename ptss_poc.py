#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:04:59 2018
@author: amaity

A quick and dirty python Proof of
concept for risk based scheduling
as proposed.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import ptss_utils as ptsl
from enum import Enum
import heapq
import queue
import timeit

def dummy_ue(subframeId,dmr3,criticality):
    """ 
        Generate a random UE,
        criticality = 'Hard' or
        criticality = 'Soft'
    """
    if criticality == "Hard" :
        crnti         = 2308
        prbs          = np.random.random_integers(15,20)
        dmr           = 0.01
    elif criticality == "Soft" :
        crnti         = 9987
        prbs          = 32 #np.random.random_integers(75,80)
        dmr           = dmr3
    else :
        raise IOError("unsupported Criticality")
    #layers        = 4
    deadline      = ptsl.D
    arrival_time  = (subframeId)*(ptsl.T)
    ue            = UE(crnti,subframeId,arrival_time,deadline,prbs,dmr)
    return ue

def get_max_core(ph):
    """
     Max Core Allocations in Each Phase (ph)
    """
    core = [0,16,12,24,32,48]
    if ph == 0:
        return 0
    return core[ph]

def get_et_dist(ph,w,m):
    """
        Obtain the et distribution
        of a phase

        INPUT
        ---------------------
        1. current phase (ph)
        2. workload (w)
        3. number of cores (m)

        OUTPUT
        ---------------------
        1. mean (mu)
        2. variance (sigma^2)

        It is assumed that the distribution
        is normal.
    """
    N = get_max_core(ph)
    A = np.array([[1,1],[100/N,1]])
    B = np.array([N*ptsl.D/(ptsl.NPH*ptsl.W),ptsl.D/ptsl.NPH])
    C = np.matmul(np.linalg.inv(A),B)

    #print(C)
    k1,c = C

   
    mu    = k1*(w/m) + c
    sigma = (10/100)*mu     # 10% of the mu
    # print("k1:%d,c:%d,w:%d,m:%d,mu:%f"%(k1,c,w,m,mu))
    
    # Visualize
    # dist  = stats.norm(loc=mu,scale=sigma)
    # x = dist.rvs(10000)
    # print(np.min(x))
    # print(np.max(x))
    # plt.hist(x,bins=2000,label="w:%d,m:%d"%(w,m))
    # plt.legend()

    # Return
    return (mu,sigma**2)


class UE(object):
    def __init__(self,\
                 crnti,\
                 subframe,\
                 arrival_time,\
                 rd,\
                 prbs,\
                 dmr):
        assert (dmr < 1),"DMR cannot be more than 1"
        self.crnti    = crnti                   # UE Connection ID, constant accross a connection
        self.subframe = subframe                # Subframe ID

        # Timing Information
        self.alloc    = 0                         # The number of allocated cores, (-1 denotes that the UE is dropped)
        self.state    = 0                         # Number of phases completed
        self.dmr      = dmr                       # Probability of Missing the deadline (Pr{R > D}), may not be used in first version
        self.risk     = dmr                       # Risk of missing the deadline
        self.arrival  = arrival_time              # (Absolute) Arrival Time
        self.deadline = arrival_time + rd         # (Absolute) UE Latency Deadline 
        self.ticks    = arrival_time              # (Absolute) Elapsed Time, not yet dropped/finished
        self.start    = arrival_time              # (Absolute) Start Time
        self.finish   = arrival_time              # (Absolute) Finish/Drop Time

        # Add workload information
        self.prbs     = prbs                      # Synonymous with workload                
    
    def execute(self,num_cores):
        """ 
            Execute a phase/state with allocation = num_cores,
            No preemption while exection of a phase
            
            Return Value
            ------------
            1. (Absolute) Start and Finish Time of the exceution
            
            State Changes
            -------------
            2. Changes the UEExcutionState 
            3. Update the ticks to reflect the time spent in the system
        """
        #print(self)
        if (num_cores > self.alloc) :
            self.alloc = num_cores      # Only used for debugging puposes
        if(self.state == ptsl.NPH or self.state == -1) :
            print("UE{%d,%d} is already FINISHED/DROPPED, still want to executed "%(self.subframe,self.crnti))
            return (self.ticks,self.ticks)
        else :
            if (num_cores <= 0): # No cores allocated
                start      = self.ticks
                self.ticks = self.ticks + 2 # Increment by 2.
                finish     = self.ticks

                return (start,finish)
            else :
                start      = self.ticks
                
                # Capture the UE start time
                if (self.state == 0) :
                    self.start = start
    
                mu,var     = get_et_dist(self.state+1,self.prbs,num_cores)
                dist       = stats.norm(loc=mu,scale=np.sqrt(var))
                exec_time  = dist.rvs(1)[0]
                
                # Compute the slack and decide if the UE needs to be dropped
                slack                     = self.deadline - (start + exec_time)
                if (slack < 0):
                    self.state   = -1
                    self.ticks   = self.deadline
                else :
                    self.state   = self.state + 1
                    self.ticks   = self.ticks + exec_time
                
                finish           = self.ticks
                
                # Capture the UE Finish Time
                if(self.state == ptsl.NPH or self.state == -1):
                    self.finish = finish

                return (start,finish)

    def get_demand(self):
        """ 
            Make a demand for the number of cores
            for your next phase.
        """
        # No demands when phase 
        if (self.state == ptsl.NPH or self.state == -1):
            print("No demands: Finished")
            return 0
        
        # Compute the PDF of remaining execution time
        slack         = self.deadline - self.ticks
        w             = self.prbs
        dmrexp        = self.dmr
        risk_vs_alloc = []
        rem_states    = range(1+self.state,ptsl.NPH+1)
        mu            = 0
        var           = 0

        for m in range(1,ptsl.M) :
            mu  = 0
            var = 0
            for s in rem_states :
                t1, t2    = get_et_dist(s,w,m)
                mu        = mu  + t1
                var       = var + t2
            
            # Obtain the distribution
            dist = stats.norm(loc=mu,scale=np.sqrt(var))
            risk = dist.sf(slack)
            risk_vs_alloc.append(risk)

        norisk_array,   = np.where(np.array(risk_vs_alloc) <= dmrexp)
        if (np.size(norisk_array) == 0):
            demanded_cores = get_max_core(1+self.state)
        else :
            demanded_cores = norisk_array[0]
            
        # Display and Debug
        # print(demanded_cores)
        # print("\n\n")
        # print("\n\n")
        # plt.legend()
        # plt.savefig("rem-%d.pdf"%self.state)
        # plt.close()
        return demanded_cores
    
    def get_curr_tick(self):
        return self.ticks

    def __str__(self):
        """ Display/Visaualization Method for Each UE """
        return "id:{sf:%d,crnti:%d},workload:{prbs:%d},timing:{arrival:%d,deadline:%f,DMR:%f,start:%f,ticks:%f,finish:%f},FState:%d,Cores:%s" % \
        (self.subframe,self.crnti,self.prbs,self.arrival,\
        self.deadline,self.dmr,self.start,self.ticks,self.finish,self.state,self.alloc)
    
    # Comparable interface needed for sort/priority queue routines
    def __lt__(self,other):
        if (self.ticks == other.ticks) :
            return (self.crnti < other.crnti)
        else :
            return (self.ticks < other.ticks)
    
    def __le__(self,other):
        if (self.ticks == other.ticks) :
            return (self.crnti <= other.crnti)
        else :
            return (self.ticks <= other.ticks)
    
    def __eq__(self,other):
        return (self.ticks == other.ticks)

    def __ne__(self,other):
        return (self.ticks != other.ticks)
    
    def __gt__(self,other):
        if (self.ticks == other.ticks) :
            return (self.crnti > other.crnti)
        else :
            return (self.ticks > other.ticks)
    
    def __ge_(self,other):
        if (self.ticks == other.ticks) :
            return (self.crnti >= other.crnti)
        else :
            return (self.ticks >= other.ticks)

    # def update_dmr(observed_dmr):
    #     """
    #         Dynamically change the DMR
    #     """

def execute_1_ue(dmr):
    """ Main Execution program """

    # Generate a list of inputs 
    num_subframes = 20
    ue_list1 = [dummy_ue(i,dmr,"Hard") for i in range(0,num_subframes)]
    ue_list2 = [] #[dummy_ue(i,dmr,"Soft") for i in range(0,num_subframes)]
    ue_list  = ue_list1 + ue_list2
    heapq.heapify(ue_list)

    # Initialize the state variables
    NZ          = 1000000
    print(len(ue_list)*ptsl.NPH)
    s           = np.zeros(NZ) # Start of n-th phase (irrespective of the UE/SF)
    f           = np.zeros(NZ) # Finish of the n-th phase
    m           = np.zeros(NZ) # Available cores at the beginning of n-th phase
    m[0]        = ptsl.M
    d           = np.zeros(NZ) # demanded cores for the n-th phase
    a           = np.zeros(NZ) # Allocated cores
    infq        = queue.Queue(maxsize=0) # Stores the number of computations that are inflight

    start_time  = timeit.default_timer()
    missed      = 0
    n           = 0   # The n-th phase
    while len(ue_list) > 0:
        ue = heapq.heappop(ue_list)

        # Execute
        #print(ue)
        d[n] = ue.get_demand()
        s[n] = ue.get_curr_tick()

        m[n] = ptsl.M
        num_inflight = infq.qsize()
        
        # Compute the number of inflight instructions
        while num_inflight > 0:
            f2,a2 = infq.get(block=False)
            if (s[n] <= f2) :
                m[n] = m[n] - a2   # Decrement the number of available cores
                infq.put((f2,a2))
            num_inflight = num_inflight-1
        
        #print("a[n]=%d"%np.min([d[n],m[n]]))
            
        # Cap the number of demanded cores to the total available
        a[n] = np.min([d[n],m[n]])
        _,f[n]  = ue.execute(a[n])
        infq.put((f[n],a[n])) # Enque the (potentially) inflight computation


        # Store it back or discard it.
        if (ue.state == ptsl.NPH or ue.state == -1):
            if (ue.state == -1):
                missed = missed + 1
            print(ue)
        else :
            heapq.heappush(ue_list,ue)
    
        #print("m[%d]:%d,d[%d]:%d,a[%d]:%d"%(n,m[n],n,d[n],n,a[n]))
        #print("s[%d]:%d,f[%d]:%d"%(n,s[n],n,f[n]))
        n = n+1
    
    elapsed = timeit.default_timer() - start_time
    print("\n\nSimulation Time : %fs,DMR(observed) : %f"%(elapsed,missed/num_subframes))
    return (s,m,n)

  
def main_test():
    s,m,n = execute_1_ue(0.3)
    # numT = np.array(list(range(0,((total_time_steps-1)*(ptsl.T))+ptsl.D,2)))
    # m    = []
    # for t in numT :
    #     m.append(ptsl.get_occupied_cores(alloc_stats,t))
        
    # print(s)
    # print(m)
    max_demand = ptsl.M - np.min(m)
    plt.plot(s[0:n],m[0:n],label="Maximum Demand : %d"%max_demand)
    plt.xlabel("Time")
    plt.ylabel("Available Cores")
    plt.title("Resource Usage Profile (To meet the DMR)")
    plt.legend()
    plt.savefig("rusage-profile.pdf")

        
if __name__=="__main__":
    main_test()
    #get_et_dist(3,1,1)