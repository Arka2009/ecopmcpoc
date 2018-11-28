#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:04:59 2018
@author: amaity

A quick and dirty python Proof of
concept for risk based scheduling
as proposed.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import ptss_utils as ptsl
import ptss_synthpdf as psyn
import heapq
import queue
import timeit
from pprint import pprint

def dummy_ue(subframeId,dmr3,crnti):
    """ 
        Generate a random UE,
        with a given DMR requirement
    """
    #crnti = np.random.randint(low=2300,high=10000)
    prbs  = 32
    dmr   = dmr3
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

def get_et_dist(ph,w,m,marray,varray):
    """
        Obtain the et distribution
        of a phase

        INPUT
        ---------------------
        1. current phase (ph) (At this point it is assumed to be independent of phases)
        2. workload (w)
        3. number of cores (m)

        OUTPUT
        ---------------------
        0. A normal distribution (positive) with the following parameters
           i. Location (mu)
           ii. Scale (sigma^2)

        It is assumed that the distribution
        is normal.

        Please modify this distribution (with positive support)
    """
    # print("get_et_dist-start")
    # print(m)
    # print("get_et_dist-stop")
    return (marray[m-1],varray[m-1])


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
    
    def execute(self,num_cores,marray,varray):
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
                self.ticks = self.ticks + 5 # Increment by 2.
                finish     = self.ticks

                return (start,finish)
            else :
                start      = self.ticks
                
                # Capture the UE start time
                if (self.state == 0) :
                    self.start = start
    
                mu,var     = get_et_dist(self.state+1,self.prbs,num_cores,marray,varray)
                
                
                #print(mu,var)
                dist       = stats.norm(loc=mu,scale=np.sqrt(var))
                exec_time  = dist.rvs(1)[0]
                if (exec_time <= 0):
                    raise ValueError("Execution Time cannot be negative, check the distribution")
                
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

                #print("Execution Time : %f"%(finish-start))
                return (start,finish)

    def get_demand(self,marray,varray):
        """ 
            Make a demand for the number of cores
            for your next phase.

            2. Accepts a boolean value
               to dump the pdf values
        """
        # No demands when phase 
        if (self.state == ptsl.NPH or self.state == -1):
            print("No demands: Finished")
            return 0

        # Compute the PDF of remaining execution time
        slack         = self.deadline - self.ticks
        w             = self.prbs
        dmrexp        = self.risk
        risk_vs_alloc = []
        rem_states    = range(1+self.state,ptsl.NPH+1)
        mu            = 0
        var           = 0

        for m in range(1,ptsl.M) :
            mu  = 0
            var = 0
            for s in rem_states :
                t1, t2    = get_et_dist(s,w,m,marray,varray)
                mu        = mu  + t1
                var       = var + t2
            
            # Obtain the distribution
            dist = stats.norm(loc=mu,scale=np.sqrt(var))
            risk = dist.sf(slack)
            risk_vs_alloc.append(risk)
        
        # if dump :
        #     # Add code to display the slack
        #     plt.legend()
        #     plt.savefig("Risk-vs-Allocation.pdf")
        #     plt.close()

        norisk_array,   = np.where(np.array(risk_vs_alloc) <= dmrexp)
        if (np.size(norisk_array) == 0):
            demanded_cores = get_max_core(1+self.state)
        else :
            demanded_cores = 1 + norisk_array[0]

        # print("slack:%f,state:%d"%(slack,self.state))
        # print(risk_vs_alloc)
        # print(norisk_array)
        #print(demanded_cores)    
        return demanded_cores
    
    def get_curr_tick(self):
        return self.ticks

    def __str__(self):
        """ Display/Visaualization Method for Each UE """
        return "id:{sf:%d,crnti:%d},workload:{prbs:%d},timing:{arrival:%d,deadline:%f,DMR:%f,risk:%f,start:%f,ticks:%f,finish:%f},FState:%d,Cores:%d" % \
        (self.subframe,self.crnti,self.prbs,self.arrival,\
        self.deadline,self.dmr,self.risk,self.start,self.ticks,self.finish,self.state,self.alloc)
    
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

    def set_risk(self,risk):
        """
            Dynamically update
            the per-instance risk.
        """
        self.risk = risk


def execute_1_ue(ns,dmr2,constrain=False,expshrink=False):
    """ Main Execution program """

    # Generate a list of input
    dmr = dmr2 
    num_subframes = ns
    crnti1 = 2386
    crnti2 = 9983
    crnti3 = 9128
    crnti4 = 4451
    ue_list1 = [dummy_ue(i,dmr,crnti1) for i in range(0,num_subframes)]
    ue_list2 = [dummy_ue(i,dmr,crnti2) for i in range(0,num_subframes)]
    ue_list3 = [dummy_ue(i,dmr,crnti3) for i in range(0,num_subframes)]
    ue_list4 = [dummy_ue(i,dmr,crnti4) for i in range(0,num_subframes)]
    
    ue_list  = ue_list1 + ue_list2 + ue_list3 + ue_list4
    heapq.heapify(ue_list)

    # Initialize the state variables
    NZ          = 10000*num_subframes
    #print(len(ue_list)*ptsl.NPH)
    s           = np.zeros(NZ) # Start of n-th phase (irrespective of the UE/SF)
    f           = np.zeros(NZ) # Finish of the n-th phase
    m           = [0 for i in range(NZ)] # Total cores busy at beginning of time step n
    d           = [0 for i in range(NZ)] # demanded cores for the n-th phase
    a           = [0 for i in range(NZ)] # Allocated cores at beginning of time step n
    u           = [0 for i in range(NZ)] # Available/Free Cores 
    u[0]        = ptsl.M
    infq        = queue.Queue(maxsize=0) # Stores the number of computations that are inflight
    
    # Construct the distributions (For now assume all phases are balanced)
    phD              = ptsl.D2/ptsl.NPH    # Relative Deadline of Each phase
    #var              = (1e-4)*(phD)
    var              = (1e-1)*(phD)
    marray,varray    = psyn.construct_pdf(phD,ptsl.M,var,False)
  

    # "Global" Variables, whose state is maintained across subframe instances
    #risk        = np.zeros(num_subframes)
    #et          = []
    start_time  = timeit.default_timer()
    missed1      = 0
    missed2      = 0
    missed3      = 0
    missed4      = 0
    n           = 0   # The n-th phase
    while len(ue_list) > 0:
        ue = heapq.heappop(ue_list)

        # Execute
        if expshrink :
            d[n] = ue.get_demand(marray,varray)
        else :
            d[n] = 16 # Statically Allocate the cores
        s[n] = ue.get_curr_tick()
        m[n] = 0

        
        # Constrain the number of cores available
        num_inflight = infq.qsize()
        
        # Compute the number of inflight instructions
        while num_inflight > 0:
            f2,a2 = infq.get(block=False)
            if (s[n] <= f2) :
                m[n] = m[n] + a2   # Decrement the number of available cores
                infq.put((f2,a2))
            num_inflight = num_inflight-1
    
        u[n]    = ptsl.M - m[n]

        # Cap the number of demanded cores to the total available (in a constrained system)
        if constrain :
            if not expshrink :
                if d[n] > u[n] :
                    a[n] = 0      # No cores to be allocated when demand is greater than is available
                else :
                    a[n] = d[n]
            else :
                a[n] = np.min([d[n],u[n]])
        else :
            a[n] = d[n]    
        
        _,f[n]  = ue.execute(a[n],marray,varray)
        infq.put((f[n],a[n])) # Enque the (potentially) inflight computation

        # print("Finished Processing...")
        # print(ue)
        # Store it back or discard it.
        if (ue.state == ptsl.NPH or ue.state == -1):
            if (ue.state == -1):
                if ue.crnti == crnti1 :
                    missed1 = missed1 + 1
                elif ue.crnti == crnti2 :
                    missed2 = missed2 + 1
                elif ue.crnti == crnti3 :
                    missed3 = missed3 + 1
                if ue.crnti == crnti4 :
                    missed4 = missed4 + 1
            # else :
            #     et.append(ue.finish-ue.start)
            
            print(ue)
        else :
            heapq.heappush(ue_list,ue)
    
        #print("m[%d]:%d,d[%d]:%d,a[%d]:%d"%(n,m[n],n,d[n],n,a[n]))
        #print("s[%d]:%d,f[%d]:%d"%(n,s[n],n,f[n]))
        n = n+1
    
    elapsed = timeit.default_timer() - start_time
    print("\n\n")
    print("Simulation Time : %fs,DMR-App1(observed) : %f"%(elapsed,missed1/num_subframes))
    print("Simulation Time : %fs,DMR-App2(observed) : %f"%(elapsed,missed2/num_subframes))
    print("Simulation Time : %fs,DMR-App3(observed) : %f"%(elapsed,missed3/num_subframes))
    print("Simulation Time : %fs,DMR-App4(observed) : %f"%(elapsed,missed4/num_subframes))
    return (s,f,m,n)

  
def main_test():
    NS   = 500
    s,f,m,n = execute_1_ue(NS,0.23,constrain=True,expshrink=True)
    max_demand = np.max(m)
    sum2 = 0
    for t in range(1,n):
        delt = s[t] - s[t-1]
        sum2 = sum2 + delt*m[t]
    #print("t:%d,n:%d"%(t,n))
    avg_demand = sum2/(f[t]-s[0]) 
    print("Demand (Max:%d,Avg:%f)"%(max_demand,avg_demand))
    # print(all(s[i] <= s[i+1] for i in range(len(s[0:n])-1)))
    # for t in range(n):
    #      print(s[t])
    # plt.plot(s[0:n],m[0:n],label="Demand (Max:%d,Avg:%f)"%(max_demand,avg_demand))
    # plt.xlabel("Time")
    # plt.ylabel("Available Cores")
    # plt.title("Resource Usage Profile (To meet the DMR)")
    # plt.legend()
    # plt.savefig("rusage-profile.pdf")

        
if __name__=="__main__":
    main_test()