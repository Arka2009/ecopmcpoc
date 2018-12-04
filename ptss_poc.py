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
import ptss_dfspdf as pgen
import heapq
import queue
import timeit

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

        m is zero-indexed
    """
    # print("get_et_dist-start")
    # print(m)
    # print("get_et_dist-stop")
    return (marray[m],varray[m])


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
    
    def execute(self,num_cores,delta,dist):
        """ 
            Execute a phase/state with allocation = num_cores,
            No preemption while exection of a phase
            
            If the number of cores allocated is zero
            than ticks is merely updated by delta
            amount without changing the phase.

            The execution time is a sample drawn from
            dist (distribution object) which is a
            scipy.stats.rv_continuous/rv_histogram
            object. 

            Return Value
            ------------
            1. (Absolute) Start and Finish Time of the exceution
            
            State Changes
            -------------
            2. Changes the UEExcutionState 
            3. Update the ticks to reflect the time spent in the system
        """
        self.alloc = num_cores # How many cores are allocated
        if(self.state == ptsl.NPH or self.state == -1) :
            raise ValueError("Cannot Execute, already FINISHED/DROPPED")
        else :
            if (num_cores <= 0): # No cores allocated
                start      = self.ticks
                self.ticks = self.ticks + delta # Increment by 2.
                finish     = self.ticks

                return (start,finish)
            else :
                start      = self.ticks
                
                # Capture the UE start time
                if (self.state == 0) :
                    self.start = start
                
                exec_time  = dist.rvs(1)

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

                # print("UE(%d|%d|%d)et,%f"%(self.subframe,self.crnti,self.state,exec_time))
                return (start,finish)

    def get_demand(self,marray,varray,introduce_slack=True):
        """ 
            Make a demand for the number of cores
            for your next phase.

            2. Accepts a boolean value
               to dump the pdf values
        """
        # No demands when phase 
        if (self.state == ptsl.NPH or self.state == -1):
            raise ValueError("Cannot make demands for an finished/dropped UE")

        # Compute the PDF of remaining execution time
        slack         = self.deadline - self.ticks
        w             = self.prbs
        dmrexp        = self.risk
        rem_states    = range(1+self.state,ptsl.NPH+1)
        mu            = 0
        var           = 0

        # print("UE(%d,%d,%d) - Slack : %f"%(self.subframe,self.crnti,self.state,slack))
        for m in range(1,ptsl.M+1) :
            mu  = 0
            var = 0
            for s in rem_states :
                t1, t2    = get_et_dist(s,w,m,marray,varray)
                mu        = mu  + t1
                var       = var + t2
            
            # Obtain the distribution
            dist = stats.norm(loc=mu,scale=np.sqrt(var))
            risk = dist.sf(slack)

            not_risky = (risk <= dmrexp) or (ptsl.close(dmrexp,risk,tol=1e-8))
            # print("risk[%d] : %f, dmr : %f, diff : %f"%(m,risk,dmrexp,abs(dmrexp-risk)*100/dmrexp))
            if not_risky :
                break
        
        
        # print("UE(%d,%d,%d) - ISF@risk-%f : %f"%(self.subframe,self.crnti,self.state,dmrexp,dist.isf(dmrexp)))
        demanded_cores = m+1
        ins_slack      = 0

        # Are you too far from the required risk and not risky
        if introduce_slack :
            if  not_risky :
                if not ptsl.close(dmrexp,risk,tol=0.005):
                    new_slack      = dist.isf(dmrexp)
                    ins_slack      = slack - new_slack
                    print("UE(%d,%d,%d) - Slack Inserted : %f, hypo-demand : %d"%(self.subframe,self.crnti,self.state,ins_slack,demanded_cores))
                    demanded_cores = 0
                else :
                    ins_slack      = 0
                    # print("Slack Inserted : %f, demand : %d"%(ins_slack,demanded_cores))
            else :
                ins_slack      = 0

            if ins_slack < 0:
                # print("Slack Inserted : %f, hypo-demand2 : %d"%(ins_slack,demanded_cores))
                raise ValueError("Why negative slack")
        else :
            ins_slack      = 0
            # print("Slack Inserted : %f, hypo-demand : %d"%(ins_slack,demanded_cores))

        # Compute the difference between the achievable
        return (ins_slack, demanded_cores)
    
    def get_curr_tick(self):
        return self.ticks

    def __str__(self):
        """ Display/Visaualization Method for Each UE """
        return "id:{sf:%d,crnti:%d},workload:{prbs:%d},timing:{arrival:%d,deadline:%f,DMR:%f,risk:%f,start:%f,ticks:%f,finish:%f},FState:%d,Cores:%d" % \
        (self.subframe,self.crnti,self.prbs,self.arrival,\
        self.deadline-self.arrival,self.dmr,self.risk,self.start,self.ticks-self.arrival,self.finish-self.arrival,self.state,self.alloc)
    
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


def execute_1_ue(ns,dmr2,constrain=False,expshrink=False,introduce_slack=True):
    """ Main Execution program """

    # Generate a list of input
    dmr = dmr2 
    num_subframes = ns
    crnti1 = 2386
    crnti2 = 9983
    crnti3 = 9128
    crnti4 = 4451
    ue_list1 = [dummy_ue(i,dmr,crnti1) for i in range(0,num_subframes)]
    ue_list2 = [] #[dummy_ue(i,dmr,crnti2) for i in range(0,num_subframes)]
    ue_list3 = [] #[dummy_ue(i,dmr,crnti3) for i in range(0,num_subframes)]
    ue_list4 = [] #[dummy_ue(i,dmr,crnti4) for i in range(0,num_subframes)]
    
    ue_list  = ue_list1 + ue_list2 + ue_list3 + ue_list4
    heapq.heapify(ue_list)

    # Initialize the state variables
    NZ          = 100*num_subframes
    s           = np.zeros(NZ) # Start of n-th phase (irrespective of the UE/SF)
    f           = np.zeros(NZ) # Finish of the n-th phase
    m           = [0 for i in range(NZ)] # Total cores busy at beginning of time step n
    d           = [0 for i in range(NZ)] # demanded cores for the n-th phase
    a           = [0 for i in range(NZ)] # Allocated cores at beginning of time step n
    u           = [0 for i in range(NZ)] # Available/Free Cores 
    u[0]        = ptsl.M
    infq        = queue.Queue(maxsize=0) # Stores the number of computations that are inflight
    
    # Construct the distributions (For now assume all phases are balanced)
    marray,varray     = pgen.get_pdf()
    distarry          = pgen.gen_hist(ptsl.NUMBINS)
    approx2           = []
    for idx in range(len(marray)) :
        approx2.append(stats.norm(loc=marray[idx],scale=np.sqrt(varray[idx])))
    print("Offline Profiling Complete\n\n")
    
  

    # "Global" Variables, whose state is maintained across subframe instances
    start_time  = timeit.default_timer()
    missed1      = 0
    missed2      = 0
    missed3      = 0
    missed4      = 0
    n            = 0   # The n-th phase
    while len(ue_list) > 0:
        ue = heapq.heappop(ue_list)

        # Execute
        if expshrink :
            ins_slack, d[n] = ue.get_demand(marray,varray,introduce_slack)
        else :
            d[n] = 11 # Statically Allocate the cores
            ins_slack = 0
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
    
        # The number of cores that are free
        u[n]    = ptsl.M - m[n] + ue.alloc 

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

        # print("UE(%d,%d,%d) - d[%d]:%d,u[%d]:%d,m[%d]:%d,a[%d]:%d,ue_alloc:%d"%(ue.subframe,ue.crnti,ue.state,n,d[n],n,u[n],n,m[n],n,a[n],ue.alloc))
        try :
            
            #_,f[n]  = ue.execute(a[n],distarry[a[n]-1])  #distarry is zero indexed
            if ins_slack == 0 :
                _,f[n]  = ue.execute(a[n],ptsl.NW,approx2[a[n]-1])
            else :
                _,f[n]  = ue.execute(a[n],ins_slack,approx2[a[n]-1])

        except IndexError :
            print("n:%d,a[n]:%d,NZ:%d"%(n,a[n],NZ))
            exit
        infq.put((f[n],a[n])) # Enque the (potentially) inflight computation
        # print("UE(%d,%d,%d) - s[%d]:%f,f[%d]:%f,a[%d]:%d,ue_alloc:%d"%(ue.subframe,ue.crnti,ue.state,n,s[n],n,f[n],n,a[n],ue.alloc))
        # print("\n")

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
        
            print(ue)
            print("-------------------------------------")
            print("\n\n\n")
        else :
            heapq.heappush(ue_list,ue)
        n = n+1
    
    elapsed = timeit.default_timer() - start_time
    print("\n")
    print("Simulation Time : %fs,DMR-App1(observed) : %f"%(elapsed,missed1/num_subframes))
    # print("Simulation Time : %fs,DMR-App2(observed) : %f"%(elapsed,missed2/num_subframes))
    # print("Simulation Time : %fs,DMR-App3(observed) : %f"%(elapsed,missed3/num_subframes))
    # print("Simulation Time : %fs,DMR-App4(observed) : %f"%(elapsed,missed4/num_subframes))
    odmr = missed1/num_subframes
    return (s,f,m,n,a,odmr,a[n-1])

  
def main_test(introduce_slack):
    NS    = 1000
    odmra = []
    utila = []
    cases = [0.01,0.05,0.1,0.25,0.5,0.75]
    for d in cases :
        #s,f,m,n,odmr,last_alloc = execute_1_ue(NS,d,constrain=True,expshrink=True,introduce_slack)
        s,f,m,n,a,odmr,last_alloc = execute_1_ue(NS,d,True,True,introduce_slack)
        odmra.append(odmr)
        max_demand = np.max(m)
        sum2 = 0
        # print("\n\nComputing the average")
        for t in range(0,n):
            # delt = s[t] - f[t]
            delt = f[t] - s[t]
            # print(delt*m[t])
            sum2 = sum2 + delt*a[t]
        # print((f[t]-s[t])*last_alloc)
        # sum2 = sum2 + (f[t]-s[t])*last_alloc
        avg_demand = sum2*100/((ptsl.M)*(f[t]-s[0]))
        print("Demand (Max:%d,Util:%f)"%(max_demand,avg_demand))
        utila.append(avg_demand)
    
    print("Finished Case - %f"%d)
    
    return (cases,odmra,utila)


        
if __name__=="__main__":
    cases1,odmra1,utila1 = main_test(introduce_slack=False)
    print("\n\n---------------------------\n\n")
    cases2,odmra2,utila2 = main_test(introduce_slack=True)

    print("\n\n")
    print(cases1)
    print(odmra1)
    print(utila1)

    print("\n\n")
    print(cases2)
    print(odmra2)
    print(utila2)
