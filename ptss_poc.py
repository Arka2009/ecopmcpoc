#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:04:59 2018

@author: amaity
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

ph1_table = np.load("analyses/ph1db.npy")
ph2_table = np.load("analyses/ph2db.npy")
ph3_table = np.load("analyses/ph3db.npy")
ph4_table = np.load("analyses/ph4db.npy")
waiting_remcomp_db = np.load("analyses/ph1231234db_cum.npy")
ph1s1_remcomp_db = np.load("analyses/ph231234db_cum.npy")
ph2s1_remcomp_db = np.load("analyses/ph31234db_cum.npy")
ph3s1_remcomp_db = np.load("analyses/ph1234db_cum.npy")
ph1s2_remcomp_db = np.load("analyses/ph234db_cum.npy")
ph2s2_rempcom_db = np.load("analyses/ph34db_cum.npy")
ph3s2_remcomp_db = np.load("analyses/ph4db_cum.npy")

class UEExecState(Enum):
    """ 
        The execution is broken down into multiple states.
        A state may either denote a phase where multiple 
        parallel control flow might exist concurrently or
        it might also denote a waiting state, when the UE
        has arrived but still waiting to be assigned to be 
        assigned a 
        set of cores.

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

def dummy_ue(subframeId,criticality):
    """ 
        Generate a random UE,
        criticality = 'Hard' or
        criticality = 'Soft'
    """
    if criticality == "Hard" :
        crnti         = 2308
        prbs          = np.random.random_integers(10,20)
        dmr           = 0.01
    elif criticality == "Soft" :
        crnti         = 9987
        prbs          = np.random.random_integers(80,90)
        dmr           = 0.2
    else :
        raise IOError("unsupported Criticality")
    layers        = 4
    deadline      = ptsl.D
    arrival_time  = (subframeId)*(ptsl.T)
    ue            = UE(crnti,subframeId,arrival_time,deadline,prbs,layers,dmr)
    #subframeId    = subframeId + 1
    #print(ue)
    return ue

class UE(object):
    def __init__(self,\
                 crnti,\
                 subframe,\
                 arrival_time,\
                 deadline,\
                 prbs,\
                 layers,\
                 dmr):
        assert (dmr < 1),"DMR cannot be more than 1"
        self.crnti    = crnti                   # UE Connection ID, constant accross a connection
        self.subframe = subframe                # Subframe ID

        # Timing Information
        self.alloc    = 0                         # The number of allocated cores
        self.state    = UEExecState.WAITING       # The different states of execution
        self.dmr      = dmr                       # Probability of Missing the deadline (Pr{R > D}), may not be used in first version
        self.risk     = dmr                       # Risk of missing the deadline
        self.arrival  = arrival_time              # Arrival Time
        self.deadline = arrival_time + deadline   # UE Latency Deadline (Absolute)
        self.ticks    = arrival_time              # Elapsed Time (Absolute), but not yet dropped
        self.start    = arrival_time              # Start Time (Absolute)
        self.finish   = arrival_time              # Finish/Drop Time (Absolute)

        # Add workload information
        self.prbs     = prbs
        self.layers   = layers                            
    
    def execute(self,num_cores):
        """ 
            Execute a phase with allocation = num_cores,
            
            Return Value
            ------------
            1. (Absolute) Start and Finish Time of the exceution
            
            State Changes
            -------------
            2. Changes the UEExcutionState 
            3. Update the ticks to reflect the time spent in the system
        """
        if (num_cores > self.alloc) :
            self.alloc = num_cores      # Only used for debugging puposes
        #print("Entering in state = %s, ticks = %f" % (self.state.name,self.ticks))
        if(self.state == UEExecState.DROPPED or self.state == UEExecState.FINISHED) :
            print("Hey the UE{%d,%d} is already FINISHED/DROPPED, still want to executed "%(self.subframe,self.crnti))
            return (self.ticks,self.ticks)
        else :
            if (num_cores <= 0): # No cores allocated
                self.state = UEExecState.WAITING
                start      = self.ticks
                self.ticks = self.ticks + 10 # Increment by 5us
                finish     = self.ticks

                print("No cores allocate for UE{%d,%d} during this phase-%s during [%d,%d]"\
                %(self.subframe,self.crnti,self.state.name,start,finish))
                return (start,finish)
            else :
                start                     = self.ticks
                
                # Capture the UE start time
                if (self.state == UEExecState.WAITING) :
                    self.start = start
                
                # Compute the execution time
                bins      = list(range(0,12000))
                # First obtain the appropriate distribution
                if (self.state == UEExecState.WAITING or self.state == UEExecState.PH3S1) :
                    pdf           = ph1_table[self.prbs,num_cores,:]
                    xp, cdf       = ptsl.cdf2(pdf,bins)     
                
                if (self.state == UEExecState.PH1S1 or self.state == UEExecState.PH1S2) :
                    pdf = ph2_table[self.prbs,num_cores,:]
                    xp, cdf       = ptsl.cdf2(pdf,bins)
                
                if (self.state == UEExecState.PH2S1 or self.state == UEExecState.PH2S2) :
                    pdf = ph3_table[self.prbs,num_cores,:]
                    xp, cdf       = ptsl.cdf2(pdf,bins)
                
                if (self.state == UEExecState.PH3S2) :
                    pdf = ph4_table[self.prbs,num_cores,:]
                    xp, cdf       = ptsl.cdf2(pdf,bins)
                
                # Draw a random sample from the above distribution
                exec_time         = ptsl.inv_transform_sampling(cdf,xp[1:])

                # Compute the slack and decide if the UE needs to be dropped
                slack                     = self.deadline - (start + exec_time)
                if (slack <= 0):
                    self.state            = UEExecState.DROPPED
                    self.ticks            = self.deadline
                else :
                    self.state            = next_state(self.state)
                    self.ticks            = self.ticks + exec_time
                
                finish                    = self.ticks
                
                # Capture the UE Finish Time
                if(self.state == UEExecState.DROPPED or self.state == UEExecState.FINISHED):
                    self.finish = finish

                return (start,finish)

    def execute2(self,num_cores):
        """ 
            Dummy execute method only for testing
        """
        #print("Entering in state = %s, ticks = %f" % (self.state.name,self.ticks))
        if(self.state == UEExecState.DROPPED or self.state == UEExecState.FINISHED) :
            print("Hey the UE{%d,%d} is already FINISHED/DROPPED, still want to executed "%(self.subframe,self.crnti))
            return (self.ticks,self.ticks)
        else :
            if (num_cores <= 0): # No cores allocated
                self.state = UEExecState.WAITING
                start      = self.ticks
                self.ticks = self.ticks + 0.05
                finish     = self.ticks

                #print("No cores allocate for UE{%d,%d} during this phase-%s during [%d,%d]"\
                #%(self.subframe,self.crnti,self.state.name,start,finish))
                return (start,finish)
            else :
                start             = self.ticks
                exec_time         = 100

                # Compute the slack and decide if the UE needs to be dropped
                slack                     = self.deadline - (start + exec_time)
                if (slack <= 0):
                    self.state            = UEExecState.DROPPED
                    self.ticks            = self.deadline
                else :
                    self.state            = next_state(self.state)
                    self.ticks            = self.ticks + exec_time
                
                finish                    = self.ticks
                
                # Capture the UE Finish Time
                if(self.state == UEExecState.DROPPED or self.state == UEExecState.FINISHED):
                    self.finish = finish

                return (start,finish)

    def get_demand(self):
        """ Make it more fancy """
        # Compute the PDF of remaining execution time
        
        slack  = self.deadline - self.ticks
        w      = self.prbs
        dmrexp = self.dmr
        risk_vs_alloc = [1.0]

        if(self.state == UEExecState.WAITING):
            for c in range(1,ptsl.M) :
                xp   = range(0,83987)
                Fp   = waiting_remcomp_db[w,c,:]
                risk = (1-np.interp(slack,xp,Fp))
                risk_vs_alloc.append(risk)
            
            # TODO : Is there a more *elegant* way of this
            try :
                demanded_cores = np.min(np.where(np.array(risk_vs_alloc) <= dmrexp))
            except ValueError:
                demanded_cores = 0
            
            return demanded_cores


        if(self.state == UEExecState.PH1S1):
            for c in range(1,ptsl.M) :
                xp   = range(0,71989)
                Fp   = ph1s1_remcomp_db[w,c,:]
                risk = (1-np.interp(slack,xp,Fp))
                risk_vs_alloc.append(risk)
            
            # TODO : Is there a more *elegant* way of this
            try :
                demanded_cores = np.min(np.where(np.array(risk_vs_alloc) <= dmrexp))
            except ValueError:
                demanded_cores = 0
            return demanded_cores
        
        elif(self.state == UEExecState.PH2S1):
            for c in range(1,ptsl.M) :
                xp   = range(0,59991)
                Fp   = ph2s1_remcomp_db[w,c,:]
                risk = (1-np.interp(slack,xp,Fp))
                risk_vs_alloc.append(risk)
                
            # TODO : Is there a more *elegant* way of this
            try :
                demanded_cores = np.min(np.where(np.array(risk_vs_alloc) <= dmrexp))
            except ValueError:
                demanded_cores = 0
            return demanded_cores
        
        elif(self.state == UEExecState.PH3S1):
            for c in range(1,ptsl.M) :
                xp   = range(0,47993)
                Fp   = ph3s1_remcomp_db[w,c,:]
                risk = (1-np.interp(slack,xp,Fp))
                risk_vs_alloc.append(risk)
            
            # TODO : Is there a more *elegant* way of this
            try :
                demanded_cores = np.min(np.where(np.array(risk_vs_alloc) <= dmrexp))
            except ValueError:
                demanded_cores = 0
            return demanded_cores
        

        elif(self.state == UEExecState.PH1S2):
            for c in range(1,ptsl.M) :
                xp   = range(0,35995)
                Fp   = ph1s2_remcomp_db[w,c,:]
                risk = (1-np.interp(slack,xp,Fp))
                risk_vs_alloc.append(risk)
            
            # TODO : Is there a more *elegant* way of this
            try :
                demanded_cores = np.min(np.where(np.array(risk_vs_alloc) <= dmrexp))
            except ValueError:
                demanded_cores = 0
            return demanded_cores
        

        elif(self.state == UEExecState.PH2S2):
            for c in range(1,ptsl.M) :
                xp   = range(0,23997)
                Fp   = ph2s2_rempcom_db[w,c,:]
                risk = (1-np.interp(slack,xp,Fp))
                risk_vs_alloc.append(risk)
                
            # TODO : Is there a more *elegant* way of this
            try :
                demanded_cores = np.min(np.where(np.array(risk_vs_alloc) <= dmrexp))
            except ValueError:
                demanded_cores = 0
            return demanded_cores
        
        elif(self.state == UEExecState.PH3S2):
            for c in range(1,ptsl.M) :
                xp   = range(0,11999)
                Fp   = ph3s2_remcomp_db[w,c,:]
                risk = (1-np.interp(slack,xp,Fp))
                risk_vs_alloc.append(risk)
                
            # TODO : Is there a more *elegant* way of this
            try :
                demanded_cores = np.min(np.where(np.array(risk_vs_alloc) <= dmrexp))
            except ValueError:
                demanded_cores = 0
            return demanded_cores
        
        else : 
            raise IOError("Illegal Execution State")

    def get_demand2(self):
        """ Dummy demand only for testing """
        return 2

    def grab(self,num_cores):
        """ Grab some cores """
        return 0
    
    def release(self,num_cores):
        """ Release some core """
        return 0
    
    def update_alloc(self,num_cores):
        return 0
    
    def publish_state(self):
        #return (self.state,self.risk,self.alloc,self.ticks)
        return self.state

    def update_ticks(self,add2):
        self.ticks = self.ticks + add2
    
    def __str__(self):
        """ Display/Visaualization Method for Each UE """
        return "id:{sf:%d,crnti:%d},workload:{prbs:%d,layer:%d},timing:{arrival:%d,deadline:%f,DMR:%f,start:%f,ticks:%f,finish:%f},FState:%s,Cores:%s" % \
        (self.subframe,self.crnti,self.prbs,self.layers,self.arrival,\
        self.deadline,self.dmr,self.start,self.ticks,self.finish,self.state.name,self.alloc)
    
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

def execute_1_ue():
    """ Main Execution program """
    # Load the pdf tables for all (workload,allocation)
    # Combination
    #(ph1tbl,ph2tbl,ph3tbl,ph4tbl) = ptsl.load_risk_alloc_table()
    

    num_subframes = 20000
    ue_list1 = [dummy_ue(i,"Hard") for i in range(0,num_subframes)]
    ue_list2 = [dummy_ue(i,"Soft") for i in range(0,num_subframes)]
    ue_list  = ue_list1 + ue_list2
    numT    = num_subframes
    heapq.heapify(ue_list)

    s                    = 0
    f                    = 0
    #allocinfo_q = queue.Queue(1000000)            # Store (start,end,occupied cores)
    alloc_stats = []
    ue_stats    = []                              # Store the UE info
    while len(ue_list) > 0:
        start_time     = timeit.default_timer()
        ue = heapq.heappop(ue_list)
        
        # Free up the previously allocated cores if you start at a later time, TODO : There are some bugs in the logic rectify
#        (s,e,m)              = (0,0,0)               # (start,end,occupied cores)
#        total_utilized_cores = 0
#        while (ue.ticks > e) and (not allocinfo_q.empty()) :
#            s,e,m = (allocinfo_q.queue)[0]
#            allocinfo_q.get()
#            total_utilized_cores = total_utilized_cores - m
#            alloc_stats.append((s,e,m))
#        # How many cores are active furthermore
#        for s,e,m in allocinfo_q.queue:
#            total_utilized_cores = total_utilized_cores + m
#        print("Total active cores = %d at time = %f\n" % (total_utilized_cores,ue.ticks))
        
            
        # Execute
        demanded_cores = ue.get_demand()
        curr_alloc     = demanded_cores            # TODO : Update the policy if there are lesser number of cores
        s,f            = ue.execute(curr_alloc)
        if (ue.state == UEExecState.FINISHED or ue.state == UEExecState.DROPPED):
            ue_stats.append(ue)
        else :
            heapq.heappush(ue_list,ue)
    
        # Store the allocation information
        #allocinfo_q.put((s,f,prev_alloc))
        alloc_stats.append((s,f,curr_alloc))

        elapsed = timeit.default_timer() - start_time
        if(ue.state == UEExecState.DROPPED or ue.state == UEExecState.FINISHED):
            print(ue)
            print("Simulation Time : %f s\n\n"%elapsed)
    
    # Drain the allocinfo_q
#    while not allocinfo_q.empty():
#        s,e,m = allocinfo_q.get()
#        alloc_stats.append((s,e,m))

    return (ue_stats,alloc_stats,numT)


def test_inverse_sampling():
    pdf       = ph1_table[3,2,:]
    bins      = list(range(0,12000))
    xp, cdf       = ptsl.cdf2(pdf,bins)
    print(ptsl.inv_transform_sampling(cdf,xp[1:]))
  
def main_test():
    ue_stats,alloc_stats,total_time_steps = execute_1_ue()
    numT = list(range(0,((total_time_steps-1)*(ptsl.T))+ptsl.D,2))
    m    = []
    for t in numT :
        m.append(ptsl.get_occupied_cores(alloc_stats,t))
        
    #print(numT)
    #print(m)
    plt.plot(numT,m)
    plt.xlabel("Time (us)")
    plt.ylabel("Total Demanded Cores")
    plt.title("Core Demand Profile (To meet the DMR)")
    plt.savefig("core-demand-profile.pdf")
    
#    for s,e,m in alloc_stats:
#        print("%f\t%f\t%f"%(s,e,m))
        
if __name__=="__main__":
    main_test()
    #print(dummy_ue(898,"Hard"))
        
    