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

# Global Constants
T = 1                       # In ms
D = 2.5                     # Deadline
M = 64                      # Total number of Cores available in the system

def compute_miss_risk(slack,num_cores,curr_state,prbs,layers,DMR):
    """ 
        Compute my risk (probability) of missing the deadline, given the
        1. The time that have been elapsed ever since the UE arrival - Deadline (aka slack)
        2. The number of cores currently allocated
        3. Computation that is remaining (aka all the states that the execution still has to traverse)
        4. The workload information (prbs and layers)
        5. Timing Requirements presented as DMR
        ----------------------------------------------------------------
        When do you invoke this ?
        1. Whenever you perform a state transition (aka at the end of execute call)
        ----------------------------------------------------------------
        Return Value
        ------------
        1. The risk of missing the deadline with the current allocation
        2. The (minimum) number of cores required such that (risk <= DMR)
    """
    # For now blindly return a value
    if(slack < 0):
        risk           = 0.0
        demanded_cores = 0
    else:
        risk           = 0.3
        demanded_cores = 18
    return (risk,demanded_cores)

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
    INVALID  = 12        # The UE has not arrived but its workload is known.
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
        UEExecState.WAITING  : UEExecState.PH1S1,
        UEExecState.PH1S1    : UEExecState.PH2S1,
        UEExecState.PH2S1    : UEExecState.PH3S1,
        UEExecState.PH3S1    : UEExecState.PH1S2,
        UEExecState.PH1S2    : UEExecState.PH2S2,
        UEExecState.PH2S2    : UEExecState.PH3S2,
        UEExecState.PH3S2    : UEExecState.FINISHED,
        UEExecState.FINISHED : UEExecState.FINISHED,
        UEExecState.DROPPED  : UEExecState.DROPPED
        }

    return ns[curr_state]

    
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
        self.crnti    = 0                       # UE Connection ID, constant accross a connection
        self.subframe = subframe                # Subframe ID

        # Timing Information
        self.ticks    = 0                       # The Elapsed Time (Waiting or Computing) ever since the UE arrived
        self.arrival  = arrival_time            # Arrival Time
        self.alloc    = 0                       # The number of allocated cores
        self.state    = UEExecState.WAITING     # Waiting for execution (represents a phase)
        self.deadline = D                       # UE Latency Deadline
        self.dmr      = dmr                     # Probability of Missing the deadline (Pr{R > D}), may not be used in first version
        self.risk     = dmr                     # Risk of missing the deadline
        self.start    = -1                      # Start Time
        self.finish   = -1                      # Relative Finish Time (or -1 when the SF is dropped)

        # Add workload information
        self.prbs     = prbs
        self.layers   = layers                            
    
    def execute(self,num_cores,start_time):
        """ 
            Execute a phase with allocation = num_cores,
            with the absolute start time    = start_time
            
            Return Value
            ------------
            1. the risk of not being able finish
               the remaining phases/computations within the stipulated
               deadline.
             
            2. The number of cores that must be allocated in the next
               phase in-order to keep this value bounded by the DMR requirements.
            
            The execution starts in state PH1S1.
        """
        #print("Entering in state = %s, ticks = %f" % (self.state.name,self.ticks))
        if(self.state == UEExecState.DROPPED or self.state == UEExecState.FINISHED) :
            # The UE have been already dropped previously
            risk           = 0.0
            demanded_cores = 0
            exec_time      = 0
        else :
            if(self.state == UEExecState.WAITING) :
                self.start = start_time             # Update the start time just before execution

            
            exec_time                 = 0.1 #np.random.gumbel(0.8,0.1)
            #self.ticks               = self.ticks + start_time + exec_time  - self.arrival
            self.ticks                = self.ticks + exec_time
            slack                     = D - self.ticks
            risk,demanded_cores       = \
            compute_miss_risk(slack,num_cores,self.state,self.prbs,self.layers,self.dmr)
            
            self.risk                 = risk
            if(slack < 0) :
                self.state = UEExecState.DROPPED
            else :
                self.state = next_state(self.state)

            if(self.state == UEExecState.FINISHED):
                self.finish = self.ticks           # Update the finish time just after execution
            
        #print("Exiting in state = %s, ticks = %f" % (self.state.name,self.ticks))
        return (risk,demanded_cores,exec_time)

    def grab(self,num_cores):
        """ Grab some cores """
        return 0
    
    def release(self,num_cores):
        """ Release some core """
        return 0
    
    def update_alloc(self,num_cores):
        return 0
    
    def publish_state(self):
        return (self.state,self.risk,self.alloc,self.ticks)

    def update_ticks(self,add2):
        self.ticks = self.ticks + add2
    
    def __str__(self):
        """ Display/Visaualization Method for Each UE """
        return "id:{sf:%d,crnti:%d},workload:{prbs:%d,layer:%d},timing:{arrival:%d,deadline:%d,DMR:%d,start:%f,finish:%f},FState:%s" % \
        (self.subframe,self.crnti,self.prbs,self.layers,self.arrival,self.deadline,self.dmr,self.start,self.finish,self.state.name)


def risk_scheduler_main():
    for t in range(0,2300):
        # A set of UE arrives
        sf = [UE(ue,87,t*T,D,10,4,0.3) for ue in range(0,1)]
        
        # When should the current set of UE start (Whenever the previous UE/SF have finished)
        # if (t > 0):
        #     prev_finish_time = np.max([ue.finish for ue in prevsf]) - T
        #     if(prev_finish_time < 0):
        #         prev_finish_time = 0
        # else:
        #     prev_finish_time = 0
                
        # Set the start_time of all the UE in the current sf
        #init_core_alloc = 32
        #[ue.start(init_core_alloc for ue in sf]
        
        # Execute UE in the subframes ?
        for ue in sf:
            start_time                     = t*T
            
            risk,demanded_cores,exec_time  = ue.execute(64,start_time)              # Phase 1 Slot 1 Completed
            start_time                     = start_time + exec_time

            risk,demanded_cores,exec_time  = ue.execute(demanded_cores,start_time)  # Phase 2 Slot 1 Completed
            start_time                     = start_time + exec_time

            risk,demanded_cores,exec_time  = ue.execute(demanded_cores,start_time)  # Phase 3 Slot 1 Completed
            start_time                     = start_time + exec_time

            risk,demanded_cores,exec_time  = ue.execute(demanded_cores,start_time)  # Phase 1 Slot 2 Completed
            start_time                     = start_time + exec_time

            risk,demanded_cores,exec_time  = ue.execute(demanded_cores,start_time)  # Phase 2 Slot 2 Completed
            start_time                     = start_time + exec_time

            risk,demanded_cores,exec_time  = ue.execute(demanded_cores,start_time)  # Phase 3 Slot 2 Completed
            start_time                     = start_time + exec_time

            risk,demanded_cores,exec_time  = ue.execute(demanded_cores,start_time)  # Phase 4 Completed
            start_time                     = start_time + exec_time

            print(ue)
            print("\n\n")
        
        # Save the old subframe
        #prevsf = sf

if __name__=="__main__":
    risk_scheduler_main()