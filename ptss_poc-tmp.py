#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 16:19:11 2018

@author: amaity
"""
    # def start(self,start_time):
    #     """ 
    #         Start the "execution".
    #         It is assumed that the start_time is
    #         relative to arrival time

    #         NOTE : You might want to change this assumption
    #         later
    #     """
    #     if(self.state == UEExecState.WAITING):
    #         self.state    = next_state(self.state)
    #     else:
    #         raise IOError("Illegal State while starting execution")

# Workload,Allocation vs Distribution Tables
# def init_workload_risk_alloc_table2():
    
#     for w in range(1,101):
#         for c in range(1,M):

# Model the execution of each phase as Gamma Distribution with 
# parameters (k,theta) computed from the workload and the Allocation


            # if(self.state == UEExecState.WAITING) :
            #     self.start = self.arrival + start_time       # Update the start time just before execution

            
            # exec_time                 = 0.1 #np.random.gumbel(0.8,0.1)
            # slack                     = ptsl.D - (self.ticks + exec_time)
            # # Drop the UE there is no slack time left
            # if(slack < 0) :
            #     self.state  = UEExecState.DROPPED
            #     self.ticks  = ptsl.D
            #     self.finish = self.ticks
            #     self.risk   = 0

            #     # The UE is now useless and hence must be dropped
            #     risk           = 0.0
            #     demanded_cores = 0
            #     exec_time      = ptsl.D - self.ticks

            #     return (risk,demanded_cores,exec_time)
            # else :
            #     risk,demanded_cores       = \
            #     compute_miss_risk(slack,num_cores,self.state,self.prbs,self.layers,self.dmr)
                
            #     self.risk                 = risk
            #     self.state                = next_state(self.state)
            #     self.ticks                = self.ticks + exec_time
                
            #     if(self.state == UEExecState.FINISHED):
            #         self.finish = self.ticks           # Update the finish time just after execution
                
            #     return (risk,demanded_cores,exec_time)
   



#########################################
# def risk_scheduler_test():
#     for t in range(0,10):
#         # A UE arrives
#         ue = UE(0,87,t*(ptsl.T),ptsl.D,10,4,0.3)

#         start_time                     = 0                                      # Here the start_time is relative to UE arrival time
        
#         risk,demanded_cores,exec_time  = ue.execute(64,start_time)              # Phase 1 Slot 1 Completed
#         start_time                     = start_time + exec_time

#         risk,demanded_cores,exec_time  = ue.execute(demanded_cores,start_time)  # Phase 2 Slot 1 Completed
#         start_time                     = start_time + exec_time

#         risk,demanded_cores,exec_time  = ue.execute(demanded_cores,start_time)  # Phase 3 Slot 1 Completed
#         start_time                     = start_time + exec_time

#         risk,demanded_cores,exec_time  = ue.execute(demanded_cores,start_time)  # Phase 1 Slot 2 Completed
#         start_time                     = start_time + exec_time

#         risk,demanded_cores,exec_time  = ue.execute(demanded_cores,start_time)  # Phase 2 Slot 2 Completed
#         start_time                     = start_time + exec_time

#         risk,demanded_cores,exec_time  = ue.execute(demanded_cores,start_time)  # Phase 3 Slot 2 Completed
#         start_time                     = start_time + exec_time

#         risk,demanded_cores,exec_time  = ue.execute(demanded_cores,start_time)  # Phase 4 Completed
#         start_time                     = start_time + exec_time

#         print(ue)
#         print("\n\n")
        
#         #Save the old subframe
#         #prevsf = sf

# class UEPipeline(object):
#     """ Represents the UE pipeline """
#     def __init__(self,crnti):
#         #self.ueq           = queue.Queue(depth)         # Is it really required
#         #self.alloc_history = queue.Queue(depth-1)        # Save The core-allocation of the previous depth-1 UEs
#         self.crnti         = 89927
#         self.subframe      = 0
    
#     def step(self,m):
#         """ 
#             Step a single Time period, with m cores.
#             "m" is vestigeal for now, but might be 
#             useful later on.

#             Return value
#             ------------
#             Return the allocation details for each phase
#             in the following format :
            
#             (start_time (absolute), end_time(absolute), core_allocations)

#         """

#         # Granularity of time step increments when checking for number of free cores (may be move this to different file)
#         #delta = 0.01
        
#         # Insert a new UE in the Queue
#         curr_time = (self.subframe)*(ptsl.T)
#         ue        = UE(self.crnti,self.subframe,curr_time,ptsl.D,10,4,0.3)

#         # Determine the execution start time of this UE,
#         # Consider the fact that the previous UE might be
#         # Occupying the all cores in curr_time.
#         # Start Execution when you have atleast one Core Free.
#         #start_time2 = curr_time
#         #while prev_free_cores <= 0:
#         #    prev_occ_cores       = 0
#         #    for alloc in self.alloc_history.queue:           
#         #        prev_occ_cores             = prev_occ_cores+ get_occupied_cores(alloc,start_time2)
#         #    prev_free_cores = m - prev_occ_cores
#         #    start_time2 = start_time2 + delta
#         # Store allocation's of the current UE execution

#         alloc                           = [] # Store the allocation details to allow post-processing

#         demanded_cores1                 = m                                      # TODO : Number of cores allocated for phase1 (recompute this using the PDFs)

#         # Complete Phase 1 Slot 1 and advance the time step
#         start_time                      = 0                                      # Here the start_time is relative to UE arrival time
#         risk,demanded_cores2,exec_time  = ue.execute(demanded_cores1,start_time)
#         alloc.append((curr_time+start_time,curr_time+start_time+exec_time,demanded_cores1))

#         # Complete Phase 2 Slot 1 and advance the time step
#         start_time                      = start_time + exec_time
#         risk,demanded_cores3,exec_time  = ue.execute(demanded_cores2,start_time)
#         alloc.append((curr_time+start_time,curr_time+start_time+exec_time,demanded_cores2))

#         # Complete Phase 3 Slot 1 and advance the time step
#         start_time                      = start_time + exec_time
#         risk,demanded_cores4,exec_time  = ue.execute(demanded_cores3,start_time)
#         alloc.append((curr_time+start_time,curr_time+start_time+exec_time,demanded_cores3))

#         # Complete Phase 2 Slot 1 and advance the time step
#         start_time                      = start_time + exec_time
#         risk,demanded_cores5,exec_time  = ue.execute(demanded_cores4,start_time)
#         alloc.append((curr_time+start_time,curr_time+start_time+exec_time,demanded_cores4))

#         # Complete Phase 2 Slot 2 and advance the time step
#         start_time                      = start_time + exec_time
#         risk,demanded_cores6,exec_time  = ue.execute(demanded_cores5,start_time)
#         alloc.append((curr_time+start_time,curr_time+start_time+exec_time,demanded_cores5))

#         # Complete Phase 2 Slot 3 and advance the time step
#         start_time                      = start_time + exec_time
#         risk,demanded_cores7,exec_time  = ue.execute(demanded_cores6,start_time)
#         alloc.append((curr_time+start_time,curr_time+start_time+exec_time,demanded_cores6))

#         # Complete Phase 4 and advance the time step
#         start_time                      = start_time + exec_time
#         risk,demanded_cores7,exec_time  = ue.execute(demanded_cores7,start_time)
#         alloc.append((curr_time+start_time,curr_time+start_time+exec_time,demanded_cores7))

#         self.subframe = self.subframe + 1

#         return alloc



# def compute_miss_risk(slack,num_cores,curr_state,prbs,layers,DMR):
#     """ 
#         Compute my risk (probability) of missing the deadline, given the
#         1. The time that have been elapsed ever since the UE arrival - Deadline (aka slack)
#         2. The number of cores currently allocated
#         3. Computation that is remaining (aka all the states that the execution still has to traverse)
#         4. The workload information (prbs and layers)
#         5. Timing Requirements presented as DMR
#         ----------------------------------------------------------------
#         When do you invoke this ?
#         1. Whenever you perform a state transition (aka at the end of execute call)
#         ----------------------------------------------------------------
#         Return Value
#         ------------
#         1. The risk of missing the deadline with the current allocation
#         2. The (minimum) number of cores required such that (risk <= DMR)
#     """
#     # For now blindly return a value
#     if(slack < 0):
#         risk           = 0.0
#         demanded_cores = 0
#     else:
#         risk           = 0.3
#         demanded_cores = 18
#     return (risk,demanded_cores)

###############################################################
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
#import ptss_poc as ps
import heapq
import queue

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
    
#def dummy_ue_heapify_test():
#    # Create a list of UE
#    ue_list = [ps.dummy_ue(i) for i in range(10,0,-1)]
#    [print(ue) for ue in ue_list]
#    heapq.heapify(ue_list)
#    print("\n\n\n")
#    [print(ue) for ue in ue_list]

if __name__=="__main__":
    #numpy_load_test()
    alloc_q_test()