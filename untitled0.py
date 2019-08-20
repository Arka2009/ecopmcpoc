#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 16:23:41 2019

@author: amaity
"""
from scipy.stats import randint

def main(N,fl):
    X  = randint.rvs(2,65536,size=N)
    print(X)
    #fl = "SeqS.in"
    fd = open(fl,'w')
    for x in X:
        fd.write(f'{x}\n')
    fd.close()

if __name__=="__main__":
    main(160,'SeqS.in')
    main(160,'SeqT.in')