# -*- coding: utf-8 -*-
"""
Analyzing circuit
@author: Guangyu Robert Yang, 2015-2016
"""
from __future__ import division

import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import scipy.stats

from figtools import Figure,Subplot,colors
from matplotlib.ticker import LinearLocator


datafile='library/mouse_isocortex_data_usc.pkl'
with open(datafile,'rb') as f:
    p = pickle.load(f)

W = p['W_mean']>0

areas = p['areas']
n_area = len(areas)

W = np.zeros((n_area,n_area))
for i in range(1,n_area):
    W[i,i-1] = 1
W[:,-1] = 1

idx_0 = areas.index('SSp-bfd')
idx_1 = areas.index('SSs')

hiers = np.ones(n_area)
#hiers = np.random.rand(n_area)*5

alpha = 0.5

for i in range(1000):
    input_degree = W.sum(axis=1)
    output_degree = W.sum(axis=0)
    
    input_hiers = (np.dot(W,hiers))/input_degree
    
    output_hiers = (np.dot(W.T,hiers))/output_degree
    
    target_hiers = (input_hiers+1)/2 + (output_hiers-1)/2
    
    target_hiers = input_hiers+1
    
    hiers = hiers*(1-alpha)+alpha*target_hiers
    
    hiers[idx_0] = 0
    hiers[idx_1] = 1

print hiers