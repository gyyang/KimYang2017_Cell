# -*- coding: utf-8 -*-
"""
Storing all data for interneuronal circuit building
Allows for generation of connection matrix
Allows for generation of latex output (tbf)
@author: Guangyu Robert Yang, 2015-2016
"""
from __future__ import division

import time
import pickle
import numpy as np
import matplotlib.pyplot as plt


#---------------------------------------------------------------------------------
# Parameter dictionary
#---------------------------------------------------------------------------------

p = dict()

#---------------------------------------------------------------------------------
# Neuron density, unit: /mm^3
#---------------------------------------------------------------------------------

p['den_pyr']        = {'type':  'neuron density',
                       'value': 79.3*1e3,
                       'ref':   'meyer11',
                       'layer': '1-6B',
                       'area':  'S1',
                       'animal':'rat'}

name_list_kim =          ['pv',      'sst',      'vip']
den_list_kim  = np.array([5.517e3,   4.687e3,    1.973e3])

for den, name in zip(den_list_kim, name_list_kim):
    p['den_'+name]  = {'type':  'neuron density',
                       'value': den,
                       'ref':   'Kim unpublished',
                       'layer': '1-6a',
                       'area':  'SSp',
                       'animal':'mouse'}

#---------------------------------------------------------------------------------
# Connection probability
#---------------------------------------------------------------------------------

p['pconn_pyr2pyr']  = {'type':  'connection probability',
                       'value': 0.6,
                       'ref':   'lee09',
                       'layer': '2/3',
                       'area':  'V1'}

p['pconn_pyr2pv']   = {'type':  'connection probability',
                       'value': 0.6,
                       'ref':   'lee09',
                       'layer': '2/3',
                       'area':  'V1'}

p['pconn_pyr2sst']  = {'type':  'connection probability',
                       'value': 0.6,
                       'ref':   'lee09',
                       'layer': '2/3',
                       'area':  'V1'}

p['pconn_pyr2vip']  = {'type':  'connection probability',
                       'value': 0.6,
                       'ref':   'lee09',
                       'layer': '2/3',
                       'area':  'V1'}

#---------------------------------------------------------------------------------
# Connection strength
#---------------------------------------------------------------------------------

p['wconn_pyr2pyr']  = {'type':  'connection probability',
                       'value': 0.6,
                       'ref':   'lee09',
                       'layer': '2/3',
                       'area':  'V1'}

p['wconn_pyr2pv']   = {'type':  'connection probability',
                       'value': 0.6,
                       'ref':   'lee09',
                       'layer': '2/3',
                       'area':  'V1'}

p['wconn_pyr2sst']  = {'type':  'connection probability',
                       'value': 0.6,
                       'ref':   'lee09',
                       'layer': '2/3',
                       'area':  'V1'}

p['wconn_pyr2vip']  = {'type':  'connection probability',
                       'value': 0.6,
                       'ref':   'lee09',
                       'layer': '2/3',
                       'area':  'V1'}


# From pyramidal to sst, Kapfer, ..., Scanziani 2007 NN

#---------------------------------------------------------------------------------
# Connection strength based on data from Pfeffer et al. 2013
#---------------------------------------------------------------------------------

conn_list_pfeffer = [1.00,      1.01,       0.03,       0.22,
                     0.54,      0.33,       0.02,       0.77,
                     0.02,      0.02,       0.15,       0.02]

name_list_pfeffer = ['pv2pyr',  'pv2pv',    'pv2sst',   'pv2vip',
                     'sst2pyr', 'sst2pv',   'sst2sst',  'sst2vip',
                     'vip2pyr', 'vip2pv',   'vip2sst',  'vip2vip']

for conn, name in zip(conn_list_pfeffer,name_list_pfeffer):
    p['indconnorm_'+name]= {'type':  'individual contribution normalized',
                           'value': conn,
                           'ref':   'pfeffer13',
                           'layer': '2/3',
                           'area':  'V1',
                           'dist':  (25,100)}


#---------------------------------------------------------------------------------
# Properties of neurons
#---------------------------------------------------------------------------------

# SOM neurons from Ma, .., Agmon JNS 2006

#---------------------------------------------------------------------------------
# Properties of neurons from Jiang,..., Tolias 2015
#---------------------------------------------------------------------------------

name_list_jiang = np.array(['bc'     ,'mc'   ,'btc'])
vm_list_jiang   = np.array([-68.1    ,-66.3  ,-66.5]) # resting potential mV
R_list_jiang    = np.array([84.5     ,126.1  ,131.6]) # input resistance MOmega
tau_list_jiang  = np.array([3.9      ,9.1    ,5.8])   # time constant ms
vth_list_jiang  = np.array([-44.1    ,-42.4  ,-44.2]) # spiking threshold mV
rmax_list_jiang = np.array([145.2    ,25.5   ,19.1])  # maximum firing rate Hz

# Refer to WorkingReport_2015_12_01_LocalCircuitParameters

dvth_list_jiang = vth_list_jiang - vm_list_jiang
Ith_list_jiang  = dvth_list_jiang/R_list_jiang # Current threshold nA
Imax_list_jiang = dvth_list_jiang/(R_list_jiang*(1-np.exp(-1/(tau_list_jiang*rmax_list_jiang/1000))))
beta_list_jiang = rmax_list_jiang/(Imax_list_jiang-Ith_list_jiang) # Hz/nA


#---------------------------------------------------------------------------------
# Connection probability based on Jiang,..., Tolias 2015
#---------------------------------------------------------------------------------

name_list_jiang  = np.array(['pyr'  ,'bc'   ,'mc'   ,'btc'])

# Connection probability for cell pairs with inter-soma distances less than 150mum
# matrix multiplication convention
pconn_mat_jiang  = np.array([[02.0  ,35.2   ,43.6   ,18.3],
                             [18.6  ,46.8   ,48.9   ,02.9],
                             [18.9  ,18.2   ,0      ,26.1],
                             [20.0  ,11.4   ,47.8   ,14.4]])/100.

intersomadist    = 150 # mum

# peak of IPSPs or EPSPs mV, matrix multiplication convention
psp_mat_jiang    = np.array([[0.34,  0.48,  0.31,   0.28],
                             [1.60,  0.68,  0.50,   0.18],
                             [0.86,  0.42,  0.00,   0.32],
                             [1.31,  0.41,  0.46,   0.44]])


den_list = np.array([p['den_pyr']['value']])
den_list = np.concatenate((den_list,den_list_kim*np.array([0.975,0.66,0.55])))
# multiply by the proportion. SHOULD CHANGE THIS LATER


N_list              = den_list*(4./3*np.pi*(intersomadist/1000.)**3)


tau_syn_list        = np.array([10,10,10,10]) # synaptic time constant ms. NEED CHANGE

R_list              = np.insert(R_list_jiang,0,150) # insert pyramidal input resistance MOmega

vclamp_list_jiang   = np.array([-70,-57,-57,-57]) # holding potential when activating pre-synaptic neurons mV

vsyn_list           = np.array([0,-70,-70,-70]) # synapse reversal potential mV

vvivo               = -50 # Assume the membrane potential fluctuate around this value in vivo

pre_factor          = tau_syn_list*(vvivo-vsyn_list)/(vclamp_list_jiang-vsyn_list)
post_factor         = 1./R_list

# synaptic weight matrix (pA/Hz or pC)
wsyn_mat            = psp_mat_jiang*pre_factor*post_factor[:,np.newaxis]

# Connection weight matrix on population level (pA/Hz)
w_mat               = wsyn_mat*pconn_mat_jiang*N_list


def print_tex(mat,names):
    s = ' '
    for name in names:
        s += ('& ' + name)
    s += '\\\\ \hline \n'
    for i in range(mat.shape[0]):
        mat_i = mat[i]
        s += names[i]
        for x in mat_i:
            s += '& {:3f} '.format(x)
        s += '\\\\ \hline \n'

    print s

#print_tex(w_mat,['Pyr','BC','MC','BTC'])