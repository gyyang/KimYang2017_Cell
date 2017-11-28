# -*- coding: utf-8 -*-
"""
This file reproduces results in Figures 4, 5 of the paper
Brain-wide Maps Reveal Stereotyped Cell-Type-Based Cortical Architecture and
Subcortical Sexual Dimorphism
Cell 2017

@author: Guangyu Robert Yang, 2015-2017
"""
from __future__ import division

import os
import copy
import numpy as np

from ClassificationAnalysis import do_classification
from CircuitDataAnalysis import plot_inhden
from CircuitDataAnalysis import plot_bar_two_area
from RateModel import Model
from RateModelAnalysis import plot_areadiv
from RateModelAnalysis import fancy_localcircuit_plot
from RateModelAnalysis import fancy_localcircuit_plot_currents
from RateModelAnalysis import plot_rateresponse_all
from RateModelAnalysis import plot_rateresponse_all_special
from RateModelAnalysis import plot_currentresponse_all
from RateModelAnalysis import plot_currentresponse_allinone


if not os.path.exists('figure'):
    os.makedirs('figure')
if not os.path.exists('data'):
    os.makedirs('data')

EXC = 0
PV = 1
SST = 2
VIP = 3

# Reproduce Figure 4
for layer in ['2/3', '5']:
    # Plot cortical areas in PV-SST space
    plot_areadiv(x_type=PV, y_type=SST, layer=layer, plot_txt=True)

    # Classification analysis
    do_classification(layer)

    # Plot inhibitory neuron density
    plot_inhden(inh_types=('PV', 'SST'), layer=layer)


# Matrix from Litwin-Kumar et al. 2016 Journal of Neurophysiology
#               from  soma  pv    sst   vip
W_local0 = np.array([[0.80, -1.00, -1.00, 0.00],  # to soma
                     [1.00, -1.00, -0.50, -0.00],  # to pv
                     [1.00, -0.00, -0.00, -0.25],  # to sst
                     [1.00, -0.00, -0.60, -0.00]])  # to vip

ext_params = {
        'alpha_local': 1.0,
        'alpha_inhden': [1, 1, 1],
        'W_local0': W_local0,
        'sigma_stim': 0,
        'layer': '2/3',
        'taus0': np.array([20., 10., 20., 20.]),
        }


ext_params_local = copy.deepcopy(ext_params)
model = Model(ext_params=ext_params_local, all_areas=True)

# Plot example area responses
area_runs_list = [('SSp-bfd', 'ILA'), ('AUDpo', 'RSPv')]
for area_runs in area_runs_list:
    plot_ylabel = (area_runs == ('SSp-bfd', 'ILA'))

    plot_bar_two_area(
            area_runs[0], area_runs[1], layer='2/3', ylabel=plot_ylabel)

    fancy_localcircuit_plot(model=model, input_type='to_pv',
                            area_runs=area_runs, plot_xlabel=False,
                            plot_ylabel=plot_ylabel)

    for weight_from, weight_to in zip([PV,  SST, EXC, VIP],
                                      [EXC, EXC, EXC, SST]):
        fancy_localcircuit_plot_currents(
                model=model,
                input_type='to_pv',
                area_runs=area_runs,
                weight_froms=[weight_from],
                weight_to=weight_to,
                plot_xlabel=False,
                plot_ylabel=plot_ylabel
                )

# Reproduce Figure 5
# Plot circuit responses
plot_rateresponse_all(model, input_types=[EXC, PV, SST])
plot_currentresponse_all(model)
plot_currentresponse_allinone(model)
plot_rateresponse_all_special(model, input_types=[EXC, PV, SST])
