# -*- coding: utf-8 -*-
"""Analyzing circuit data."""
from __future__ import division

import pickle
import numpy as np
import matplotlib.pyplot as plt


PV = 0
SST = 1
VIP = 2
Gad = 3

names = ['PV', 'SST', 'VIP', 'Gad2']

# Interneuron density data
with open('mouse_interneuron_density.pkl', 'rb') as f:
    p_kim = pickle.load(f)

inh_density_full = np.array([p_kim['pv_list'],
                             p_kim['sst_list'],
                             p_kim['vip_list'],
                             p_kim['gad_list']]).T

# Connectivity data
dataset = 'allen'
with open('mouse_isocortex_data_'+dataset+'.pkl', 'rb') as f:
    p_conn = pickle.load(f)

areas = p_conn['areas']
n_area = len(areas)

# get density for related areas
idx = [p_kim['areas'].index(area) for area in areas]
inh_density = inh_density_full[idx, :]

layers = ['1', '2/3', '5', '6a']
inh_den_layers = dict()
for layer in layers:
    idx = [p_kim['areas'].index(area+layer) for area in areas]
    inh_den_layers[layer] = inh_density_full[idx, :]
inh_den_layers['all'] = inh_density

inh_den_layers['5/6'] = (inh_den_layers['5']+inh_den_layers['6a'])/2

with open('Zingg_areadivision.pkl', 'rb') as f:
    div = pickle.load(f)
div_color_list = np.array([[31, 120, 180],
                           [166, 118, 29],
                           [253, 180, 98],
                           [117, 112, 179],
                           [231, 41, 138]])/255.
div_name_list = ['somatic',
                 'vis-aud',
                 'medial associa.',
                 'medial prefrontal',
                 'lateral']


def plot_inhden(inh_types, layer):
    dataset = 'allen'
    with open('mouse_isocortex_data_'+dataset+'.pkl', 'rb') as f:
        p_conn = pickle.load(f)

    inh_den_plot = inh_den_layers[layer]

    i1, i2 = inh_types

    mapping = {'PV': PV, 'SST': SST, 'VIP': VIP}
    i1, i2 = mapping[i1], mapping[i2]

    x_plot = inh_den_plot[:, i1] / (inh_den_plot[:, i2] + inh_den_plot[:, i1])
    x_labeltxt = names[i1]+'/('+names[i1]+'+'+names[i2]+') density'

    y_ticks = p_conn['areas']

    idx_sort = np.argsort(x_plot)
    if i1 == SST and i2 == PV:
        idx_sort = idx_sort[::-1]
    y_ticks = [y_ticks[i] for i in idx_sort]

    color1 = 'black'
    color2 = 'red'
    fs = 7
    fig = plt.figure(figsize=(2.5, 6.5))
    ax = fig.add_axes([0.2, 0.1, 0.75, 0.8])
    ax.plot(x_plot[idx_sort], range(n_area), 'o-', color=color1, markersize=4)
    ax.set_yticks(range(n_area))
    ax.set_yticklabels(y_ticks, fontsize=7)
    ax.yaxis.grid(True, 'major')

    for i, ytick in zip(idx_sort, ax.get_yticklabels()):
        for div_name, div_color in zip(div_name_list, div_color_list):
            if p_conn['areas'][i] in div[div_name]:
                ytick.set_color(div_color)

    if i1 == PV and i2 == SST:
        ax.set_xlim([0.1, 0.65])
    elif i1 == SST and i2 == PV:
        ax.set_xlim([0.4, 0.9])
    ax.set_xlabel(x_labeltxt, color=color1, fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs)

    ax2 = ax.twiny()
    ax2.plot(inh_den_plot[:, i1][idx_sort], range(n_area),
             'o-', color=color2, markersize=4, markeredgecolor=color2)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_color(color2)
    ax2.set_xlabel(names[i1]+' density ($\mathrm{mm}^{-3}$)',
                   color=color2, fontsize=fs)
    ax2.xaxis.label.set_color(color2)
    ax2.tick_params(axis='x', colors=color2)

    plt.tick_params(axis='both', which='major', labelsize=fs)
    if layer == '2/3':
        layertxt = '_L23'
    else:
        layertxt = '_L'+layer
    plt.savefig('figure/sorted_'+names[i1]+names[i2]+'ratio'+layertxt+'.pdf',
                transparent=True)


def plot_bar_two_area(area1, area2, ylabel=True, layer='all'):
    fig = plt.figure(figsize=(1.3, 0.8))
    ax = fig.add_axes([0.25, 0.25, 0.7, 0.6])
    fs = 7
    width = 0.25
    cell_types = [PV, SST, VIP]
    for area, shift in zip([area1, area2], [-width, 0]):
        area_layer = area
        if layer is not 'all':
            area_layer += layer

        inh_den = inh_density_full[p_kim['areas'].index(area_layer)]/1000.

        center = np.arange(len(cell_types))
        for i in cell_types:
            if area == area1:
                color = np.array([227, 74, 51])/255.
            else:
                color = np.array([37, 37, 37])/255.
            if i == PV:
                ax.bar(center[i]+shift, inh_den[i],
                       width=width, color=color, edgecolor='none', label=area)
            else:
                ax.bar(center[i]+shift, inh_den[i],
                       width=width, color=color, edgecolor='none')
        ax.set_xlim([center[0]-2*width, center[-1]+2*width])
        ax.set_xticks(center)
        ax.set_xticklabels(names, fontsize=fs)
        ax.set_yticks([0, 4, 8])
        if ylabel:
            ax.set_ylabel(r'$10^3$cells/mm$^3$', fontsize=fs, labelpad=0)
        else:
            ax.set_yticklabels([])
    ax.set_ylim([0, 8.2])
    ax.legend(bbox_to_anchor=(0.0, 1.2), frameon=False,
              fontsize=6, borderaxespad=0., loc='upper left')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', which='major', labelsize=fs, length=2)
    plt.savefig('figure/bar_'+area1+area2+'.pdf', transparent=True)
