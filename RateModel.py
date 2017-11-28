# -*- coding: utf-8 -*-
"""Rate model."""
from __future__ import division

import pickle
import numpy as np

EXC, PV, SST, VIP = 0, 1, 2, 3

colors = np.array([[8, 48, 107],
                   [228, 26, 28],
                   [152, 78, 163],
                   [77, 175, 74]]) / 255.

names = ['soma', 'dend', 'pv', 'sst', 'vip']


class Model(object):
    """The model."""

    def __init__(self, datafile='mouse_isocortex_data_allen.pkl',
                 ext_params={}, all_areas=False):
        self.rng = np.random.RandomState(520)

        # Loading Anatomical Data from the Allen Institute
        with open(datafile, 'rb') as f:
            p = pickle.load(f)

        p['datafile'] = datafile
        p['n_area'] = len(p['areas'])

        # Loading Interneuron Density Data from Kim's group
        with open('mouse_interneuron_density.pkl', 'rb') as f:
            p_kim = pickle.load(f)

        # Single Neuron Parameters
        pop_list = ['soma', 'pv', 'sst', 'vip']
        self.pop_list = pop_list

        p['n_pop'] = 4     # number of populations

        p['taus0'] = np.array([20., 10., 20., 20.])        # time constant ms

        def relu(x): return x * (x > 0)
        self.relu = relu

        p['r_tgt0'] = np.array([5, 10, 20, 5])  # Target spontaneous activity

        # Local connectivity matrix
        #           from       soma,  pv,   sst,  vip
        W_local0 = np.array([[0.75, -1.00, 0.00, 0.00],  # to soma
                             [1.00, -1.00, -0.50, -0.00],  # to pv
                             [0.50, -0.00, -0.00, -1.50],  # to sst
                             [0.50, -0.00, -0.50, -0.00]])  # to vips
        p['W_local0'] = W_local0

        # choosing between variable density or uniform density across areas
        p['alpha_inhden'] = 1

        p['sigma_stim'] = 1  # Noise level

        for key, value in ext_params.iteritems():
            p[key] = value

        # Derived parameters
        if 'layer' not in p:
            layer = ''
        else:
            layer = p['layer']

        inh_density_full = np.array(
            [p_kim['pv_list'], p_kim['sst_list'], p_kim['vip_list']])

        if all_areas:
            # include the two areas that were not in the USC dataset
            p['areas'] = p['areas'] + ['VISpl', 'AUDpo']
        p['n_area'] = len(p['areas'])
        idx = [p_kim['areas'].index(area + layer) for area in p['areas']]

        self.inh_density = inh_density_full[:, idx]

        # Density of neurons. Will scale the connection weights
        den_norm = np.ones((p['n_pop'], p['n_area']))
        # Normalize by their mean
        inh_den_norm = self.inh_density / \
            self.inh_density.mean(axis=1, keepdims=True)

        inh_den_norm_mean = np.mean(inh_den_norm, axis=1)
        if 'inh_den_norm_mean' in p:
            inh_den_norm_mean = inh_den_norm_mean * \
                p['inh_den_norm_mean']
        inh_den_norm_mean = inh_den_norm_mean[:, np.newaxis].dot(
            np.ones((1, p['n_area'])))

        if not hasattr(p['alpha_inhden'], "__len__"):
            p['alpha_inhden'] = np.array(
                [1, 1, 1]) * p['alpha_inhden']  # Turn into an array
            # alpha_inhden should represent the ratio of data-like distribution
            # for each type
        else:
            p['alpha_inhden'] = np.array(p['alpha_inhden'])

        den_norm[[PV, SST, VIP]] = (
                inh_den_norm_mean * (1 - p['alpha_inhden'][:, np.newaxis]) +
                inh_den_norm * p['alpha_inhden'][:, np.newaxis]
                )

        p['den_norm'] = den_norm

        p['W_local'] = p['W_local0']

        self.p = p

    def make_effective_matrix_local(self, area_run):
        """Calculate the local effective connectivity matrix."""
        p = self.p
        area_run_idx = p['areas'].index(area_run)  # Index of stimulated area
        W_local = p['W_local']

        den_norm = p['den_norm'][:, area_run_idx]

        W = W_local * den_norm

        # Effective weight matrix
        W_eff = ((W - np.eye(p['n_pop'])).T / p['taus0']).T

        self.W = W
        self.W_eff = W_eff

    def run_localcircuit(self, resultfile='data/run_local.pkl',
                         input_type='top down', area_run='VISp'):
        """
        Run a local circuit disconnected from the large-scale network.
        """
        p = self.p

        if input_type == 'to_vip':  # disinhibitory input targeting only VIP
            stim_tos = [VIP]
        elif input_type == 'to_pv':  # feedforward inhibition
            stim_tos = [PV]
        elif input_type == 'to_sst':
            stim_tos = [SST]

        self.make_effective_matrix_local(area_run)

        # Simulation parameters
        dt = 0.02   # ms
        dt_record = 0.5   # ms
        T = 300  # ms
        n_t = int(round(T // dt)) + 1
        n_recorddt = int(round(dt_record / dt))

        # From target background firing inverts background inputs
        r_tgt = p['r_tgt0']

        # From target background firing inverts background inputs
        I_bkg = r_tgt - np.dot(self.W, r_tgt)

        # External stimulus
        stim_on = 50
        stim_off = 150
        stim_amp = 1
        sigma_stim = 0.1 * 0

        # Initialize activity to background firing
        r = r_tgt

        # Storage
        r_store = [r]
        I_stim_store = [np.zeros(p['n_pop'])]
        t_plot = [0]

        # Running the network
        for i_t in xrange(1, n_t):
            t = i_t * dt

            I_stim = self.rng.randn(p['n_pop']) / np.sqrt(dt) * sigma_stim
            I_stim[stim_tos] += stim_amp * (t > stim_on) * (t <= stim_off)
            I_local = np.dot(self.W, r)
            r = r + (-r + self.relu(I_local + I_stim + I_bkg)) * \
                dt / p['taus0']

            r = np.minimum(r, 300)
            r = np.maximum(r, 0)

            if i_t % n_recorddt == 0:
                r_store.append(r)
                I_stim_store.append(I_stim)
                t_plot.append(t)

        result = {'r': np.array(r_store),
                  'I_stim': np.array(I_stim_store),
                  'p': p,
                  't_plot': np.array(t_plot),
                  'type': 'local_run',
                  'area_run': area_run,
                  'input_type': input_type,
                  'W_eff': self.W_eff,
                  'stim_on': stim_on,
                  'stim_off': stim_off}

        with open(resultfile, 'wb') as f:
            pickle.dump(result, f)
