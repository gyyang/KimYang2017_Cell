# -*- coding: utf-8 -*-
"""Rate model analysis."""
from __future__ import division

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from RateModel import Model

EXC, PV, SST, VIP = 0, 1, 2, 3
pops = [EXC, PV, SST, VIP]

colors = np.array([[8, 48, 107],         # dark-blue
                   [228, 26, 28],        # red
                   [152, 78, 163],       # purple
                   [77, 175, 74]]) / 255.  # green

names = ['E', 'PV', 'SST', 'VIP']

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


def div_sort(areas, inh_density):
    idx_sort = []
    if 'VISpl' in div['medial']:
        div['medial'].remove('VISpl')
        div['medial'].remove('AUDpo')
        div['vis-aud'].remove('VISpl')
        div['vis-aud'].remove('AUDpo')

    for div_name in div_name_list:
        div_areas = div[div_name]
        div_areas_ind = [areas.index(area) for area in div_areas]
        idx_sort_div = np.argsort(-inh_density[0][div_areas_ind])
        div_areas_ind_new = [div_areas_ind[i] for i in idx_sort_div]
        idx_sort.extend(div_areas_ind_new)
    return idx_sort


def fancy_localcircuit_plot(
        model=None,
        input_type='bottom up',
        area_runs=['SSp-n', 'AIp'],
        plot_ylabel=True,
        plot_xlabel=True):
    """Fancy local circuit plot."""
    fig = plt.figure(figsize=(1.2, 0.6))
    ax = fig.add_axes((0.15, 0, 0.65, 1))
    fs = 7

    result_list = list()
    for i_area in range(2):
        resultfile = 'data/run_local_' + input_type + \
            '_' + area_runs[i_area] + '.pkl'
        if model is not None:
            model.run_localcircuit(resultfile=resultfile,
                                   input_type=input_type,
                                   area_run=area_runs[i_area])
        with open(resultfile, 'rb') as f:
            result = pickle.load(f)
        result_list.append(result)

    pop = EXC
    for i_area in range(2):
        if i_area == 0:
            color = np.array([227, 74, 51]) / 255.
        else:
            color = np.array([37, 37, 37]) / 255.
        result = result_list[i_area]
        y_pop_plot = result['r'][:, pop]
        ax.plot(result['t_plot'], y_pop_plot - y_pop_plot[0],
                color=color, label=area_runs[i_area])

    ll = ax.plot([result['stim_on'], result['stim_off']],
                 [0.3] * 2, color='black', linewidth=1.5)
    ll[0].set_clip_on(False)
    ax.text(result['stim_on'], 0.5, '100 ms', fontsize=7)

    yticks = [-1, 0]
    ax.set_yticks(yticks)

    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylabel('')
    if plot_ylabel:
        ax.text(200, 0.5, r'$\Delta r_{\mathrm{E}}$', fontsize=7)
        ll = ax.plot([-20, -20], [-1, 0], color='black', linewidth=1.5)
        ll[0].set_clip_on(False)
        ax.text(-30, -1, '1 a.u.', fontsize=7, rotation='vertical',
                horizontalalignment='right', verticalalignment='bottom')
    if plot_xlabel:
        ax.set_xlabel('Time (ms)', fontsize=fs, labelpad=0)
        ax.set_xticks([0, 300])
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel('')
        ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([0, 300])

    for axis in ['x', 'y']:
        ax.tick_params(axis=axis, size=3.5, labelsize=7, pad=2.5)

    plotfile = 'run_local_fancyplot_' + \
        input_type.replace(" ", "") + area_runs[0] + area_runs[1] + '.pdf'
    plt.savefig('figure/' + plotfile, transparent=True)
    print('Figure saved at ' + plotfile)


def fancy_localcircuit_plot_currents(
        model=None,
        input_type='bottom up',
        area_runs=['SSp-n', 'AIp'],
        weight_froms=[EXC],
        weight_to=EXC,
        plot_ylabel=True,
        plot_xlabel=True):
    """Fancy local circuit plot of current response."""
    result_list = list()
    W_list = list()
    for i_area in range(2):
        resultfile = 'data/run_local_' + input_type + \
            '_' + area_runs[i_area] + '.pkl'
        model.run_localcircuit(resultfile=resultfile,
                               input_type=input_type,
                               area_run=area_runs[i_area])
        with open(resultfile, 'rb') as f:
            result = pickle.load(f)
        result_list.append(result)
        W_list.append(model.W)

    for weight_from in weight_froms:
        fig = plt.figure(figsize=(1.2, 0.6))
        ax = fig.add_axes((0.15, 0, 0.65, 1))
        fs = 7
        for i_area in range(2):
            result = result_list[i_area]
            if i_area == 0:
                color = np.array([227, 74, 51]) / 255.
            else:
                color = np.array([37, 37, 37]) / 255.
            y_pop_plot = (
                    W_list[i_area][weight_to,
                                   weight_from] * result['r'][:, weight_from])
            y_pop_plot = y_pop_plot - y_pop_plot[0]
            ax.plot(result['t_plot'], y_pop_plot,
                    color=color, label=area_runs[i_area])

        ymax = np.ceil(np.mean(y_pop_plot))
        if weight_from == VIP:
            ymax = 1
        # ymax=1
        ax.set_yticks([ymax - 1, ymax])
        ylabels = {EXC: r'$\Delta I_{\mathrm{E \rightarrow E}}$',
                   PV: r'$\Delta I_{\mathrm{PV \rightarrow E}}$',
                   SST: r'$\Delta I_{\mathrm{SST \rightarrow E}}$',
                   VIP: r'$\Delta I_{\mathrm{VIP \rightarrow SST}}$'}
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel('')
        if plot_ylabel:
            ax.text(200, 0.5, ylabels[weight_from], fontsize=7)

        if plot_xlabel:
            ax.set_xlabel('Time (ms)', fontsize=fs, labelpad=-5)
            ax.set_xticks([0, 300])
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_xlabel('')
            ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_ylim([-1.2, 1.2])

        for axis in ['x', 'y']:
            ax.tick_params(axis=axis, size=3.5, labelsize=7, pad=2.5)

        plotfile = 'run_local_fancyplot_currents_' + \
            input_type.replace(" ", "") + \
            names[weight_from] + area_runs[0] + area_runs[1] + '.pdf'
        plt.savefig('figure/' + plotfile, transparent=True)
        print('Figure saved at ' + plotfile)


def plot_areadiv(
        x_type=PV,
        y_type=SST,
        layer='2/3',
        figsize=(3.6, 3.6),
        plot_ticks=True,
        plot_txt=True):
    ext_params = {'layer': layer}
    model = Model(ext_params=ext_params, all_areas=True)
    p = model.p

    if layer == '2/3':
        extent = [0., 2, 0., 2]

    elif layer == '5':
        if (x_type, y_type) == (PV, SST):
            extent = [0., 2, 0., 2]
        elif (x_type, y_type) in [(SST, VIP), (PV, VIP)]:
            extent = [0., 2, 0.2, 2]
        else:
            extent = [0, 2.5, 0, 2.5]

    fs = 7
    fig = plt.figure(figsize=figsize)
    if figsize[0] > 3:
        ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    else:
        ax = fig.add_axes([0.2, 0.2, 0.7, 0.75])
    s_size = int(figsize[0] * 3)

    shift_dict = {'ACAd': (0.02, 0.02),
                  'ACAv': (0.02, 0.02),
                  'AId': (0.02, 0.025),
                  'AIp': (-0.1, 0.02),
                  'AIv': (0.02, 0.02),
                  'AUDd': (0.015, -0.06),
                  'AUDp': (-0.16, -0.06),
                  'AUDv': (-0.16, 0.02),
                  'ECT': (-0.13, -0.06),
                  'GU': (0.02, -0.06),
                  'ILA': (0.02, 0.02),
                  'MOp': (0.02, 0.02),
                  'MOs': (-0.15, -0.02),
                  'ORBl': (0.02, 0.02),
                  'ORBm': (0.02, 0.02),
                  'ORBvl': (0.02, 0.02),
                  'PERI': (0.01, 0.03),
                  'PL': (-0.10, 0.02),
                  'PTLp': (0.02, 0.02),
                  'RSPagl': (0.02, 0.02),
                  'RSPd': (0.02, 0.02),
                  'RSPv': (0.02, 0.02),
                  'SSp-bfd': (-0.1, -0.07),
                  'SSp-ll': (-0.18, -0.02),
                  'SSp-m': (0.01, 0.03),
                  'SSp-n': (0.02, 0.02),
                  'SSp-tr': (0.02, -0.02),
                  'SSp-ul': (0.03, -0.04),
                  'SSs': (0.02, 0.02),
                  'TEa': (-0.1, -0.06),
                  'VISC': (-0.06, 0.03),
                  'VISal': (-0.06, 0.03),
                  'VISam': (-0.18, -0.06),
                  'VISl': (-0.12, 0.02),
                  'VISp': (0.02, -0.06),
                  'VISpm': (-0.2, 0.02),
                  'VISpl': (-0.16, 0.02),
                  'AUDpo': (-0.2, 0.02)}

    if x_type == PV and y_type == SST and layer == '2/3':
        adjusttxt = False
    else:
        adjusttxt = True

    texts = []
    for div_name, div_color in zip(div_name_list, div_color_list):
        area_ids = [p['areas'].index(area) for area in div[div_name]]
        div_colors = [div_color] * len(area_ids)
        ax.scatter(p['den_norm'][x_type, area_ids],
                   p['den_norm'][y_type, area_ids],
                   c=div_colors, edgecolor=div_colors,
                   s=s_size, label=div_name)

        if plot_txt:
            for area_id in area_ids:
                xshift, yshift = shift_dict[p['areas'][area_id]]
                if adjusttxt:
                    xshift, yshift = 0, 0
                ax_ = ax.text(p['den_norm'][x_type, area_id] + xshift,
                              p['den_norm'][y_type, area_id] + yshift,
                              p['areas'][area_id], fontsize=6, color=div_color)
                texts.append(ax_)

    if plot_txt and adjusttxt:
        from adjustText import adjust_text
        adjust_text(
            texts,
            arrowprops=dict(
                arrowstyle="-",
                color='gray',
                lw=0.5))

    # ax.set_aspect('equal')
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])

    figname = 'plot_areadivs' + 'L' + \
        layer.replace("/", "") + names[x_type] + names[y_type]

    if plot_ticks:
        ax.set_xticks([0., .5, 1.0, 1.5, 2.0])
        if layer == '2/3':
            ax.set_yticks([0., 0.5, 1, 1.5, 2])
        elif layer == '5':
            ax.set_yticks([0., 0.5, 1, 1.5, 2])

        ax.set_ylabel(names[y_type] + ' normalized density', fontsize=fs)

        plt.tick_params(axis='both', which='major', labelsize=fs)
        ax.set_xlabel(names[x_type] + ' normalized density', fontsize=fs)
    else:
        plt.axis('off')
        figname += '_small'

    plt.savefig('figure/' + figname + '.pdf', transparent=True)


def plot_var_densityspace(
        model,
        x_type=PV,
        y_type=SST,
        var_type='eff_conn',
        var_ids=None,
        savename_addon='',
        plot_colorbar=False,
        plot_area=False,
        plot_ylabel=True,
        plot_xlabel=True,
        simple_label=False,
        **kwargs):
    p = model.p
    W_local = p['W_local']

    x = np.linspace(0.0, 2.0, 31)
    if y_type == SST:
        y = np.linspace(0.5, 2.0, 31)
    elif y_type == VIP:
        y = np.linspace(0.2, 2.0, 31)
    elif y_type == PV:
        y = np.linspace(0.2, 2.0, 31)

    Y, X = np.meshgrid(x, y)

    Z = np.zeros(X.shape)
    L = np.zeros(X.shape)
    for i_x in range(len(x)):
        for i_y in range(len(y)):
            den_norm = np.ones(4)
            den_norm[x_type] = x[i_x]
            den_norm[y_type] = y[i_y]

            # Get input response
            W = W_local * den_norm
            try:
                Omega = np.linalg.inv(np.eye(p['n_pop']) - W)
            except BaseException:
                print(den_norm)
            if var_type == 'eff_conn':
                input_type, response_type = var_ids
                Z[i_y, i_x] = Omega[response_type, input_type]
            elif var_type == 'input_current':
                input_type, conn_from_type, conn_to_type = var_ids
                Z[i_y, i_x] = Omega[conn_from_type, input_type] * \
                    W[conn_to_type, conn_from_type]
            elif var_type == 'contrast':
                input_type = PV
                a1 = W[EXC, PV] * (Omega[PV, input_type] + 2)
                a2 = W[EXC, SST] * (Omega[SST, input_type] + 2)
                Z[i_y, i_x] = abs(a1 - a2) / (a1 + a2)
                # Z[i_y,i_x] = a1+a2
            elif var_type == 'cocktail':
                response_type = var_ids
                if 'mix_input' in kwargs:
                    mix_input = kwargs['mix_input']
                else:
                    # Scaled by data from Wall et al. JNS 2016 Figure 5C
                    mix_input = np.array([0.5, 0.55, 0.3, 1]) * 2
                Z[i_y, i_x] = np.dot(Omega[response_type, :], mix_input)

            # Effective weight matrix
            W_eff = ((W - np.eye(p['n_pop'])).T / p['taus0']).T
            l, v = np.linalg.eig(W_eff)
            L[i_y, i_x] = np.real(l).max()

    cmap = 'Reds'
    if ((var_type == 'eff_conn' and input_type in [PV, SST]) or
            (var_type == 'input_current' and conn_from_type is not EXC)):
        cmap = 'Blues_r'  # reverse the color map if inhibitory
    if var_type == 'contrast':
        cmap = 'Greens'
        var_ids = 'contrast'
    if var_type == 'cocktail':
        cmap = 'bwr_r'
        var_ids = 'cocktail'

    if 'ax' in kwargs:
        ax = kwargs['ax']
    else:
        if 'figsize' in kwargs:
            figsize = kwargs['figsize']
        else:
            figsize = (2.5, 2.3)
        fig = plt.figure(figsize=figsize)
        if 'rect' in kwargs:
            rect = kwargs['rect']
        else:
            rect = [0.2, 0.2, 0.7, 0.6]
        ax = fig.add_axes(rect)

    im = ax.imshow(Z, cmap=cmap, origin='lower',
                   extent=[x[0], x[-1], y[0], y[-1]], alpha=0.3)
    CS = ax.contour(Z, 5, extent=[x[0], x[-1], y[0], y[-1]],
                    colors='k', linestyles='solid')

    ax.set_xlim((x[0], x[-1]))
    ax.set_ylim((y[0], y[-1]))

    if simple_label:
        fs = 6
        labels = {PV: 'PV', SST: r'SST', VIP: r'VIP'}

        ax.set_xticks([0, 2.0])
        if plot_xlabel:
            ax.set_xlabel(labels[x_type], fontsize=fs, labelpad=-5)
        else:
            ax.set_xticklabels([])

        ax.set_yticks([0.5, 2.0])
        if plot_ylabel:
            ax.set_ylabel(labels[y_type], fontsize=fs, labelpad=-5)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', labelsize=fs)
    else:
        fs = 7
        labels = {PV: r'PV normalized density $\rho_{\mathrm{PV}}$',
                  SST: r'SST normalized density $\rho_{\mathrm{SST}}$',
                  VIP: r'VIP normalized density $\rho_{\mathrm{VIP}}$'}

        ax.set_xticks([0, 0.5, 1, 1.5, 2.0])
        if plot_xlabel:
            ax.set_xlabel(labels[x_type], fontsize=fs)
        else:
            ax.set_xticklabels([])

        ax.set_yticks([0.5, 1, 1.5, 2.0])
        if plot_ylabel:
            ax.set_ylabel(labels[y_type], fontsize=fs)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', labelsize=fs)

    ax.clabel(CS, fontsize=fs, inline=True, fmt='%0.2f')

    if plot_area:
        for div_name, div_color in zip(div_name_list, div_color_list):
            area_id = [p['areas'].index(area) for area in div[div_name]]
            div_colors = [div_color] * len(area_id)
            ax.scatter(p['den_norm'][x_type, area_id],
                       p['den_norm'][y_type, area_id],
                       c=div_colors, edgecolor=div_colors, s=10)

    if plot_colorbar:
        if var_type == 'eff_conn':
            cb_label = (names[response_type] + ' Rate Response\n ' +
                        r'$\Delta r_' + names[response_type] +
                        '/I_{\mathrm{' + names[input_type] +
                        ',\mathrm{ext}}}$')
        elif var_type == 'input_current':
            cb_label = (
                    names[conn_from_type] + '-' + names[conn_to_type] +
                    ' Current Response\n' + r'$\Delta I_{\mathrm{' +
                    names[conn_from_type] + r'\rightarrow ' +
                    names[conn_to_type] + r'}}/I_{\mathrm{' +
                    names[input_type] + r',\mathrm{ext}}}$')
        else:
            cb_label = var_type

        labelpad = -25
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax, orientation='horizontal')
        ticks = [np.ceil(Z.min() * 1e2) * 1e-2, np.floor(Z.max() * 1e2) * 1e-2]
        cb.set_ticks(ticks)
        cb.set_label(cb_label, labelpad=labelpad, fontsize=fs)
        cb.ax.xaxis.set_ticks_position('top')
        plt.tick_params(axis='both', which='major', labelsize=fs)

    if var_type == 'eff_conn':
        fig_name_type = names[input_type] + names[response_type]
    elif var_type == 'input_current':
        fig_name_type = names[input_type] + \
            names[conn_from_type] + names[conn_to_type]
    elif var_type == 'contrast':
        fig_name_type = 'inh_contrast'
    elif var_type == 'cocktail':
        fig_name_type = names[response_type]
    figname = ('plot_' + var_type + 'L' + p['layer'].replace("/", "") +
               names[x_type] + names[y_type] + '_' +
               fig_name_type + savename_addon)

    if 'nosave' not in kwargs:
        plt.savefig('figure/' + figname + '.pdf', transparent=True)

    return Z, L


def plot_rateresponse_all(model, input_types=[EXC, PV, SST, VIP]):
    """Plot response matrices."""
    response_type = EXC
    Z_list = list()
    for input_type in input_types:
        Z, L = plot_var_densityspace(
                model,
                x_type=PV,
                y_type=SST,
                var_type='eff_conn',
                var_ids=(input_type, response_type),
                plot_area=False,
                plot_colorbar=True,
                plot_ylabel=(input_type in [PV, SST]))
        Z_list.append(Z)


def plot_rateresponse_all_special(model, input_types=[EXC, PV, SST, VIP]):
    fs = (1.1, 1.1)
    rect = [0.25, 0.25, 0.7, 0.6]
    response_type = EXC
    Z_list = list()
    for input_type in input_types:
        Z, L = plot_var_densityspace(
            model,
            x_type=PV,
            y_type=SST,
            var_type='eff_conn',
            var_ids=(input_type, response_type),
            plot_area=False,
            plot_colorbar=False,
            plot_ylabel=(input_type == PV),
            plot_xlabel=(input_type == PV),
            simple_label=True,
            figsize=fs,
            rect=rect)
        Z_list.append(Z)

    # No recurrent excitation
    W_local0 = model.p['W_local0']
    W_local1 = W_local0.copy()
    W_local1[[EXC, PV, SST, VIP], EXC] = 0
    ext_params = model.p.copy()
    ext_params['W_local0'] = W_local1
    model = Model(ext_params=ext_params)
    for input_type in [PV, SST]:
        plot_var_densityspace(
            model,
            x_type=PV,
            y_type=SST,
            var_type='eff_conn',
            var_ids=(input_type, response_type),
            plot_colorbar=False,
            savename_addon='norecE',
            plot_ylabel=False,
            plot_xlabel=False,
            simple_label=True,
            figsize=fs,
            rect=rect)

    # Weaken connections
    W_local1 = W_local0 * 0.3
    ext_params['W_local0'] = W_local1
    model = Model(ext_params=ext_params)
    for input_type in [PV, SST]:
        plot_var_densityspace(
            model,
            x_type=PV,
            y_type=SST,
            var_type='eff_conn',
            var_ids=(input_type, response_type),
            plot_colorbar=False,
            savename_addon='weakConn',
            plot_ylabel=False,
            plot_xlabel=False,
            simple_label=True,
            figsize=fs,
            rect=rect)

    return Z_list


def plot_currentresponse_all(model):
    Z_list = list()
    input_type = PV
    conn_to_type = EXC
    for conn_from_type in [EXC, PV, SST]:
        Z, L = plot_var_densityspace(
            model,
            x_type=PV,
            y_type=SST,
            var_type='input_current',
            var_ids=(input_type, conn_from_type, conn_to_type),
            plot_area=False,
            plot_colorbar=True,
            plot_ylabel=(conn_from_type == SST))
        Z_list.append(Z)

    return Z_list


def plot_currentresponse_allinone(model):
    conn_to_type = EXC
    f, axarr = plt.subplots(3, 4, figsize=(4, 3))
    for i, input_type in enumerate([EXC, PV, SST, VIP]):
        for j, conn_from_type in enumerate([EXC, PV, SST]):
            pl = (i == 0) and (j == 0)
            var_ids = (input_type, conn_from_type, conn_to_type)
            plot_var_densityspace(model,
                                  x_type=PV,
                                  y_type=SST,
                                  var_type='input_current',
                                  var_ids=var_ids,
                                  plot_area=False,
                                  plot_colorbar=False,
                                  simple_label=True,
                                  plot_ylabel=pl,
                                  plot_xlabel=pl,
                                  ax=axarr[j, i],
                                  nosave=True)
    plt.tight_layout()
    plt.savefig('figure/input_current_all.pdf', transparent=True)
