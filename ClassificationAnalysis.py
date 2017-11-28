# -*- coding: utf-8 -*-
"""Run classification analysis."""
from __future__ import division

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from RateModel import Model

EXC = 0
PV = 1
SST = 2
VIP = 3

pops = [EXC, PV, SST, VIP]

names = ['E', 'PV', 'SST', 'VIP']

with open('Zingg_areadivision.pkl', 'rb') as f:
    div = pickle.load(f)

div_color_list = np.array([[31, 120, 180],
                           [166, 118, 29],
                           [253, 180, 98],
                           [117, 112, 179],
                           [231, 41, 138]]) / 255.
div_name_list = ['somatic',
                 'vis-aud',
                 'medial associa.',
                 'medial prefrontal',
                 'lateral']


def do_classification(layer):
    """Do classification analysis."""
    ext_params = {'layer': layer}
    model = Model(ext_params=ext_params, all_areas=True)
    p = model.p

    div_type_list = list()
    for area in p['areas']:
        for i, div_name in enumerate(div_name_list):
            if area in div[div_name]:
                div_type_list.append(i)
    div_type_list = np.array(div_type_list)

    clf = LinearDiscriminantAnalysis()

    def get_pred(clf, dims, shuffle=False):
        # Get mean prediction on hold-one-out data
        X = p['den_norm'][dims, :].T
        y = np.copy(div_type_list)

        if shuffle:
            np.random.shuffle(y)

        preds = list()
        # Looping over data points
        for i in range(len(y)):
            ind = range(len(y))
            ind.pop(i)
            X_train = X[ind, :]
            y_train = y[ind]
            clf.fit(X_train, y_train)
            preds.append((y[i] == clf.predict(X[np.newaxis, i, :]))[0])
        return np.mean(preds)

    dims_list = [[PV], [SST], [VIP], [PV, SST], [PV, VIP],
                 [SST, VIP], [PV, SST, VIP]]
    pred_list = [get_pred(clf, dims) for dims in dims_list]

    # Get confidence interval
    n_rep = 400
    pred_shuffle = np.array([
            get_pred(clf, [PV, SST, VIP], shuffle=True) for j in range(n_rep)])
    pred_low, pred_high = np.percentile(pred_shuffle, [2.5, 97.5])
    print(pred_low, pred_high)

    # Plot
    fs = 7
    fig = plt.figure(figsize=(4.2, 0.9))
    ax = fig.add_axes([.15, .25, .8, .65])
    width = 0.3
    ax.bar(np.arange(len(pred_list))-width/2, pred_list,
           width=width,
           color='blue',
           edgecolor='none')
    ax.set_xticks(np.arange(len(pred_list)))
    ax.plot([-1, len(pred_list)], [0.2, 0.2], '--', color='gray')
    ax.fill_between([-1, len(pred_list)], [pred_low] * 2, [pred_high] * 2,
                    facecolor='gray', alpha=0.2)
    xticklabels = list()
    for dims in dims_list:
        xticklabel = ''
        for j, dim in enumerate(dims):
            if j > 0:
                xticklabel += ','
            xticklabel += names[dim]
        xticklabels.append(xticklabel)

    ax.set_xticklabels(xticklabels, fontsize=fs)
    ax.set_xlim(-0.5, len(pred_list)-0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel('Cross-validated \nperformance', fontsize=fs)
    # ax.set_xlabel('Density information used', fontsize=6)
    plt.tick_params(axis='both', which='major', labelsize=fs)
    figname = 'classificationperf_L' + p['layer'].replace("/", "")
    plt.savefig('figure/'+figname+'.pdf', transparent=True)
    plt.show()

    def plot_decisionbound(clf, dims):
        assert(len(dims) == 2)
        X = p['den_norm'][dims, :].T
        y = div_type_list

        # Train with all data
        clf.fit(X, y)

        # Plotting decision regions
        x_min, x_max = 0, 2
        y_min, y_max = 0, 2
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.005),
                             np.arange(y_min, y_max, 0.005))

        fig = plt.figure(figsize=(1.5, 1.5))
        ax = fig.add_axes([.25, .25, .65, .65])

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        from matplotlib import colors
        cmap = colors.ListedColormap(div_color_list)
        bounds = [0, 1, 2, 3, 4, 5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        ax.imshow(Z, alpha=0.3, cmap=cmap, norm=norm,
                  extent=[x_min, x_max, y_min, y_max],
                  origin='lower', interpolation='nearest')

        div_colors = [div_color_list[j] for j in y]
        ax.scatter(X[:, 0], X[:, 1],
                   c=div_colors, edgecolor=div_colors, alpha=0.8, s=5)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks([0, 2])
        ax.set_yticks([0, 2])
        ax.set_xlabel(names[dims[0]], fontsize=fs, labelpad=-4)
        ax.set_ylabel(names[dims[1]], fontsize=fs, labelpad=-4)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        figname = ('decisionbounds' + names[dims[0]] + names[dims[1]] +
                   'L' + p['layer'].replace("/", ""))
        plt.savefig('figure/' + figname + '.pdf', transparent=True)
        plt.show()

    # Plot decision boundaries
    for dims in [[PV, SST], [PV, VIP], [SST, VIP]]:
        plot_decisionbound(clf, dims)
