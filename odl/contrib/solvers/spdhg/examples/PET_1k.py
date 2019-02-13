# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""An example of using the SPDHG algorithm to solve a PET reconstruction
problem with total variation prior. As we do not exploit any smoothness here we
only expect a 1/k convergence of the ergodic sequence in a Bregman sense. We
compare different algorithms for this problem and visualize the results as in
[CERS2017].

Note that this example uses the ASTRA toolbox https://www.astra-toolbox.com/.

Reference
---------
[CERS2017] A. Chambolle, M. J. Ehrhardt, P. Richtarik and C.-B. Schoenlieb,
*Stochastic Primal-Dual Hybrid Gradient Algorithm with Arbitrary Sampling
and Imaging Applications*. ArXiv: http://arxiv.org/abs/1706.04957 (2017).
"""

from __future__ import division, print_function
import os
import odl.contrib.solvers.spdhg as spdhg
import odl.contrib.datasets.images as images
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import odl
import brewer2mpl

# create folder structure and set parameters
folder_out = '.'  # to be changed
filename = 'PET_1k'
nepoch = 300
niter_target = 2000
subfolder = '{}epochs'.format(nepoch)
nvoxelx = 250  # set problem size
filename = '{}_{}x{}'.format(filename, nvoxelx, nvoxelx)

folder_main = '{}/{}'.format(folder_out, filename)
if not os.path.exists(folder_main):
    os.makedirs(folder_main)

folder_today = '{}/{}'.format(folder_main, subfolder)
if not os.path.exists(folder_today):
    os.makedirs(folder_today)

folder_npy = '{}/npy'.format(folder_today)
if not os.path.exists(folder_npy):
    os.makedirs(folder_npy)

# create geometry of operator
X = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[nvoxelx, nvoxelx])

geometry = odl.tomo.parallel_beam_geometry(X, num_angles=200, det_shape=250)

G = odl.BroadcastOperator(*[odl.tomo.RayTransform(X, g, impl='astra_cpu')
                            for g in geometry])

# create ground truth
Y = G.range
groundtruth = X.element(images.resolution_phantom(shape=X.shape))
clim = [0, 1]
tol_norm = 1.05

# save images and data
file_data = '{}/data.npy'.format(folder_main)
if not os.path.exists(file_data):
    sino = G(groundtruth)

    support = X.element(groundtruth.ufuncs.greater(0))
    factors = -G(0.005 / X.cell_sides[0] * support)
    factors.ufuncs.exp(out=factors)

    counts_observed = (factors * sino).ufuncs.sum()
    counts_desired = 5e+6
    counts_background = 1e+6

    factors *= counts_desired / counts_observed

    sino_supp = sino.ufuncs.greater(0)
    smooth_supp = Y.element([gaussian_filter(s, sigma=[1, 2 / X.cell_sides[0]])
                             for s in sino_supp])
    background = 10 * smooth_supp + 10
    background *= counts_background / background.ufuncs.sum()
    data = odl.phantom.poisson_noise(factors * sino + background, seed=1807)

    arr = np.empty(3, dtype=object)
    arr[0] = data
    arr[1] = factors
    arr[2] = background
    np.save(file_data, arr)

    spdhg.save_image(groundtruth, 'groundtruth', folder_main, 1, clim=clim)

    fig2 = plt.figure(2)
    fig2.clf()
    i = 11
    plt.plot((sino[i]).asarray()[0], label='G(x)')
    plt.plot((factors[i] * sino[i]).asarray()[0], label='factors * G(x)')
    plt.plot(data[i].asarray()[0], label='data')
    plt.plot(background[i].asarray()[0], label='background')
    plt.legend()

    fig2.savefig('{}/components1D.png'.format(folder_main),
                 bbox_inches='tight')

else:
    (data, factors, background) = np.load(file_data)

# data fit
f = odl.solvers.SeparableSum(
    *[odl.solvers.KullbackLeibler(Yi, yi).translated(-ri)
      for Yi, yi, ri in zip(Y, data, background)])
# TODO: should be ideally like
# f = odl.solvers.KullbackLeibler(Y, data).translated(-background)

# prior and regularisation parameter
g = spdhg.TotalVariationNonNegative(X, alpha=2e-1)
g.prox_options['niter'] = 20

# operator
A = odl.BroadcastOperator(*[fi * Gi for fi, Gi in zip(factors, G)])

obj_fun = f * A + g  # objective functional
gamma = 0.99  # square root of step size upper bound

# create target / compute a saddle point
file_target = '{}/target.npy'.format(folder_main)
if not os.path.exists(file_target):
    file_normA = '{}/norms_{}subsets.npy'.format(folder_main, 1)
    if not os.path.exists(file_normA):
        # compute norm of operator
        normA = [tol_norm * A.norm(estimate=True)]
        np.save(file_normA, normA)

    else:
        normA = np.load(file_normA)

    # create callback
    callback = (odl.solvers.CallbackPrintIteration(step=10, end=', ') &
                odl.solvers.CallbackPrintTiming(step=10, cumulative=True))

    sigma, tau = [gamma / normA[0]] * 2  # step size parameters
    x_opt, y_opt = X.zero(), Y.zero()  # initialise variables
    g.prox_options['p'] = None

    # compute a saddle point with PDHG and time the reconstruction
    odl.solvers.pdhg(x_opt, f, g, A, tau, sigma, niter_target, y=y_opt,
                     callback=callback)

    # compute the subgradients of the saddle point
    subx_opt = -A.adjoint(y_opt)
    suby_opt = A(x_opt)

    # show saddle point and subgradients
    spdhg.save_image(x_opt, 'x_saddle', folder_main, 1, clim=clim)
    spdhg.save_signal(y_opt[0], 'y_saddle[0]', folder_main, 2)
    spdhg.save_image(subx_opt, 'q_saddle', folder_main, 3)
    spdhg.save_signal(suby_opt[0], 'r_saddle[0]', folder_main, 4)

    obj_opt = obj_fun(x_opt)  # objective value at the saddle point

    # save saddle point
    np.save(file_target, (x_opt, y_opt, subx_opt, suby_opt, obj_opt))

else:
    (x_opt, y_opt, subx_opt, suby_opt, obj_opt) = np.load(file_target)

dist_x = odl.solvers.L2NormSquared(X).translated(x_opt)  # primal distance
dist_y = odl.solvers.L2NormSquared(Y).translated(y_opt)  # dual distance
bregman_g = spdhg.bregman(g, x_opt, subx_opt)  # primal Bregman distance
# TODO: should be like: bregman_g = g.bregman(x_opt, subgrad=subx_opt)

# dual Bregman distance
bregman_f = odl.solvers.SeparableSum(
    *[spdhg.bregman(fi.convex_conj, yi, ri)
      for fi, yi, ri in zip(f, y_opt, suby_opt)])
# TODO: should be like: bregman_f = f.bregman(y_opt, subgrad=subx_opt)


class CallbackStore(odl.solvers.Callback):
    """Callback to store function values"""

    def __init__(self, alg, iter_save, iter_plot):
        self.iter_save = iter_save
        self.iter_plot = iter_plot
        self.iter = 0
        self.alg = alg
        self.ergx, self.ergy = X.zero(), Y.zero()
        self.out = []

    def __call__(self, w, **kwargs):

        if self.iter > 0:
            k = self.iter
            self.ergx = 1 / k * ((k - 1) * self.ergx + w[0])
            self.ergy = 1 / k * ((k - 1) * self.ergy + w[1])

        if self.iter in self.iter_save:
            obj = obj_fun(w[0])
            breg_x = bregman_g(w[0])
            breg_y = bregman_f(w[1])
            breg = breg_x + breg_y
            breg_ex = bregman_g(self.ergx)
            breg_ey = bregman_f(self.ergy)
            breg_erg = breg_ex + breg_ey
            dx = dist_x(w[0])
            dy = dist_y(w[1])
            dist = dx + dy
            dex = dist_x(self.ergx)
            dey = dist_y(self.ergy)
            dist_erg = dex + dey

            self.out.append({'obj': obj, 'breg': breg, 'breg_x': breg_x,
                             'breg_y': breg_y, 'breg_erg': breg_erg,
                             'breg_ex': breg_ex, 'breg_ey': breg_ey,
                             'dist': dist, 'dist_x': dx, 'dist_y': dy,
                             'dist_erg': dist_erg, 'dist_ex': dex,
                             'dist_ey': dey, 'iter': self.iter})

        if self.iter in self.iter_plot:
            fname = '{}_{}'.format(self.alg, self.iter)
            spdhg.save_image(w[0], fname, folder_today, 1, clim=clim)

        self.iter += 1


# number of subsets for each algorithm
nsub = {'pdhg': 1, 'pesquet10': 10, 'pesquet50': 50, 'pesquet200': 200,
        'spdhg10': 10, 'spdhg50': 50, 'spdhg200': 200}

# number of iterations for each algorithm
niter, iter_save, iter_plot = {}, {}, {}
for a in nsub.keys():
    niter[a] = nepoch * nsub[a]
    iter_save[a] = range(0, niter[a] + 1, nsub[a])
    iter_plot[a] = list(np.array([5, 10, 20, 30, 40, 50, 100, 300]) * nsub[a])

# %% --- Run algorithms ---
for alg in ['pdhg', 'pesquet10', 'pesquet50', 'spdhg10', 'spdhg50']:
    print('======= ' + alg + ' =======')

    # clear variables in order not to use previous instances
    prob, sigma, tau = [None] * 3

    # create lists for subset division
    n = nsub[alg]
    (sub2ind, ind2sub) = spdhg.divide_1Darray_equally(range(len(A)), n)

    if alg == 'pdhg' or alg[0:5] == 'spdhg':
        file_normA = '{}/norms_{}subsets.npy'.format(folder_main, n)

    elif alg[0:7] == 'pesquet':
        file_normA = '{}/norms_{}subsets.npy'.format(folder_main, 1)

    if not os.path.exists(file_normA):
        A_subsets = [odl.BroadcastOperator(*[A[i] for i in subset])
                     for subset in sub2ind]
        normA = [tol_norm * Ai.norm(estimate=True) for Ai in A_subsets]
        np.save(file_normA, normA)

    else:
        normA = np.load(file_normA)

    # set random seed so that results are reproducable
    np.random.seed(1807)

    # choose parameters for algorithm
    if alg == 'pdhg':
        prob_subset = [1] * n
        prob = [1] * Y.size
        sigma = [gamma / normA[0]] * Y.size
        tau = gamma / normA[0]

    elif alg.startswith('pesquet'):
        prob_subset = [1 / n] * n
        prob = [1 / n] * Y.size
        sigma = [gamma / normA[0]] * Y.size
        tau = gamma / normA[0]

    elif alg.startswith('spdhg'):
        prob_subset = [1 / n] * n
        prob = [1 / n] * Y.size
        sigma = [gamma / normA[ind2sub[i][0]] for i in range(Y.size)]
        tau = gamma / (n * np.max(normA))

    else:
        assert False, "Parameters not defined"

    # function that selects the indices every iteration
    def fun_select(k):
        return sub2ind[int(np.random.choice(n, 1, p=prob_subset))]

    # output function to be used within the iterations
    callback = (odl.solvers.CallbackPrintIteration(fmt='iter:{:4d}', step=n,
                                                   end=', ') &
                odl.solvers.CallbackPrintTiming(fmt='time/iter: {:5.2f} s',
                                                step=n, end=', ') &
                odl.solvers.CallbackPrintTiming(fmt='time: {:5.2f} s',
                                                cumulative=True, step=n) &
                CallbackStore(alg, iter_save[alg], iter_plot[alg]))

    x, y = X.zero(), Y.zero()  # initialise variables
    callback([x, y])  # compute statistic for initialisation
    g.prox_options['p'] = None

    if alg.startswith('pdhg') or alg.startswith('spdhg'):
        spdhg.spdhg(x, f, g, A, tau, sigma, niter[alg], prob=prob, y=y,
                    fun_select=fun_select, callback=callback)

    elif alg.startswith('pesquet'):
        spdhg.spdhg_pesquet(x, f, g, A, tau, sigma, niter[alg], y=y,
                            fun_select=fun_select, callback=callback)

    else:
        assert False, "Algorithm not defined"

    out = callback.callbacks[1].out

    np.save('{}/{}_output'.format(folder_npy, alg), (iter_save[alg],
            niter[alg], x, out, nsub[alg]))

# %% --- Analyse and visualise the output ---
algs = ['pdhg', 'pesquet10', 'pesquet50', 'spdhg10', 'spdhg50']

iter_save_v, niter_v, image_v, out_v, nsub_v = {}, {}, {}, {}, {}
for a in algs:
    (iter_save_v[a], niter_v[a], image_v[a], out_v[a], nsub_v[a]) = np.load(
        '{}/{}_output.npy'.format(folder_npy, a))

epochs_save = {a: np.array(iter_save_v[a]) / np.float(nsub_v[a]) for a in algs}

out_resorted = {}
for a in algs:
    print('==== ' + a)
    out_resorted[a] = {}
    K = len(iter_save_v[a])

    for meas in out_v[a][0].keys():  # quality measures
        print('    ==== ' + meas)
        out_resorted[a][meas] = np.nan * np.ones(K)

        for k in range(K):  # iterations
            out_resorted[a][meas][k] = out_v[a][k][meas]

    meas = 'obj_rel'
    print('    ==== ' + meas)
    out_resorted[a][meas] = np.nan * np.ones(K)

    for k in range(K):  # iterations
        out_resorted[a][meas][k] = ((out_v[a][k]['obj'] - obj_opt) /
                                    (out_v[a][0]['obj'] - obj_opt))

for a in algs:  # algorithms
    for meas in out_resorted[a].keys():  # quality measures
        for k in range(K):  # iterations
            if out_resorted[a][meas][k] <= 0:
                out_resorted[a][meas][k] = np.nan

matplotlib.rc('text', usetex=False)  # set latex options

fig = plt.figure()
for a in algs:
    spdhg.save_image(image_v[a], a, folder_today, 1, clim=clim)

markers = plt.Line2D.filled_markers

all_plots = out_resorted[algs[0]].keys()
logy_plot = ['obj', 'obj_rel', 'dist', 'dist_x', 'dist_y', 'dist_erg',
             'dist_ex', 'dist_ey', 'breg', 'breg_y', 'breg_x', 'breg_erg',
             'breg_ex', 'breg_ey']

for plotx in ['linx', 'logx']:
    for meas in all_plots:
        print('============ ' + plotx + ' === ' + meas + ' ============')
        fig = plt.figure(1)
        plt.clf()

        if plotx == 'linx':
            if meas in logy_plot:
                for a in algs:
                    x = epochs_save[a]
                    y = out_resorted[a][meas]
                    plt.semilogy(x, y, linewidth=3, label=a)
            else:
                for j, a in enumerate(algs):
                    x = epochs_save[a]
                    y = out_resorted[a][meas]
                    plt.plot(x, y, linewidth=3, marker=markers[j],
                             markersize=7, markevery=.1, label=a)

        elif plotx == 'logx':
            if meas in logy_plot:
                for a in algs:
                    x = epochs_save[a][1:]
                    y = out_resorted[a][meas][1:]
                    plt.loglog(x, y, linewidth=3, label=a)
            else:
                for j, a in enumerate(algs):
                    x = epochs_save[a][1:]
                    y = out_resorted[a][meas][1:]
                    plt.semilogx(x, y, linewidth=3, marker=markers[j],
                                 markersize=7, markevery=.1, label=a)

        plt.title('{} v iteration'.format(meas))
        h = plt.gca()
        h.set_xlabel('epochs')
        plt.legend(loc='best')

        fig.savefig('{}/{}_{}_{}.png'.format(folder_today, filename, plotx,
                    meas), bbox_inches='tight')

# %% --- Prepapare visual output as in [1] ---
# set line width and style
lwidth = 2
lwidth_help = 2
lstyle = '-'
lstyle_help = '--'
# set colors using colorbrewer
bmap = brewer2mpl.get_map('Paired', 'Qualitative', 7)
colors = bmap.mpl_colors
colors.pop(1)
# set latex options
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

# set font
fsize = 15
font = {'family': 'serif', 'size': fsize}
matplotlib.rc('font', **font)
matplotlib.rc('axes', labelsize=fsize)  # fontsize of x and y labels
matplotlib.rc('xtick', labelsize=fsize)  # fontsize of xtick labels
matplotlib.rc('ytick', labelsize=fsize)  # fontsize of ytick labels
matplotlib.rc('legend', fontsize=fsize)  # legend fontsize

# markers
marker = ('o', 'v', 's', 'p', 'd')  # set markers
mevery = [(i / 30., .15) for i in range(20)]  # how many markers to draw
msize = 9  # marker size

algs = ['pdhg', 'spdhg10', 'spdhg50', 'pesquet10', 'pesquet50']
label = ['PDHG', 'SPDHG (10 subsets)', 'SPDHG (50)', 'Pesquet\\&Repetti (10)',
         'Pesquet\\&Repetti (50)']
fig = []

# ### draw figures
fig.append(plt.figure(0))
plt.clf()
xlim = [1, 300]
ylim = [2e-3, 2e+1]
meas = 'breg_erg'
for k, a in enumerate(algs[:3]):
    x = epochs_save[a]
    y = out_resorted[a][meas]
    i = (np.less_equal(x, xlim[1]) & np.greater_equal(x, xlim[0]) &
         np.less_equal(y, ylim[1]) & np.greater_equal(y, ylim[0]))
    plt.loglog(x[i], y[i], color=colors[k], linestyle=lstyle, linewidth=lwidth,
               marker=markers[k], markersize=msize, markevery=mevery[k],
               label=label[k])

y = 1e+2 / x
i = (np.less_equal(x, xlim[1]) & np.greater_equal(x, xlim[0]) &
     np.less_equal(y, ylim[1]) & np.greater_equal(y, ylim[0]))
plt.loglog(x[i], y[i], color='gray', linestyle=lstyle_help,
           linewidth=lwidth_help, label='$O(1/K)$')

plt.gca().set_xlabel('iterations [epochs]')
plt.gca().set_ylabel('ergodic Bregman distance')
plt.gca().yaxis.set_ticks(np.logspace(-2, 1, 4))
plt.legend(frameon=False)

# ### next figure
fig.append(plt.figure(1))
plt.clf()
ylim = [1e-9, 2]
meas = 'obj_rel'
for k, a in enumerate(algs):
    x = epochs_save[a]
    y = out_resorted[a][meas]
    i = (np.less_equal(x, xlim[1]) & np.greater_equal(x, xlim[0]) &
         np.less_equal(y, ylim[1]) & np.greater_equal(y, ylim[0]))
    plt.semilogy(x[i], y[i], color=colors[k], linestyle=lstyle,
                 linewidth=lwidth, marker=marker[k], markersize=msize,
                 markevery=mevery[k], label=label[k])

plt.gca().set_xlabel('iterations [epochs]')
plt.gca().set_ylabel('relative objective')
plt.gca().yaxis.set_ticks(np.logspace(-8, 0, 3))
plt.legend(frameon=False)

# %%
for i, fi in enumerate(fig):
    fi.savefig('{}/{}_output{}.png'.format(folder_today, filename, i),
               bbox_inches='tight')
