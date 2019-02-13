# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""An example of using the SPDHG algorithm to solve a TV denoising problem
with Gaussian noise. We exploit the strong convexity of the data term to get
1/k^2 convergence on the primal part. We compare different algorithms for this
problem and visualize the results as in [CERS2017].

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
import odl
import brewer2mpl

# create folder structure and set parameters
folder_out = '.'  # to be changed
filename = 'ROF_1k2_primal'
nepoch = 300
niter_target = 2000
subfolder = '{}epochs'.format(nepoch)

folder_main = '{}/{}'.format(folder_out, filename)
if not os.path.exists(folder_main):
    os.makedirs(folder_main)

folder_today = '{}/{}'.format(folder_main, subfolder)
if not os.path.exists(folder_today):
    os.makedirs(folder_today)

folder_npy = '{}/npy'.format(folder_today)
if not os.path.exists(folder_npy):
    os.makedirs(folder_npy)

# create ground truth
image_gray = images.building(gray=True)
X = odl.uniform_discr([0, 0], image_gray.shape, image_gray.shape)
groundtruth = X.element(image_gray)
clim = [0, 1]

# create data
data = odl.phantom.white_noise(X, mean=groundtruth, stddev=0.1, seed=1807)

# save images and data
if not os.path.exists('{}/groundtruth.png'.format(folder_main)):
    spdhg.save_image(groundtruth, 'groundtruth', folder_main, 1, clim=clim)
    spdhg.save_image(data, 'data', folder_main, 2, clim=clim)

alpha = .12  # set regularisation parameter
gamma = 0.99  # gamma^2 is upper bound of step size constraint

# create forward operators
Dx = odl.PartialDerivative(X, 0, pad_mode='symmetric')
Dy = odl.PartialDerivative(X, 1, pad_mode='symmetric')
A = odl.BroadcastOperator(Dx, Dy)
Y = A.range

# set up functional f
f = odl.solvers.SeparableSum(*[odl.solvers.L1Norm(Yi) for Yi in Y])
# set up functional g
g = 1 / (2 * alpha) * odl.solvers.L2NormSquared(X).translated(data)

obj_fun = f * A + g  # define objective function
mu_g = 1 / alpha  # define strong convexity constants

# create target / compute a saddle point
file_target = '{}/target.npy'.format(folder_main)
if not os.path.exists(file_target):

    # compute a saddle point with PDHG and time the reconstruction
    callback = (odl.solvers.CallbackPrintIteration(step=10, end=', ') &
                odl.solvers.CallbackPrintTiming(step=10, cumulative=True))

    x_opt, y_opt = X.zero(), Y.zero()  # initialise variables
    normA = np.sqrt(8)  # compute norm of operator
    sigma, tau = (gamma / normA,) * 2  # set step size parameters

    # compute a saddle point with PDHG and time the reconstruction
    odl.solvers.pdhg(x_opt, f, g, A, tau, sigma, niter_target, y=y_opt,
                     callback=callback)

    # subgradients at saddle
    subx_opt = -A.adjoint(y_opt)
    suby_opt = A(x_opt)

    obj_opt = obj_fun(x_opt)  # objective value at saddle

    # save saddle point
    np.save(file_target, (x_opt, y_opt, subx_opt, suby_opt, obj_opt, normA))

    # show saddle point and subgradients
    spdhg.save_image(x_opt, 'x_saddle', folder_main, 1, clim=clim)
    spdhg.save_image(y_opt[0], 'y_saddle[0]', folder_main, 2)
    spdhg.save_image(subx_opt, 'subx_saddle', folder_main, 3)
    spdhg.save_image(suby_opt[0], 'suby_saddle[0]', folder_main, 4)

else:
    (x_opt, y_opt, subx_opt, suby_opt, obj_opt, normA) = np.load(file_target)

# set norms of the primal and dual variable
dist_x = odl.solvers.L2NormSquared(X).translated(x_opt)
dist_y = odl.solvers.L2NormSquared(Y).translated(y_opt)

# create Bregman distances for f and g
bregman_g = spdhg.bregman(g, x_opt, subx_opt)

# define Bregman distance for f and f_p
bregman_f = odl.solvers.SeparableSum(
    *[spdhg.bregman(fi.convex_conj, yi, ri)
      for fi, yi, ri in zip(f, y_opt, suby_opt)])


class CallbackStore(odl.solvers.util.callback.Callback):
    """Callback to store function values"""

    def __init__(self, alg, iter_save, iter_plot):
        self.iter_save = iter_save
        self.iter_plot = iter_plot
        self.iter = 0
        self.alg = alg
        self.ex, self.ey = X.zero(), Y.zero()
        self.out = []

    def __call__(self, w):
        if self.iter > 0:
            k = self.iter
            self.ex = 1 / k * ((k - 1) * self.ex + w[0])
            self.ey = 1 / k * ((k - 1) * self.ey + w[1])

        if self.iter in self.iter_save:
            obj = obj_fun(w[0])
            breg_x = bregman_g(w[0])
            breg_y = bregman_f(w[1])
            breg = breg_x + breg_y
            breg_ex = bregman_g(self.ex)
            breg_ey = bregman_f(self.ey)
            breg_erg = breg_ex + breg_ey
            dx = dist_x(w[0])
            dy = dist_y(w[1])
            dist = dx + dy
            dex = dist_x(self.ex)
            dey = dist_y(self.ey)
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
nsub = {'pdhg': 1, 'pa_pdhg': 1, 'pesquet_uni2': 2, 'spdhg_uni2': 2,
        'pa_spdhg_uni2': 2, 'odl': 1, 'pa_odl': 1}

# number of iterations for each algorithm
niter, iter_save, iter_plot = {}, {}, {}
for alg in nsub.keys():
    niter[alg] = nepoch * nsub[alg]
    iter_save[alg] = range(0, niter[alg] + 1, nsub[alg])
    iter_plot[alg] = list(np.array([10, 20, 30, 40, 100, 300]) * nsub[alg])

# %% --- Run algorithms ---
# TODO: ODL version to be included once the callback includes dual iterates
# for alg in ['pdhg', 'pesquet_uni2', 'pa_pdhg', 'spdhg_uni2', 'pa_spdhg_uni2',
# 'odl', 'pa_odl']:
for alg in ['pdhg', 'pesquet_uni2', 'pa_pdhg', 'spdhg_uni2', 'pa_spdhg_uni2']:
    print('======= ' + alg + ' =======')

    # clear variables in order not to use previous instances
    prob, sigma, tau, theta = [None] * 4

    # create lists for subset division
    n = nsub[alg]
    (sub2ind, ind2sub) = spdhg.divide_1Darray_equally(range(2), n)

    # set random seed so that results are reproducable
    np.random.seed(1807)

    # choose parameters for algorithm
    if alg == 'pdhg' or alg == 'pa_pdhg':
        prob_subset = [1] * n
        prob = [1] * len(Y)
        sigma = [gamma / normA] * len(Y)
        tau = gamma / normA

    elif alg == 'odl' or alg == 'pa_odl':
        sigma = gamma / normA
        tau = gamma / normA

    elif alg == 'pesquet_uni2':
        prob_subset = [1 / n] * n
        prob = [1 / n] * len(Y)
        sigma = [gamma / normA] * len(Y)
        tau = gamma / normA

    elif alg in ['spdhg_uni2'] or alg in ['pa_spdhg_uni2']:
        normAi = [2] * n
        prob_subset = [1 / n] * n
        prob = [1 / n] * len(Y)
        sigma = [gamma / nA for nA in normAi]
        tau = gamma / (n * max(normAi))

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
    callback([x, y])

    if alg.startswith('pdhg') or alg.startswith('spdhg'):
        spdhg.spdhg(x, f, g, A, tau, sigma, niter[alg], prob=prob, y=y,
                    fun_select=fun_select, callback=callback)

    elif alg.startswith('pa_pdhg') or alg.startswith('pa_spdhg'):
        spdhg.pa_spdhg(x, f, g, A, tau, sigma, niter[alg], mu_g, prob=prob,
                       y=y, fun_select=fun_select, callback=callback)

    elif alg.startswith('odl'):
        odl.solvers.pdhg(x, f, g, A, tau, sigma, niter[alg], y=y,
                         callback=callback)

    elif alg.startswith('pa_odl'):
        odl.solvers.pdhg(x, f, g, A, tau, sigma, niter[alg], y=y,
                         callback=callback, gamma_primal=mu_g)

    elif alg.startswith('pesquet'):
        spdhg.spdhg_pesquet(x, f, g, A, tau, sigma, niter[alg],
                            fun_select=fun_select, y=y, callback=callback)

    else:
        assert False, "Algorithm not defined"

    out = callback.callbacks[1].out

    np.save('{}/{}_output'.format(folder_npy, alg), (iter_save[alg],
            niter[alg], x, out, nsub[alg]))

# %% --- Analyse and visualise the output ---
algs = ['pdhg', 'pesquet_uni2', 'pa_pdhg', 'spdhg_uni2', 'pa_spdhg_uni2']

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

fig = plt.figure()
markers = plt.Line2D.filled_markers

all_plots = out_resorted[algs[0]].keys()
logy_plot = ['obj', 'obj_rel', 'dist_x', 'dist_y', 'breg', 'breg_y', 'breg_x',
             'ebreg', 'ebreg_x', 'ebreg_y']

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

        fig.savefig('{}/{}_{}.png'.format(folder_today, plotx, meas),
                    bbox_inches='tight')

# %% --- Prepapare visual output as in [1] ---
# set line width and style
lwidth = 2
lwidth_help = 2
lstyle = '-'
lstyle_help = '--'

# set colors using colorbrewer
bmap = brewer2mpl.get_map('Paired', 'Qualitative', 5)
colors = bmap.mpl_colors

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
mevery = [(i / 30., .1) for i in range(20)]  # how many markers to draw
msize = 9  # marker size

algs = ['pdhg', 'pa_pdhg', 'spdhg_uni2', 'pa_spdhg_uni2', 'pesquet_uni2']
label = ['PDHG', 'PA-PDHG', 'SPDHG', 'PA-SPDHG', 'Pesquet\\&Repetti']
fig = []

# draw first figure
fig.append(plt.figure(1))
plt.clf()
xlim = [1, 300]
ylim = [2e-1, 1e+3]
meas = 'dist_x'
alg_i = [0, 1, 3]
for j in alg_i:
    a = algs[j]
    x = epochs_save[a]
    y = out_resorted[a][meas]
    i = (np.less_equal(x, xlim[1]) & np.greater_equal(x, xlim[0]) &
         np.less_equal(y, ylim[1]) & np.greater_equal(y, ylim[0]))
    plt.loglog(x[i], y[i], color=colors[j], linestyle=lstyle, linewidth=lwidth,
               marker=marker[j], markersize=msize, markevery=mevery[j],
               label=label[j])

y = 5e+4 / np.array(iter_save_v[alg])**2
plt.loglog(x[i], y[i], color='gray', linestyle=lstyle_help,
           linewidth=lwidth_help, label=r'$\mathcal O(1/K^2)$')

plt.gca().set_xlabel('iterations [epochs]')
plt.gca().set_ylabel('primal distance')
plt.gca().yaxis.set_ticks(np.logspace(0, 2, 3))
plt.ylim((5e-1, 1e+3))
plt.legend(ncol=1, frameon=False)


# ### next figure
fig.append(plt.figure(2))
plt.clf()
ylim = [1e-5, 100]
meas = 'obj_rel'
alg_i = [0, 1, 2, 3, 4]
for j in alg_i:
    a = algs[j]
    x = epochs_save[a]
    y = out_resorted[a][meas]
    i = (np.less_equal(x, xlim[1]) & np.greater_equal(x, xlim[0]) &
         np.less_equal(y, ylim[1]) & np.greater_equal(y, ylim[0]))
    plt.loglog(x[i], y[i], color=colors[j], linestyle=lstyle, linewidth=lwidth,
               marker=marker[j], markersize=msize, markevery=mevery[j],
               label=label[j])

plt.gca().set_xlabel('iterations [epochs]')
plt.gca().set_ylabel('relative objective')
plt.gca().yaxis.set_ticks(np.logspace(-5, -1, 3))
plt.legend(frameon=False)

# %%
for i, fi in enumerate(fig):
    fi.savefig('{}/output{}.png'.format(folder_today, i), bbox_inches='tight')
