# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""An example of using the SPDHG algorithm to solve a TV deblurring problem
with Poisson noise. We exploit the smoothness of the data term to get 1/k^2
convergence on the dual part. We compare different algorithms for this problem
and visualize the results as in [CERS2017].


Reference
---------
[CERS2017] A. Chambolle, M. J. Ehrhardt, P. Richtarik and C.-B. Schoenlieb,
*Stochastic Primal-Dual Hybrid Gradient Algorithm with Arbitrary Sampling
and Imaging Applications*. ArXiv: http://arxiv.org/abs/1706.04957 (2017).
"""

from __future__ import division, print_function
import os
import odl
import odl.contrib.solvers.spdhg as spdhg
import odl.contrib.datasets.images as images
import odl.contrib.fom as fom
import brewer2mpl
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# create folder structure and set parameters
folder_out = '.'  # to be changed
filename = 'deblurring_1k2_dual'
nepoch = 300
niter_target = 2000
subfolder = '{}epochs'.format(nepoch)
simage = [408, 544]
image_raw = images.rings(shape=simage, gray=True)  # load image

filename = '{}_{}x{}'.format(filename, simage[0], simage[1])

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
X = odl.uniform_discr([0, 0], simage, simage)
groundtruth = 100 * X.element(image_raw)
clim = [0, 100]
tol_norm = 1.05

# create forward operators
Dx = odl.PartialDerivative(X, 0, pad_mode='symmetric')
Dy = odl.PartialDerivative(X, 1, pad_mode='symmetric')
kernel = images.blurring_kernel(shape=[15, 15])
convolution = spdhg.Blur2D(X, kernel)
K = odl.uniform_discr([0, 0], kernel.shape, kernel.shape)
kernel = K.element(kernel)

scale = 1e+3
A = odl.BroadcastOperator(Dx, Dy, scale / clim[1] * convolution)
Y = A.range

# create data
background = 200 * Y[2].one()
data = odl.phantom.poisson_noise(A[2](groundtruth) + background, seed=1807)

# save images and data
if not os.path.exists('{}/groundtruth.png'.format(folder_main)):
    spdhg.save_image(groundtruth, 'groundtruth', folder_main, 1, clim=clim)
    spdhg.save_image(data - background, 'data', folder_main, 2,
                     clim=[0, scale])
    spdhg.save_image(kernel, 'kernel', folder_main, 3)

alpha = 0.1  # set regularisation parameter
gamma = 0.99  # auxiliary step size parameter < 1

# set up functional f
f = odl.solvers.SeparableSum(
    odl.solvers.Huber(A[0].range, gamma=1),
    odl.solvers.Huber(A[1].range, gamma=1),
    1 / alpha * spdhg.KullbackLeiblerSmooth(A[2].range, data, background))

g = odl.solvers.IndicatorBox(X, clim[0], clim[1])  # set up functional g
obj_fun = f * A + g  # define objective function

mu_i = [1 / fi.grad_lipschitz for fi in f]  # strong convexity constants of fi
mu_f = min(mu_i)  # strong convexity constants of f

# create target / compute a saddle point
file_target = '{}/target.npy'.format(folder_main)
if not os.path.exists(file_target):

    # compute norm of operator
    normA = tol_norm * A.norm(estimate=True, xstart=odl.phantom.white_noise(X))

    sigma, tau = [gamma / normA] * 2  # set step size parameters
    x_opt, y_opt = X.zero(), Y.zero()  # initialise variables

    # define callback for visual output during iterations
    callback = (odl.solvers.CallbackPrintIteration(step=10, end=', ') &
                odl.solvers.CallbackPrintTiming(step=10, cumulative=True))

    # compute a saddle point with PDHG and time the reconstruction
    odl.solvers.pdhg(x_opt, f, g, A, tau, sigma, niter_target, y=y_opt,
                     callback=callback)

    # compute subgradients at saddle
    subx_opt = -A.adjoint(y_opt)
    suby_opt = A(x_opt)

    obj_opt = obj_fun(x_opt)  # compute objective at saddle

    # save saddle point
    np.save(file_target, (x_opt, y_opt, subx_opt, suby_opt, obj_opt, normA))

    # show saddle point and subgradients
    spdhg.save_image(x_opt, 'x_saddle', folder_main, 1, clim=clim)
    spdhg.save_image(y_opt[0], 'y_saddle[0]', folder_main, 2)
    spdhg.save_image(y_opt[1], 'y_saddle[1]', folder_main, 3)
    spdhg.save_image(y_opt[2], 'y_saddle[2]', folder_main, 4)
    spdhg.save_image(subx_opt, 'subx_saddle', folder_main, 5)
    spdhg.save_image(suby_opt[0], 'suby_saddle[0]', folder_main, 6)
    spdhg.save_image(suby_opt[1], 'suby_saddle[1]', folder_main, 7)
    spdhg.save_image(suby_opt[2], 'suby_saddle[2]', folder_main, 8)

else:
    (x_opt, y_opt, subx_opt, suby_opt, obj_opt, normA) = np.load(file_target)

# set norms of the primal and dual variable
dist_x = odl.solvers.L2NormSquared(X).translated(x_opt)
dist_y = odl.solvers.L2NormSquared(Y).translated(y_opt)


class CallbackStore(odl.solvers.Callback):
    """Callback to store function values"""

    def __init__(self, alg, iter_save, iter_plot):
        self.iter_save = iter_save
        self.iter_plot = iter_plot
        self.iter = 0
        self.alg = alg
        self.out = []

    def __call__(self, w):

        if self.iter in self.iter_save:
            obj = obj_fun(w[0])
            psnr = fom.psnr(w[0], groundtruth)
            psnr_opt = fom.psnr(w[0], x_opt)
            dx = dist_x(w[0])
            dy = dist_y(w[1])
            dist = dx + dy

            self.out.append({'obj': obj, 'dist': dist, 'dist_x': dx,
                             'dist_y': dy, 'psnr': psnr, 'psnr_opt': psnr_opt,
                             'iter': self.iter})

        if self.iter in self.iter_plot:
            fname = '{}_{}'.format(self.alg, self.iter)
            spdhg.save_image(w[0], fname, folder_today, 1, clim=clim)

        self.iter += 1


# number of subsets for each algorithm
nsub = {'pdhg': 1, 'da_pdhg': 1, 'da_spdhg_uni3': 3}

# number of iterations for each algorithm
niter, iter_save, iter_plot = {}, {}, {}
for a in nsub.keys():
    niter[a] = nepoch * nsub[a]
    iter_save[a] = range(0, niter[a] + 1, nsub[a])
    iter_plot[a] = list(np.array([10, 30, 50, 70, 100, 300]) * nsub[a])

# %% --- Run algorithms ---
for alg in ['pdhg', 'da_pdhg', 'da_spdhg_uni3']:
    print('======= ' + alg + ' =======')

    # clear variables in order not to use previous instances
    prob_subset, prob, sigma, sigma_tilde, tau, theta = [None] * 6

    np.random.seed(1807)  # set random seed so results are reproducable

    # create lists for subset division
    n = nsub[alg]
    (sub2ind, ind2sub) = spdhg.divide_1Darray_equally(range(len(Y)), n)

    if alg in ['pdhg', 'da_pdhg']:
        file_normA = '{}/norms_{}subsets.npy'.format(folder_main, 1)

        if not os.path.exists(file_normA):
            xstart = odl.phantom.white_noise(X)

            norm_estimate = A.norm(estimate=True, xstart=xstart)
            normA = [tol_norm * norm_estimate]

            np.save(file_normA, normA)

        else:
            normA = np.load(file_normA)

    elif alg in ['da_spdhg_uni3']:
        file_normA = '{}/norms_{}subsets.npy'.format(folder_main, n)

        if not os.path.exists(file_normA):
            xstart = odl.phantom.white_noise(X)

            norm_estimate = A[2].norm(estimate=True, xstart=xstart)
            normA = [2, 2, tol_norm * norm_estimate]

            np.save(file_normA, normA)

        else:
            normA = np.load(file_normA)

    # choose parameters for algorithm
    if alg == 'pdhg':
        prob_subset = [1] * n
        prob = [1] * len(Y)
        sigma = [gamma / normA[0]] * len(Y)
        tau = gamma / normA[0]

    elif alg == 'da_pdhg':
        prob_subset = [1] * n
        prob = [1] * len(Y)
        tau = gamma / normA[0]
        mu = [mu_f] * len(Y)
        sigma_tilde = mu_f / normA[0]

    elif alg in ['da_spdhg_uni3']:
        prob = [1 / n] * n
        prob_subset = prob
        tau = gamma / (n * max(normA))
        mu = mu_i
        sigma_tilde = min([m * p**2 / (tau * normAi**2 + 2 * m * p * (1 - p))
                           for p, m, normAi in zip(prob, mu_i, normA)])

    else:
        assert False, "Parameters not defined"

    # function that selects the indices every iteration
    def fun_select(k):
        return sub2ind[int(np.random.choice(n, 1, p=prob_subset))]

    # output function to be used within the iterations
    callback = (odl.solvers.CallbackPrintIteration(fmt='iter:{:4d}', step=n,
                                                   end=', ') &
                odl.solvers.CallbackPrintTiming(fmt='time: {:5.2f} s',
                                                cumulative=True, step=n) &
                CallbackStore(alg, iter_save[alg], iter_plot[alg]))

    x, y = X.zero(), Y.zero()  # initialise variables
    callback([x, y])

    if alg.startswith('pdhg'):
        spdhg.spdhg(x, f, g, A, tau, sigma, niter[alg], y=y, prob=prob,
                    fun_select=fun_select, callback=callback)

    elif alg.startswith('da_pdhg') or alg.startswith('da_spdhg'):
        spdhg.da_spdhg(x, f, g, A, tau, sigma_tilde, niter[alg], mu, prob=prob,
                       fun_select=fun_select, y=y, callback=callback)

    else:
        assert False, "Algorithm not defined"

    out = callback.callbacks[1].out

    np.save('{}/{}_output'.format(folder_npy, alg), (iter_save[alg],
            niter[alg], x, out, nsub[alg]))

# %% --- Analyse and visualise the output ---
algs = ['pdhg', 'da_pdhg', 'da_spdhg_uni3']

(iter_save_v, niter_v, image_v, out_v, nsub_v) = {}, {}, {}, {}, {}
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

for a in algs:
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
logy_plot = ['obj', 'obj_rel', 'dist_x', 'dist_y']

fig = plt.figure()
for plotx in ['linx', 'logx']:
    for meas in all_plots:
        print('============{}==={}'.format(plotx, meas))
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
bmap = brewer2mpl.get_map('Paired', 'Qualitative', 6)
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
mevery = [(i / 30., .15) for i in range(10)]  # how many markers to draw
msize = 9  # marker size

algs = ['pdhg', 'da_pdhg', 'da_spdhg_uni3']
label = ['PDHG', 'DA-PDHG', 'DA-SPDHG']
fig = []

# draw first figure
fig.append(plt.figure(0))
plt.clf()
xlim = [5, 300]
ylim = [1e-4, 5e+2]
meas = 'dist_y'
for k, a in enumerate(algs):
    x = epochs_save[a]
    y = out_resorted[a][meas] / out_resorted[a][meas][0]
    i = (np.less_equal(x, xlim[1]) & np.greater_equal(x, xlim[0]) &
         np.less_equal(y, ylim[1]) & np.greater_equal(y, ylim[0]))
    plt.loglog(x[i], y[i], color=colors[k], linestyle=lstyle, linewidth=lwidth,
               marker=marker[k], markersize=msize, markevery=mevery[k],
               label=label[k])

y = 3e+3 / x ** 2
plt.loglog(x[i], y[i], color='gray', linestyle=lstyle_help,
           linewidth=lwidth_help, label='$O(1/K^2)$')

plt.gca().set_xlabel('iterations [epochs]')
plt.gca().set_ylabel('dual distance')
plt.gca().yaxis.set_ticks(np.logspace(-3, 1, 3))
plt.legend(frameon=False)


# ### next figure
fig.append(plt.figure(1))
plt.clf()
ylim = [5e-5, 1e-0]
meas = 'dist_x'
for k, a in enumerate(algs):
    x = epochs_save[a]
    y = out_resorted[a][meas] / out_resorted[a][meas][0]
    i = (np.less_equal(x, xlim[1]) & np.greater_equal(x, xlim[0]) &
         np.less_equal(y, ylim[1]) & np.greater_equal(y, ylim[0]))
    plt.semilogy(x[i], y[i], color=colors[k], linestyle=lstyle,
                 linewidth=lwidth, label=label[k],
                 marker=marker[k], markersize=msize, markevery=mevery[k])

plt.gca().set_xlabel('iterations [epochs]')
plt.gca().set_ylabel('primal distance')
plt.gca().yaxis.set_ticks(np.logspace(-5, -1, 3))
plt.legend(frameon=False)

# %%
for i, fi in enumerate(fig):
    fi.savefig('{}/output{}.png'.format(folder_today, i), bbox_inches='tight')
