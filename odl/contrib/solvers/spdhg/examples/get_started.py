# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""A simple example to get started with SPDHG [CERS2017]. The example at hand
solves the ROF denoising problem.

Reference
---------
[CERS2017] A. Chambolle, M. J. Ehrhardt, P. Richtarik and C.-B. Schoenlieb,
*Stochastic Primal-Dual Hybrid Gradient Algorithm with Arbitrary Sampling
and Imaging Applications*. ArXiv: http://arxiv.org/abs/1706.04957 (2017).
"""

from __future__ import division, print_function
import odl
import odl.contrib.solvers.spdhg as spdhg
import odl.contrib.datasets.images as images
import numpy as np

# set ground truth and data
image_gray = images.building(gray=True)
X = odl.uniform_discr([0, 0], image_gray.shape, image_gray.shape)
groundtruth = X.element(image_gray)
data = odl.phantom.white_noise(X, mean=groundtruth, stddev=0.1, seed=1807)

# set parameter
alpha = .12  # regularisation parameter
nepoch = 100

# set functionals and operator
A = odl.BroadcastOperator(*[odl.PartialDerivative(X, d, pad_mode='symmetric')
                            for d in [0, 1]])
f = odl.solvers.SeparableSum(*[odl.solvers.L1Norm(Yi) for Yi in A.range])
g = 1 / (2 * alpha) * odl.solvers.L2NormSquared(X).translated(data)

# set sampling
n = 2  # number of subsets
prob = [1 / n] * n  # probablity that a subset gets selected
S = [[0], [1]]  # all possible subsets to select from


def fun_select(k):  # subset selection function
    return S[int(np.random.choice(n, 1, p=prob))]


# set parameters for algorithm
Ai_norm = [2, 2]
gamma = 0.99
sigma = [gamma / a for a in Ai_norm]
tau = gamma / (n * max(Ai_norm))

# callback for output during the iterations
cb = (odl.solvers.CallbackPrintIteration(fmt='iter:{:4d}', step=n, end=', ') &
      odl.solvers.CallbackPrintTiming(fmt='time: {:5.2f} s', cumulative=True,
                                      step=n))

# initialise variable and run algorithm
x = X.zero()
niter = 2 * nepoch
spdhg.spdhg(x, f, g, A, tau, sigma, niter, prob=prob, fun_select=fun_select,
            callback=cb)

# show data and output
data.show()
x.show()
