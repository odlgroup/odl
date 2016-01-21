# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Denoising using Chambolle-Pock solver."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import numpy as np
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt

# Internal
import odl
from odl.solvers import chambolle_pock_solver, f_cc_prox_l2_tv, g_prox_none


# TODO: change test image
impath = '../temp/barbara.png'

# Read the image
im = np.rot90(scipy.misc.imread(impath)[::2, ::2], 3)
shape = im.shape
# Resacel to [0, 1]
im = im / np.max(im[:])
# Add noise
imnoise = im + np.random.normal(0, 0.05, shape)
imnoise = imnoise - np.min(imnoise[:])

# Discretized spaces
discr_space = odl.uniform_discr([0, 0], shape, shape)

# Original image
orig = discr_space.element(im)
# g0.show(title='original image')

# Data: noisy image
noisy = discr_space.element(imnoise)
# g.show(title='noisy image')

# Gradient operator
grad = odl.DiscreteGradient(discr_space, method='forward')

# Matrix of operators
K = odl.ProductSpaceOperator([[odl.IdentityOperator(discr_space)], [grad]])

# Operator norm
K_norm = 1.1 * odl.operator.oputils.power_method_opnorm(K, 200)

# Display partial
fig = plt.figure('iterations')
partial = odl.solvers.util.ForEachPartial(
    lambda (result, iter): result.show(
        title='iteration number:{}'.format(iter), fig=fig))

rec = chambolle_pock_solver(K, f_cc_prox_l2_tv(K.range, noisy, lam=1 / 16),
                            g_prox_none(K.domain),
                            sigma=1 / K_norm, tau=1 / K_norm,
                            niter=40, partial=partial)[0]
rec.show(title='reconstruction')
plt.show()
