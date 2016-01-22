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


# Read the image
im = np.rot90(scipy.misc.lena()[::2, ::2], 3)
shape = im.shape
# Rescale max to 1
im = im / np.max(im[:])

# Discretized spaces
discr_space = odl.uniform_discr([0, 0], shape, shape)

# Original image
orig = discr_space.element(im)

# Add noise
im = im + np.random.normal(0, 0.1, shape)
# Rescale back to [0, 1]
im = (im - np.min(im)) / (np.max(im) - np.min(im))

# Data: noisy image
noisy = discr_space.element(im)

# Gradient operator
grad = odl.DiscreteGradient(discr_space, method='forward')

# Matrix of operators
K = odl.ProductSpaceOperator([[odl.IdentityOperator(discr_space)], [grad]])

# Starting point
x = K.domain.zero()

# Operator norm
K_norm = 1.1 * odl.operator.oputils.power_method_opnorm(K, 200)

# Display partial
fig = plt.figure('iteration')
partial = odl.solvers.util.ForEachPartial(
    lambda result: result.show(fig=fig, show=False))
# partial = odl.solvers.util.ShowPartial(fig=fig, show=True, title='iteration')
partial &= odl.solvers.util.PrintIterationTimePartial()

# Run algorithms
chambolle_pock_solver(K, x,
                      f_cc_prox_l2_tv(K.range, noisy, lam=1 / 16),
                      g_prox_none(K.domain),
                      sigma=1 / K_norm,
                      tau=1 / K_norm,
                      niter=10, partial=partial)

# Display images
orig.show(title='original image')
noisy.show(title='noisy image')
# x.show(title='reconstruction')
plt.show()
