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


# TODO: Use BroadCastOperator instead of ProductSpaceOperator

# Read the image
image = np.rot90(scipy.misc.ascent()[::2, ::2], 3).astype('float')
shape = image.shape
# Rescale max to 1
image /= np.max(image[:])

# Discretized spaces
discr_space = odl.uniform_discr([0, 0], shape, shape)

# Original image
orig = discr_space.element(image)

# Add noise and rescale back to [0, 1]
image += np.random.normal(0, 0.1, shape)
image = (image - np.min(image)) / (np.max(image) - np.min(image))

# Data vector of noisy image
noisy = discr_space.element(image)

# Gradient operator
grad = odl.Gradient(discr_space, method='forward')

# Matrix of operators
op = odl.ProductSpaceOperator([[odl.IdentityOperator(discr_space)],
                               [grad]])

# Starting point
x = op.domain.zero()

# Operator norm
prod_op_norm = 1.1 * odl.operator.oputils.power_method_opnorm(op, 200)
print('Norm of the product space operator: {}'.format(prod_op_norm))

# Proximal operators related to the dual variable

# l2-data matching
prox_convconj_l2 = odl.solvers.proximal_convexconjugate_l2(discr_space,
                                                           lam=1, g=noisy)

# TV-regularization
prox_convconj_l1 = odl.solvers.proximal_convexconjugate_l1(grad.range,
                                                           lam=1/16)

# Combine proximal operators: the order must match the order of operators in K
proximal_dual = odl.solvers.combine_proximals([prox_convconj_l2,
                                               prox_convconj_l1])

# Proximal operator related to the primal variable

# Non-negativity constraint
proximal_primal = odl.solvers.proximal_nonnegativity(op.domain)

# Optional: pass partial objects to solver
partial = (odl.solvers.PrintIterationPartial() &
           odl.solvers.util.PrintTimingPartial() &
           odl.solvers.util.ShowPartial())

# Run algorithms (and display intermediates)
odl.solvers.chambolle_pock_solver(
    op, x, tau=1 / prod_op_norm, sigma=1 / prod_op_norm,
    proximal_primal=proximal_primal, proximal_dual=proximal_dual,
    niter=100, partial=partial)

# Display images
orig.show(title='original image')
noisy.show(title='noisy image')
x.show(title='reconstruction')
plt.show()
