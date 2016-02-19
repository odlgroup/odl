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

"""Total variation base image denoising using the Chambolle-Pock solver.

Let X and Y be finite-dimensional Hilbert spaces and K a linear mapping
from X to Y with induce norm ||K||. The (primal) minimization problem we
want to solve is

    min_{x in X} F(K x) + G(x)

where the proper, convex, lower-semicontinuous functionals
F : Y -> [0, +infinity] and G : X -> [0, +infinity] are given
by an l2-data fitting term regularized by isotropic total variation

    F(K x) = 1/2 ||x - g||_2^2 + lam || |grad(x)| ||_1

and by the indicator function for the set fo non-negative components of x

   G(x) = {0 if x >=0, infinity if x < 0} ,

respectively. Here, g denotes the image to denoise, ||.||_2 the l2-norm,
||.||_1 the l1-semi-norm, grad the spatial gradient, lam the regularization
parameter, |.| the point-wise magnitude across the vector components of
grad(x), and K is a column vector of operators K = (id, grad)^T with
identity mapping id.

In order to use the Chambolle-Pock solver, we have to create the column
operator K, choose a starting point x, create the proximal operator for G,
create the proximal operator for the convex conjugate of F, choose the
step sizes tau and sigma such that tau sigma ||K||_2^2 < 1, and set the
total number of iterations.

For details see :ref:`chambolle_pock`, :ref:`proximal_operators`, and
references therein.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt

import odl


# TODO: Use BroadCastOperator instead of ProductSpaceOperator

# Read test image: use only every second pixel, convert integer to float,
# and rotate to get the image upright
image = np.rot90(scipy.misc.ascent()[::2, ::2], 3).astype('float')
shape = image.shape

# Rescale max to 1
image /= image.max()

# Discretized spaces
discr_space = odl.uniform_discr([0, 0], shape, shape)

# Original image
orig = discr_space.element(image)

# Add noise
image += np.random.normal(0, 0.1, shape)

# Data of noisy image
noisy = discr_space.element(image)

# Gradient operator
gradient = odl.Gradient(discr_space, method='forward')

# Matrix of operators
op = odl.ProductSpaceOperator([[odl.IdentityOperator(discr_space)],
                               [gradient]])

# Starting point
x = op.domain.zero()

# Operator norm
prod_op_norm = 1.1 * odl.operator.oputils.power_method_opnorm(op, 100)
print('Norm of the product space operator: {}'.format(prod_op_norm))

# Proximal operators related to the dual variable

# l2-data matching
prox_convconj_l2 = odl.solvers.proximal_convexconjugate_l2(discr_space,
                                                           lam=1, g=noisy)

# TV-regularization: l1-semi norm of grad(x)
prox_convconj_l1 = odl.solvers.proximal_convexconjugate_l1(gradient.range,
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

# Number of iterations
niter = 100

# Step size for the proximal operator for the primal variable x
tau = 1 / prod_op_norm

# Step size for the proximal operator for the dual variable y
sigma = 1 / prod_op_norm

# Run algorithms (and display intermediates)
odl.solvers.chambolle_pock_solver(
    op, x, tau=tau, sigma=sigma, proximal_primal=proximal_primal,
    proximal_dual=proximal_dual, niter=niter, partial=partial)

# Display images
orig.show(title='original image')
noisy.show(title='noisy image')
x.show(title='reconstruction')
plt.show()
