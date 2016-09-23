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

"""Total variation denoising using the Chambolle-Pock solver.

Solves the optimization problem

    min_{x >= 0}  1/2 ||x - g||_2^2 + lam || |grad(x)| ||_1

Where ``grad`` the spatial gradient and ``g`` is given noisy data.

For further details and a description of the solution method used, see
:ref:`chambolle_pock` in the ODL documentation.
"""

import numpy as np
import scipy
import odl

# Read test image: use only every second pixel, convert integer to float,
# and rotate to get the image upright
image = np.rot90(scipy.misc.ascent()[::2, ::2], 3).astype('float')
shape = image.shape

# Rescale max to 1
image /= image.max()

# Discretized spaces
space = odl.uniform_discr([0, 0], shape, shape)

# Original image
orig = space.element(image)

# Add noise
image += np.random.normal(0, 0.1, shape)

# Data of noisy image
noisy = space.element(image)

# Gradient operator
gradient = odl.Gradient(space, method='forward')

# Matrix of operators
op = odl.BroadcastOperator(odl.IdentityOperator(space), gradient)

# Set up the functionals

# l2-data matching
l2_norm = odl.solvers.L2NormSquared(space).translated(noisy)

# Isotropic TV-regularization: l1-norm of grad(x)
l1_norm = 0.2 * odl.solvers.L1Norm(gradient.range)

# Combine functionals: the order must match the order of operators in K
f = odl.solvers.SeparableSum(l2_norm, l1_norm)

# Non-negativity constraint
g = odl.solvers.IndicatorNonnegativity(op.domain)


# --- Select solver parameters and solve using Chambolle-Pock --- #


# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * odl.power_method_opnorm(op, xstart=noisy)

niter = 400  # Number of iterations
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable

# Optional: pass callback objects to solver
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow(display_step=20))

# Starting point
x = op.domain.zero()

# Run algorithms (and display intermediates)
odl.solvers.chambolle_pock_solver(
    x, f, g, op, tau=tau, sigma=sigma, niter=niter, callback=callback)

# Display images
orig.show(title='original image')
noisy.show(title='noisy image')
x.show(title='reconstruction', show=True)
