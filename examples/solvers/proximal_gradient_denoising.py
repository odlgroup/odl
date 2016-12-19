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

"""L1-regularized denoising using the proximal gradient solvers.

Solves the optimization problem

    min_x || x - g ||_1 + lam || grad(x) ||_2^2

Where ``grad`` is the spatial gradient operator and ``g`` is given noisy data.

The proximal gradient solvers are also known as ISTA and FISTA.
"""

import odl


# --- Set up problem definition --- #


# Define function space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300])

# Create phantom
data = odl.phantom.shepp_logan(space, modified=True)
data = odl.phantom.salt_pepper_noise(data)

# Create gradient operator
grad = odl.Gradient(space)


# --- Set up the inverse problem --- #

# Create data discrepancy by translating the l1 norm
l1_norm = odl.solvers.L1Norm(space)
data_discrepancy = l1_norm.translated(data)

# l2-squared norm of gradient
regularizer = 0.05 * odl.solvers.L2NormSquared(grad.range) * grad

# --- Select solver parameters and solve using proximal gradient --- #

# Select step-size that guarantees convergence.
gamma = 0.01

# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow(display_step=10))

# Run the algorithm (ISTA)
x = space.zero()
odl.solvers.proximal_gradient(
    x, f=data_discrepancy, g=regularizer, niter=200, gamma=gamma,
    callback=callback)

# Compare to accelerated version (FISTA) which is much faster
callback.reset()
x_acc = space.zero()
odl.solvers.accelerated_proximal_gradient(
    x_acc, f=data_discrepancy, g=regularizer, niter=50, gamma=gamma,
    callback=callback)

# Display images
data.show(title='Data')
x.show(title='L1 regularized reconstruction')
x_acc.show(title='L1 regularized reconstruction (accelerated)',
           force_show=True)
