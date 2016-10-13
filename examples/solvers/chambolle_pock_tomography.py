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

"""Total variation tomography using the Chambolle-Pock solver.

Solves the optimization problem

    min_x  1/2 ||A(x) - g||_2^2 + lam || |grad(x)| ||_1

Where ``A`` is a parallel beam forward projector, ``grad`` the spatial
gradient and ``g`` is given noisy data.

For further details and a description of the solution method used, see
:ref:`chambolle_pock` in the ODL documentation.
"""

import numpy as np
import odl


# --- Set up the forward operator (ray transform) --- #


# Discrete reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300], dtype='float32')

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
# Detector: uniformly sampled, n = 558, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 558)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# The implementation of the ray transform to use, options:
# 'scikit'                    Requires scikit-image (can be installed by
#                             running ``pip install scikit-image``).
# 'astra_cpu', 'astra_cuda'   Require astra tomography to be installed.
#                             Astra is much faster than scikit. Webpage:
#                             https://github.com/astra-toolbox/astra-toolbox
impl = 'astra_cuda'

# Create the forward operator
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl=impl)


# --- Generate artificial data --- #


# Create phantom
discr_phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# Create sinogram of forward projected phantom with noise
data = ray_trafo(discr_phantom)
data += odl.phantom.white_noise(ray_trafo.range) * np.mean(data) * 0.1


# --- Set up the inverse problem --- #


# Initialize gradient operator
gradient = odl.Gradient(reco_space, method='forward')

# Column vector of two operators
op = odl.BroadcastOperator(ray_trafo, gradient)

# Do not use the g functional, set it to zero.
g = odl.solvers.ZeroFunctional(op.domain)

# Create functionals for the dual variable

# l2-squared data matching
l2_norm = odl.solvers.L2NormSquared(ray_trafo.range).translated(data)

# Isotropic TV-regularization i.e. the l1-norm
l1_norm = 0.03 * odl.solvers.L1Norm(gradient.range)

# Combine functionals, order must correspond to the operator K
f = odl.solvers.SeparableSum(l2_norm, l1_norm)


# --- Select solver parameters and solve using Chambolle-Pock --- #


# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * odl.power_method_opnorm(op)

niter = 100  # Number of iterations
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable
gamma = 0.2

# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())

# Choose a starting point
x = op.domain.zero()

# Run the algorithm
odl.solvers.chambolle_pock_solver(
    x, f, g, op, tau=tau, sigma=sigma, niter=niter, gamma=gamma,
    callback=callback)

# Display images
discr_phantom.show(title='original image')
data.show(title='convolved image')
x.show(title='deconvolved image', show=True)
