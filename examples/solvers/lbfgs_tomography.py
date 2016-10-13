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

"""Tomography using the `bfgs_method` solver.

Solves the optimization problem

    min_x ||A(x) - g||_2^2

Where ``A`` is a parallel beam forward projector, ``x`` the result and
 ``g`` is given noisy data.
"""

import numpy as np
import odl


# --- Set up the forward operator (ray transform) --- #


# Discrete reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 200 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[200, 200])

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 400, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, np.pi, 400)

# Detector: uniformly sampled, n = 400, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 400)
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

# --- Set up optimization problem and solve --- #

# Create objective functional ||Ax - b||_2^2 as composition of l2 norm squared
# and the residual operator.
obj_fun = odl.solvers.L2NormSquared(ray_trafo.range) * (ray_trafo - data)

# Create line search
line_search = 1.0
# line_search = odl.solvers.BacktrackingLineSearch(obj_fun)

# Create initial estimate of the inverse Hessian by a diagonal estimate
opnorm = odl.power_method_opnorm(ray_trafo)
hessinv_estimate = odl.ScalingOperator(reco_space, 1 / opnorm ** 2)

# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())

# Pick parameters
maxiter = 20
num_store = 5  # only save some vectors (Limited memory)

# Choose a starting point
x = ray_trafo.domain.zero()

# Run the algorithm
odl.solvers.bfgs_method(
    obj_fun, x, line_search=line_search, maxiter=maxiter, num_store=num_store,
    hessinv_estimate=hessinv_estimate, callback=callback)

odl.solvers.douglas_rachford_pd

# Display images
discr_phantom.show(title='original image')
data.show(title='sinogram')
x.show(title='reconstructed image', show=True)
