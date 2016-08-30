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

"""Tomography with TV regularization using the ProxImaL solver.

Solves the optimization problem

    min_{0 <= x <= 1}  ||A(x) - g||_2^2 + 0.2 || |grad(x)| ||_1

Where ``A`` is a parallel beam forward projector, ``grad`` the spatial
gradient and ``g`` is given noisy data.
"""

import numpy as np
import odl
import proximal


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
# 'astra_cpu', 'astra_cuda'   Requires astra tomography to be installed.
#                             Astra is much faster than scikit. Webpage:
#                             https://github.com/astra-toolbox/astra-toolbox
impl = 'astra_cuda'

# Initialize the ray transform (forward projection).
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl=impl)

# Convert ray transform to proximal language operator
proximal_lang_ray_trafo = odl.as_proximal_lang_operator(ray_trafo)

# Create sinogram of forward projected phantom with noise
phantom = odl.phantom.shepp_logan(reco_space, modified=True)
phantom.show('phantom')
data = ray_trafo(phantom)
data += odl.phantom.white_noise(ray_trafo.range) * np.mean(data) * 0.1
data.show('noisy data')

# Convert to array for ProxImaL
rhs_arr = data.asarray()

# Set up optimization problem
# Note that proximal is not aware of the underlying space and only works with
# matrices. Hence the norm in proximal does not match the norm in the ODL space
# exactly.
x = proximal.Variable(reco_space.shape)
funcs = [proximal.sum_squares(proximal_lang_ray_trafo(x) - rhs_arr),
         0.2 * proximal.norm1(proximal.grad(x)),
         proximal.nonneg(x),
         proximal.nonneg(1 - x)]

# Solve the problem using ProxImaL
prob = proximal.Problem(funcs)
prob.solve(verbose=True)

# Convert back to odl and display result
result_odl = reco_space.element(x.value)
result_odl.show('ProxImaL result')
