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

"""
Example of shape-based image reconstruction
using optimal information transformation.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
import numpy as np
import matplotlib.pyplot as plt
from odl.util import snr
from odl.deform import optimal_information_transport_solver
import odl
standard_library.install_aliases()


I0name = './pictures/c_highres.png'
I1name = './pictures/i_highres.png'

I0 = np.rot90(plt.imread(I0name).astype('float'), -1)[::2, ::2]
I1 = np.rot90(plt.imread(I1name).astype('float'), -1)[::2, ::2]

# Discrete reconstruction space: discretized functions on the rectangle
space = odl.uniform_discr(
    min_corner=[-16, -16], max_corner=[16, 16], nsamples=[128, 128],
    dtype='float32', interp='linear')

# Create the uniformly distributed directions
angle_partition = odl.uniform_partition(0, np.pi, 6)

# Create 2-D projection domain
# The length should be 1.5 times of that of the reconstruction space
detector_partition = odl.uniform_partition(-24, 24, 192)

# Create 2-D parallel projection geometry
geometry = odl.tomo.Parallel2dGeometry(angle_partition,
                                       detector_partition)

# ray transform aka forward projection. We use ASTRA CUDA backend.
ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

# Create the ground truth as the given image
ground_truth = space.element(I0)

# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo(ground_truth)

# Add white Gaussion noise onto the noiseless data
noise = odl.util.white_noise(ray_trafo.range) * 3

# Create the noisy projection data
noise_proj_data = proj_data + noise

# Compute the signal-to-noise ratio in dB
snr = snr(proj_data, noise, impl='dB')

# Output the signal-to-noise ratio
print('snr = {!r}'.format(snr))

# Maximum iteration number
niter = 2000

callback = odl.solvers.CallbackShow(
    'iterates', display_step=50) & odl.solvers.CallbackPrintIteration()

template = space.element(I1)
template *= np.sum(ground_truth) / np.sum(template)

ground_truth.show('phantom')
template.show('template')

# For image reconstruction
eps = 0.002
sigma = 1e2

# Create the forward operator for image reconstruction
op = ray_trafo

# Create the gradient operator for the L2 functional
gradS = op.adjoint * odl.ResidualOperator(op, noise_proj_data)

# Compute by optimal information transport solver
optimal_information_transport_solver(gradS, template, niter,
                                     eps, sigma, callback)

# # For image matching
# eps = 0.2
# sigma = 1
# # Create the forward operator for image matching
# op = odl.IdentityOperator(space)
#
# # Create the gradient operator for the L2 functional
# gradS = op.adjoint * odl.ResidualOperator(op, ground_truth)
#
# # Compute by optimal information transport solver
# optimal_information_transport_solver(gradS, template, niter,
#                                      eps, sigma, callback)
