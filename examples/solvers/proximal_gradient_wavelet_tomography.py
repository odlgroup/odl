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

    min_x  lam || T(W.inverse(x)) - g ||_2^2 + lam * || x ||_1

Where ``W`` is a wavelet operator, ``T`` is a parallel beam ray transform and
 ``g`` is given noisy data.

The proximal gradient solvers are also known as ISTA and FISTA.
"""

import odl
import numpy as np


# --- Set up problem definition --- #


# Discrete reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 256 samples per dimension.
space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[256, 256], dtype='float32')

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 300, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, 300)
# Detector: uniformly sampled, n = 300, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 300)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# The implementation of the ray transform to use, options:
# 'scikit'                    Requires scikit-image (can be installed by
#                             running ``pip install scikit-image``).
# 'astra_cpu', 'astra_cuda'   Requires astra tomography to be installed.
#                             Astra is much faster than scikit. Webpage:
#                             https://github.com/astra-toolbox/astra-toolbox
impl = 'astra_cuda'

# Create the forward operator, and also the vectorial forward operator.
ray_trafo = odl.tomo.RayTransform(space, geometry, impl=impl)


# --- Generate artificial data --- #


# Create phantom
discr_phantom = odl.phantom.shepp_logan(space, modified=True)

# Create sinogram of forward projected phantom with noise
data = ray_trafo(discr_phantom)
data += odl.phantom.white_noise(ray_trafo.range) * np.mean(data) * 0.1


# --- Set up the inverse problem --- #


# Create wavelet operator
W = odl.trafos.WaveletTransform(space, wavelet='haar', nlevels=5)

# The wavelets bases are normalized to constant norm regardless of scale.
# since we want to penalize "small" wavelets more than "large" ones, we need
# to weight by the scale of the wavelets.
# The "area" of the wavelets scales as 2 ^ scale, but we use a slightly smaller
# number in order to allow some high frequencies.
scales = W.scales()
Wtrafoinv = W.inverse * (1 / (np.power(1.7, scales)))

# Create regularizer as l1 norm
regularizer = 0.0002 * odl.solvers.L1Norm(W.range)

# l2-squared norm of residual
l2_norm_sq = odl.solvers.L2NormSquared(ray_trafo.range).translated(data)

# Compose from the right with ray transform and wavelet transform
data_discrepancy = l2_norm_sq * ray_trafo * Wtrafoinv

# --- Select solver parameters and solve using proximal gradient --- #

# Select step-size that guarantees convergence.
gamma = 0.05

# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow(display_step=10))


def callb(x):
    """Callback that displays the inverse wavelet transform of current iter."""
    callback(Wtrafoinv(x))

# Run the algorithm (FISTA)
x = data_discrepancy.domain.zero()
odl.solvers.accelerated_proximal_gradient(
    x, f=regularizer, g=data_discrepancy, niter=200, gamma=gamma,
    callback=callb)

# Display images
data.show(title='Data')
x.show(title='Wavelet coefficients')
Wtrafoinv(x).show('Wavelet regularized reconstruction')
