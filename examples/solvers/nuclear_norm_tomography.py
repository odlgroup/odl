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

"""Tomography with nuclear norm regularization.

Solves the optimization problem

    min_{0 <= x_1 <= 1, 0 <= x_2 <= 1}
        ||A(x_1) - g_1||_2^2 + 0.1 ||A(x_2) - g_2||_2^2 +
        lam || [grad(x_1), grad(x_2)] ||_*

where ``A`` is the ray transform,  ``grad`` is the spatial gradient,
``g_1``, ``g_2`` the given noisy data and ``|| . ||_*`` is the nuclear-norm.

The nuclear norm takes a vectorwise matrix norm, the spectral norm, i.e. the
p-norm of the singular values. It introduces a coupling between the terms
which allows better reconstructions by using the edge information from one term
in reconstructing the other term.

In this case we assume that ``g_2`` is much more noisy than ``g_1``, but we can
still get an acceptable reconstruction.

The data is assumed to be similar, but not exactly the same, for the channels.

Note that this is an advanced example.
"""


import numpy as np
import odl


# --- Set up the forward operator (ray transform) --- #


# Discrete reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 100 samples per dimension.
space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[100, 100], dtype='float32')

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
forward_op = odl.DiagonalOperator(ray_trafo, 2)

# Create phantom where the first component contains only part of the
# information in the  second component.
# We do this by using a sub-set of the ellipses in the well known Shepp-Logan
# phantom.
ellipses = odl.phantom.shepp_logan_ellipsoids(space.ndim, modified=True)
phantom = forward_op.domain.element(
    [odl.phantom.ellipsoid_phantom(space, ellipses[:2]),
     odl.phantom.ellipsoid_phantom(space, ellipses)])
phantom.show('phantom')

# Create data where second channel is highly noisy (SNR = 1)
data = forward_op(phantom)
data[1] += odl.phantom.white_noise(forward_op.range[1]) * np.mean(data[1])
data.show('data')

# Set up gradient and vectorial gradient
gradient = odl.Gradient(ray_trafo.domain)
pgradient = odl.DiagonalOperator(gradient, 2)

# Create data discrepancy functionals
l2err1 = odl.solvers.L2NormSquared(ray_trafo.range).translated(data[0])
l2err2 = odl.solvers.L2NormSquared(ray_trafo.range).translated(data[1])

# Scale the error term of the second channel so it is more heavily regularized.
# Note that we need to use SeparableSum, otherwise the solver would not be able
# to compute the proximal.
# The separable sum is defined by: l2err([x, y]) = l2err1(x) + 0.1 * l2err(y)
l2err = odl.solvers.SeparableSum(l2err1, 0.1 * l2err2)

# Create nuclear norm
nuc_norm = odl.solvers.NuclearNorm(pgradient.range,
                                   singular_vector_exp=1)

# Assemble the functionals and operators for the solver
lam = 0.1
lin_ops = [forward_op, pgradient]
g = [l2err, lam * nuc_norm]
f = odl.solvers.IndicatorBox(forward_op.domain, 0, 1)

# Create callback that prints current iterate value and displays every 20th
# iterate.
func = f + l2err * forward_op + lam * nuc_norm * pgradient
callback = (odl.solvers.CallbackShow() &
            odl.solvers.CallbackPrint(func=func))

# Solve the problem. Here the parameters are chosen in order to ensure
# convergence, see the documentation for further information.
x = forward_op.domain.zero()
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                tau=0.5, sigma=[0.01, 0.1],
                                niter=100, callback=callback)
