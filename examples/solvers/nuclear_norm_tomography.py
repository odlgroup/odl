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

"""Tomography with Nuclear norm regularization .

Solves the optimization problem

    min_{0 <= x_1 <= 1, 0 <= y_2 <= 1}
        ||Ax_1 - g_1||_2^2 + ||Ax_2 - g_2||_2^2 +
        lam || [grad(x_1), grad(x_2)] ||_*

Where ``A`` is the ray transform,  ``grad`` is the spatial gradient,
``g_1``, ``g_2`` the given noisy data and ``|| . ||_*`` is the nuclear-norm.

The nuclear norm takes a vectorwise matrix norm, the spectral norm, i.e. the
p-norm of the singular vectors. It introduces a coupling between the terms
which allows better reconstructions.

In this case we assume that ``g_2`` is much more noisy than ``g_1``, but we can
still get an acceptable reconstruction.

The data is assumed to be similar, but not exactly the same, for the channels.
"""


import numpy as np
import odl


# --- Set up the forward operator (ray transform) --- #


# Discrete reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[100, 100], dtype='float32')

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
ray_trafo = odl.tomo.RayTransform(space, geometry, impl=impl)
forward_op = odl.DiagonalOperator(ray_trafo, ray_trafo)

ellipses = odl.phantom.shepp_logan_ellipses(space.ndim, modified=True)

phantom = forward_op.domain.element(
    [odl.phantom.ellipse_phantom(space, ellipses[:2]),
     odl.phantom.ellipse_phantom(space, ellipses)])
phantom.show('phantom')

rhs = forward_op(phantom)
rhs[1] += odl.phantom.white_noise(forward_op.range[1]) * np.mean(rhs[1]) * 1.0
rhs.show('rhs')

# Set up gradient
gradient = odl.Gradient(ray_trafo.domain)
pgradient = odl.DiagonalOperator(gradient, gradient)

# Assemble all operators
lin_ops = [forward_op, pgradient]

# Create functionals as needed
l2err = odl.solvers.L2Norm(forward_op.range).translated(rhs)
nuc_norm = odl.solvers.NuclearNorm(pgradient.range,
                                   singular_vector_exp=2,
                                   weighting=[1, 0.1])

const = 0.001

g = [l2err, const * nuc_norm]
f = odl.solvers.IndicatorBox(forward_op.domain, 0, 1)

# Solve
x = forward_op.domain.zero()
callback = (odl.solvers.CallbackShow(display_step=10) &
            odl.solvers.CallbackPrintIteration())
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                tau=0.5, sigma=[0.01, 0.1],
                                niter=300, callback=callback)
