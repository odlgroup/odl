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

"""Total variation base image deconvolution using the Chambolle-Pock solver.

Let X and Y be finite-dimensional Hilbert spaces and K a linear mapping from
X to Y with induce norm ||K||. The (primal) minimization problem we want to
solve is

    min_{x in X} F(K x) + G(x)

where the proper, convex, lower-semicontinuous functionals
F : Y -> [0, +infinity] and G : X -> [0, +infinity] are given
by an l2-data fitting term regularized by isotropic total variation

    F(K x) = 1/2 ||conv(x) - g||_2^2 + lam || |grad(x)| ||_1

and

   G(x) = 0 ,

respectively. Here, conv denotes the convolution operator, g the image to
deconvolve, ||.||_2 the l2-norm, ||.||_1  the l1-semi-norm, grad the spatial
gradient, lam the regularization parameter, |.| the point-wise magnitude
across the vector components of grad(x), and K is a column vector of
operators K = (conv, grad)^T.

First we define a convolution operator and generate an image to be
deconvolved by convolving a Shepp-Logan phantom with a Gaussian kernel.

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
import odl

# Discrete reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_corner=[-20, -20], max_corner=[20, 20], nsamples=[300, 300],
    dtype='float32')

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
# Detector: uniformly sampled, n = 558, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 558)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# The implementation of the ray transform to use, options:
# 'astra_cpu', 'astra_cuda'.  Require astra tomography to be installed.
# 'scikit'.                   Requires scikit-image (can be installed by
#                             running ``pip install scikit-image``).
impl = 'scikit'

# ray transform aka forward projection.
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl=impl)

# Optional: Run diagnostics to assure the adjoint is properly implemented
# odl.diagnostics.OperatorTest(conv_op).run_tests()

# Create phantom
discr_phantom = odl.util.shepp_logan(reco_space, modified=True)

# Create vector of convolved phantom
data = ray_trafo(discr_phantom)
data += odl.util.white_noise(ray_trafo.range) * np.mean(data) * 0.1

# Set up the Chambolle-Pock solver:

# Initialize gradient operator
gradient = odl.Gradient(reco_space, method='forward')

# Column vector of two operators
op = odl.BroadcastOperator(ray_trafo, gradient)

# Choose a starting point
x = op.domain.one()

# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * odl.operator.oputils.power_method_opnorm(op, 5)
print('Norm of the product space operator: {}'.format(op_norm))

# Create the proximal operator for unconstrained primal variable
proximal_primal = odl.solvers.proximal_zero(op.domain)

# Create proximal operators for the dual variable

# l2-data matching
prox_convconj_l2 = odl.solvers.proximal_convexconjugate_l2(ray_trafo.range,
                                                           g=data)

# TV-regularization i.e. the l1-norm
prox_convconj_l1 = odl.solvers.proximal_convexconjugate_l1(
    gradient.range, lam=0.01)

# Combine proximal operators, order must correspond to the operator K
proximal_dual = odl.solvers.combine_proximals(
    [prox_convconj_l2, prox_convconj_l1])

# Number of iterations
niter = 400

# Step size for the proximal operator for the primal variable x
tau = 1 / op_norm

# Step size for the proximal operator for the dual variable y
sigma = 1 / op_norm

# Optionally pass partial to the solver to display intermediate results
partial = (odl.solvers.util.PrintIterationPartial() &
           odl.solvers.util.PrintTimingPartial() &
           odl.solvers.util.ShowPartial())

# Run the algorithm
odl.solvers.chambolle_pock_solver(
    op, x, tau=tau, sigma=sigma, proximal_primal=proximal_primal,
    proximal_dual=proximal_dual, niter=niter, partial=partial)

# Display images
discr_phantom.show(title='original image')
data.show(title='convolved image')
x.show(title='deconvolved image', show=True)
