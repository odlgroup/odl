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

"""NLM-TV tomography using the forward-backward primal dual solver.

Solves the optimization problem

    min_{0 <= x <= 1} ||A(x) - g||_2^2 + lam_1 TV(x) + lam_2 NLM(x)

where ``A`` is a ray transform, ``g`` the given noisy data ,
``TV`` total variation functional, and ``NLM`` is a Non-Local Means
regularizer. ``lam_1``, ``lam_2`` are regularization constants.

By using a combination of regularizers, a better result is achieved.
"""

import numpy as np
import odl


# Select what type of denoising to use. Options: 'TV', 'NLM' and 'TV_NLM'
model = 'TV_NLM'


# --- Set up the forward operator (ray transform) --- #


# Discrete reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 256 samples per dimension.
space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[256, 256], dtype='float32')

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, 1000)
# Detector: uniformly sampled, n = 558, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 1000)
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


# --- Generate artificial data --- #


# Create phantom
phantom = odl.phantom.forbild(space)
phantom.show('phantom', clim=[1, 1.1])

# Create sinogram of forward projected phantom with noise
data = ray_trafo(phantom)
data += odl.phantom.white_noise(ray_trafo.range) * np.mean(data) * 0.01


# --- Set up the inverse problem --- #

gradient = odl.Gradient(space)


# The implementation of Non-Local Means transform to use, options:
# 'skimage'                   Requires scikit-image (can be installed by
#                             running ``pip install scikit-image``).
# 'opencv'                    Require opencv (can be installed
#                             by running ``pip install opencv-python``).
impl = 'opencv'

# Create functionals for the regularizers and the bound constrains.
l1_norm = odl.solvers.GroupL1Norm(gradient.range)
nlm_func = odl.solvers.NLMRegularizer(space, h=0.02, impl=impl,
                                      patch_size=5, patch_distance=11)
f = odl.solvers.IndicatorBox(space, 0, 2)

# Assemble the linear operators. Here the TV-term is represented as a
# composition of the 1-norm and the gradient. See the documentation of the
# solver `forward_backward_pd` for the general form of the problem.
if model == 'TV':
    lin_ops = [gradient]
    g = [0.004 * l1_norm]
    sigma = [0.05]
elif model == 'NLM':
    lin_ops = [odl.IdentityOperator(space)]
    g = [nlm_func]
    sigma = [2.0]
elif model == 'TV_NLM':
    lin_ops = [gradient, odl.IdentityOperator(space)]
    g = [0.002 * l1_norm, nlm_func]
    sigma = [0.05, 2.0]
else:
    raise RuntimeError('Unknown model')

# This gradient encodes the differentiable term(s) of the goal functional,
# which corresponds to the "forward" part of the method. In this example the
# differentiable part is the squared 2-norm.
l2_norm = odl.solvers.L2NormSquared(ray_trafo.range)
h = l2_norm.translated(data) * ray_trafo

# Used to display intermediate results and print iteration number.
callback = (odl.solvers.CallbackShow(display_step=5, clim=[1, 1.1]) &
            odl.solvers.CallbackPrintIteration())

# Use FBP as initial guess
fbp_op = odl.tomo.fbp_op(ray_trafo,
                         filter_type='Hann', frequency_scaling=0.3)
fbp = fbp_op(data)
fbp.show('fbp', clim=[1, 1.1])

# Call the solver. x is updated in-place with the consecutive iterates.
x = fbp.copy()
odl.solvers.forward_backward_pd(x, f, g, lin_ops, h, tau=0.005,
                                sigma=sigma, niter=1000, callback=callback)

x.show('final result {}'.format(model), clim=[1, 1.1])
