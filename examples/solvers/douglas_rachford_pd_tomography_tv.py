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

"""Total variation sparse angle tomography using the Douglas-Rachford solver.

Solves the optimization problem

    min_{0 <= x <= 1, Ax = g} lam ||grad(x)||_x

where ``A`` is ray transform operator, ``g`` the given noisy data, ``eps`` is
a small number, ``grad`` is the spatial gradient and || . ||_x is the so called
cross-norm, giving rise to isotropic Total Variation.

We do this by rewriting the problem on the form

    min_{0 <= x <= 1} lam ||grad(x)||_x + I_{Ax = g}

where I_{.} is the indicator function, which is zero if ``Ax = g`` and infinity
otherwise. This is a standard convex optimization problem that can
be solved with the `douglas_rachford_pd` solver.

In this example, the angles are highly under-sampled and the problem is solved
using only 6 angles. Dispite this, we get a perfect reconstruction.
A filtered backprojection (pseudoinverse) reconstruction is also shown at the
end for comparsion.

Note that the ``Ax = g`` condition could be relaxed to ``||Ax - g|| < eps`` for
some small eps in order to account for noise. This would be done using the
`IndicatorLpUnitBall` functional instead of `IndicatorZero`, as in this
example.
"""

import numpy as np
import odl

# Parameters
lam = 0.01

# --- Create spaces, forward operator and simulated data ---

# Discrete reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 128 samples per dimension.
space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[128, 128])

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 6, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, 6)
# Detector: uniformly sampled, n = 1000, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 1000)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Ray transform (= forward projection). We use ASTRA CUDA backend.
ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

# Create sinogram
phantom = odl.phantom.shepp_logan(space, modified=True)
data = ray_trafo(phantom)
phantom.show('Phantom')
data.show('Sinogram')

# --- Create functionals for solving the optimization problem ---

# Gradient for TV regularization
gradient = odl.Gradient(space)

# Functional to enforce 0 <= x <= 1
f = odl.solvers.IndicatorBox(space, 0, 1)

# Functional for I_{Ax = g}
indicator_data = odl.solvers.IndicatorZero(ray_trafo.range).translated(data)

# Functional for TV minimization
cross_norm = lam * odl.solvers.GroupL1Norm(gradient.range)

# --- Create functionals for solving the optimization problem ---

# Assemble operators and functionals for the solver
lin_ops = [ray_trafo, gradient]
g = [indicator_data, cross_norm]

# Create callback that prints iteration number and shows partial results
callback = (odl.solvers.CallbackShow(display_step=50, clim=[0, 1]) &
            odl.solvers.CallbackPrintIteration())

# Solve with initial guess x = 0.
# Step size parameters are selected to ensure convergece.
# See douglas_rachford_pd doc for more information.
x = ray_trafo.domain.zero()
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                tau=0.1, sigma=[0.1, 0.1], lam=1.5,
                                niter=1000, callback=callback)

# Compare with filtered backprojection
fbp_recon = odl.tomo.fbp_op(ray_trafo)(data)
fbp_recon.show('FBP reconstruction')
