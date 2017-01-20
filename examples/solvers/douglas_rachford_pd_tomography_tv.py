"""Total variation sparse angle tomography using the Douglas-Rachford solver.

Solves the optimization problem

    min_{0 <= x <= 1, Ax = g} lam ||grad(x)||_x

where ``A`` is ray transform operator, ``g`` the given noisy data, ``grad`` is
the spatial gradient and || . ||_x is the so called cross-norm, giving rise to
isotropic Total Variation.

This problem can be rewritten in the form

    min_{0 <= x <= 1} lam ||grad(x)||_x + I_{Ax = g}

where I_{.} is the indicator function, which is zero if ``Ax = g`` and infinity
otherwise. This is a standard convex optimization problem that can
be solved with the `douglas_rachford_pd` solver.

In this example, the problem is solved with only 22 angles available, which is
highly under-sampled data. Despite this, we get a perfect reconstruction.
A filtered back-projection (pseudoinverse) reconstruction is also shown at the
end for comparsion.

This is an implementation of the "puzzling numerical experiment" in the seminal
paper "Robust Uncertainty Principles: Exact Signal Reconstruction from Highly
Incomplete Frequency Information", Candes et. al. 2006.

Note that the ``Ax = g`` condition can be relaxed to ``||Ax - g|| <= eps``
for some small ``eps`` in order to account for noise. This can be done using
the `IndicatorLpUnitBall` functional instead of `IndicatorZero`.
"""

import numpy as np
import odl

# Parameters
lam = 0.01
data_matching = 'exact'

# --- Create spaces, forward operator and simulated data ---

# Discrete reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 512 samples per dimension.
space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[512, 512])

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 22, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, 22)
# Detector: uniformly sampled, n = 512, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 512)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Ray transform (= forward projection).
ray_trafo = odl.tomo.RayTransform(space, geometry)

# Create sinogram
phantom = odl.phantom.shepp_logan(space, modified=True)
data = ray_trafo(phantom)

# --- Create functionals for solving the optimization problem ---

# Gradient for TV regularization
gradient = odl.Gradient(space)

# Functional to enforce 0 <= x <= 1
f = odl.solvers.IndicatorBox(space, 0, 1)

if data_matching == 'exact':
    # Functional to enforce Ax = g
    # Due to the splitting used in the douglas_rachford_pd solver, we only
    # create the functional for the indicator function on g here, the forward
    # model is handled separately.
    indicator_zero = odl.solvers.IndicatorZero(ray_trafo.range)
    indicator_data = indicator_zero.translated(data)
elif data_matching == 'inexact':
    # Functional to enforce ||Ax - g||_2 < eps
    # We do this by rewriting the condition on the form
    # f(x) = 0 if ||A(x/eps) - (g/eps)||_2 < 1, infinity otherwise
    # That function (with A handled separately, as mentioned above) is
    # implemented in ODL as the IndicatorLpUnitBall function.
    # Note that we use right multiplication in order to scale in input argument
    # instead of the result of the functional, as would be the case with left
    # multiplication.
    eps = 5.0

    # Add noise to data
    raw_noise = odl.phantom.white_noise(ray_trafo.range)
    data += raw_noise * eps / raw_noise.norm()

    # Create indicator
    indicator_l2_ball = odl.solvers.IndicatorLpUnitBall(ray_trafo.range, 2)
    indicator_data = indicator_l2_ball.translated(data / eps) * (1 / eps)
else:
    raise RuntimeError('unknown data_matching')

# Functional for TV minimization
cross_norm = lam * odl.solvers.GroupL1Norm(gradient.range)

# --- Create functionals for solving the optimization problem ---

# Assemble operators and functionals for the solver
lin_ops = [ray_trafo, gradient]
g = [indicator_data, cross_norm]

# Create callback that prints the iteration number and shows partial results
callback = (odl.solvers.CallbackShow('iterates',
                                     display_step=5, clim=[0, 1]) &
            odl.solvers.CallbackPrintIteration())

# Solve with initial guess x = 0.
# Step size parameters are selected to ensure convergence.
# See douglas_rachford_pd doc for more information.
x = ray_trafo.domain.zero()
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                tau=0.1, sigma=[0.1, 0.02], lam=1.5,
                                niter=200, callback=callback)

# Compare with filtered back-projection
fbp_recon = odl.tomo.fbp_op(ray_trafo)(data)
fbp_recon.show('FBP reconstruction')
phantom.show('Phantom')
data.show('Sinogram')
