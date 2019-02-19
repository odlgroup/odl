"""Total variation tomography using PDHG.

Solves the optimization problem

    min_x  1/2 ||A(x) - g||_2^2 + lam || |grad(x)| ||_1

Where ``A`` is a parallel beam forward projector, ``grad`` the spatial
gradient and ``g`` is given noisy data.

For further details and a description of the solution method used, see
https://odlgroup.github.io/odl/guide/pdhg_guide.html in the ODL documentation.
"""

import numpy as np
import odl

# --- Set up the forward operator (ray transform) --- #

# Reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300], dtype='float32')

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, 360)
# Detector: uniformly sampled, n = 512, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 512)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Create the forward operator
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

# --- Generate artificial data --- #

# Create phantom
discr_phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# Create sinogram of forward projected phantom with noise
data = ray_trafo(discr_phantom)
data += odl.phantom.white_noise(ray_trafo.range) * np.mean(data) * 0.1

# --- Set up the inverse problem --- #

# Initialize gradient operator
gradient = odl.Gradient(reco_space)

# Column vector of two operators
op = odl.BroadcastOperator(ray_trafo, gradient)

# Do not use the f functional, set it to zero.
f = odl.solvers.ZeroFunctional(op.domain)

# Create functionals for the dual variable

# l2-squared data matching
l2_norm = odl.solvers.L2NormSquared(ray_trafo.range).translated(data)

# Isotropic TV-regularization i.e. the l1-norm
l1_norm = 0.015 * odl.solvers.L1Norm(gradient.range)

# Combine functionals, order must correspond to the operator K
g = odl.solvers.SeparableSum(l2_norm, l1_norm)

# --- Select solver parameters and solve using PDHG --- #

# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * odl.power_method_opnorm(op)

niter = 200  # Number of iterations
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable

# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration(step=10) &
            odl.solvers.CallbackShow(step=10))

# Choose a starting point
x = op.domain.zero()

# Run the algorithm
odl.solvers.pdhg(x, f, g, op, niter=niter, tau=tau, sigma=sigma,
                 callback=callback)

# Display images
discr_phantom.show(title='Phantom')
data.show(title='Simulated Data (Sinogram)')
x.show(title='TV Reconstruction', force_show=True)
