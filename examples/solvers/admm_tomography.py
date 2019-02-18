"""Total variation tomography using linearized ADMM.

In this example we solve the optimization problem

    min_x  ||A(x) - y||_2^2 + lam * ||grad(x)||_1

Where ``A`` is a parallel beam ray transform, ``grad`` the spatial
gradient and ``y`` given noisy data.

The problem is rewritten in decoupled form as

    min_x g(L(x))

with a separable sum ``g`` of functionals and the stacked operator ``L``:

    g(z) = ||z_1 - g||_2^2 + lam * ||z_2||_1,

               ( A(x)    )
    z = L(x) = ( grad(x) ).

See the documentation of the `admm_linearized` solver for further details.
"""

import numpy as np
import odl

# --- Set up the forward operator (ray transform) --- #

# Reconstruction space: functions on the rectangle [-20, 20]^2
# discretized with 300 samples per dimension
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300], dtype='float32')

# Make a parallel beam geometry with flat detector, using 360 angles
geometry = odl.tomo.parallel_beam_geometry(reco_space, num_angles=180)

# Create the forward operator
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

# --- Generate artificial data --- #

# Create phantom and noisy projection data
phantom = odl.phantom.shepp_logan(reco_space, modified=True)
data = ray_trafo(phantom)
data += odl.phantom.white_noise(ray_trafo.range) * np.mean(data) * 0.1

# --- Set up the inverse problem --- #

# Gradient operator for the TV part
grad = odl.Gradient(reco_space)

# Stacking of the two operators
L = odl.BroadcastOperator(ray_trafo, grad)

# Data matching and regularization functionals
data_fit = odl.solvers.L2NormSquared(ray_trafo.range).translated(data)
reg_func = 0.015 * odl.solvers.L1Norm(grad.range)
g = odl.solvers.SeparableSum(data_fit, reg_func)

# We don't use the f functional, setting it to zero
f = odl.solvers.ZeroFunctional(L.domain)

# --- Select parameters and solve using ADMM --- #

# Estimated operator norm, add 10 percent for some safety margin
op_norm = 1.1 * odl.power_method_opnorm(L, maxiter=20)

niter = 200  # Number of iterations
sigma = 2.0  # Step size for g.proximal
tau = sigma / op_norm ** 2  # Step size for f.proximal

# Optionally pass a callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration(step=10) &
            odl.solvers.CallbackShow(step=10))

# Choose a starting point
x = L.domain.zero()

# Run the algorithm
odl.solvers.admm_linearized(x, f, g, L, tau, sigma, niter, callback=callback)

# Display images
phantom.show(title='Phantom')
data.show(title='Simulated Data (Sinogram)')
x.show(title='TV Reconstruction', force_show=True)
