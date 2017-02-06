"""Total generalized variation tomography using the Chambolle-Pock solver.

Solves the optimization problem

    min_x ||A(x) - g||_2^2 + TGV_2(x)

Where ``A`` is a parallel beam forward projector and ``g`` is given noisy data.
TGV_2 is the second order total generalized variation of x, defined as

    TGV_2(x) = min_y lam_1 ||grad(x) - y||_1 + lam_2 ||eps(y)||_1

where ``grad`` is the (vectorial) spatial gradient and ``eps`` is the matrix
valued spatial second derivative. The problem is rewritten as

    min_{x, y} ||A(x) - g||_2^2 +  lam_1 ||grad(x) - y||_1 + lam_2 ||eps(y)||_1

which can then be solved with the chambolle pock method.

For further details and a description of the solution method used, see
:ref:`chambolle_pock` in the ODL documentation.
"""

import numpy as np
import odl


# --- Set up the forward operator (ray transform) --- #


# Discrete reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300], dtype='float32')

# Make a parallel beam geometry with flat detector
geometry = odl.tomo.parallel_beam_geometry(reco_space)

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

eps = odl.DiagonalOperator(gradient, 2)

# Column vector of three operators
domain = odl.ProductSpace(reco_space, gradient.range)
op = odl.BroadcastOperator(
    ray_trafo * odl.ComponentProjection(domain, 0),
    odl.ReductionOperator(gradient, odl.ScalingOperator(gradient.range, -1)),
    eps * odl.ComponentProjection(domain, 1))

# Do not use the g functional, set it to zero.
g = odl.solvers.ZeroFunctional(op.domain)

# Create functionals for the dual variable

# l2-squared data matching
l2_norm = odl.solvers.L2NormSquared(ray_trafo.range).translated(data)

# The l1-norms
l1_norm_1 = 0.015 * odl.solvers.L1Norm(gradient.range)
l1_norm_2 = 0.001 * odl.solvers.L1Norm(eps.range)

# Combine functionals, order must correspond to the operator K
f = odl.solvers.SeparableSum(l2_norm, l1_norm_1, l1_norm_2)


# --- Select solver parameters and solve using Chambolle-Pock --- #


# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * odl.power_method_opnorm(op)

niter = 400  # Number of iterations
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable
gamma = 0.5

# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow('iterates', indices=0))

# Choose a starting point
x = op.domain.zero()

# Run the algorithm
odl.solvers.chambolle_pock_solver(
    x, f, g, op, tau=tau, sigma=sigma, niter=niter, gamma=gamma,
    callback=callback)

# Display images
discr_phantom.show(title='Phantom')
data.show(title='Simulated data (Sinogram)')
x[0].show(title='TGV reconstruction')
x[1].show(title='Derivatives of reconstruction', force_show=True)
