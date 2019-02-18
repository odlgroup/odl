"""Total Generalized Variation tomography using PDHG.

Solves the optimization problem

    min_x ||Ax - d||_2^2 + alpha * TGV_2(x)

Where ``A`` is a parallel beam forward projector and ``d`` is given noisy data.
TGV_2 is the second order total generalized variation of ``x``, defined as

    TGV_2(x) = min_y ||Gx - y||_1 + beta ||Ey||_1

where ``G`` is the spatial gradient operating on the scalar-field ``x`` and
``E`` is the symmetrized gradient operating on the vector-field ``y``.
Both one-norms ||.||_1 take the one-norm globally and the two-norm locally,
i.e. ||y||_1 := sum_i sqrt(sum_j y_i(j)^2) where y_i(j) is the j-th value
of the vector y_i at location i.

The problem is rewritten as

    min_{x, y} ||Ax - d||_2^2 + alpha ||Gx - y||_1 + alpha * beta ||Ey||_1

which can then be solved with PDHG.

References
----------
[1] K. Bredies and M. Holler. *A TGV-based framework for variational image
decompression, zooming and reconstruction. Part II: Numerics.* SIAM Journal
on Imaging Sciences, 8(4):2851-2886, 2015.
"""

import numpy as np
import odl

# --- Set up the forward operator (ray transform) --- #

# Reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 100 samples per dimension.
U = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[100, 100], dtype='float32')

# Make a parallel beam geometry with flat detector
geometry = odl.tomo.parallel_beam_geometry(U)

# Create the forward operator
A = odl.tomo.RayTransform(U, geometry)

# --- Generate artificial data --- #

# Create phantom
phantom = odl.phantom.tgv_phantom(U)
phantom.show(title='Phantom')

# Create sinogram of forward projected phantom with noise
data = A(phantom)
data += odl.phantom.white_noise(A.range) * np.mean(data) * 0.1

data.show(title='Simulated Data (Sinogram)')

# --- Set up the inverse problem --- #

# Initialize gradient operator
G = odl.Gradient(U, method='forward', pad_mode='symmetric')
V = G.range

Dx = odl.PartialDerivative(U, 0, method='backward', pad_mode='symmetric')
Dy = odl.PartialDerivative(U, 1, method='backward', pad_mode='symmetric')

# Create symmetrized operator and weighted space.
# TODO: As the weighted space is currently not supported in ODL we find a
# workaround.
# W = odl.ProductSpace(U, 3, weighting=[1, 1, 2])
# sym_gradient = odl.operator.ProductSpaceOperator(
#    [[Dx, 0], [0, Dy], [0.5*Dy, 0.5*Dx]], range=W)
E = odl.operator.ProductSpaceOperator(
    [[Dx, 0], [0, Dy], [0.5 * Dy, 0.5 * Dx], [0.5 * Dy, 0.5 * Dx]])
W = E.range

# Create the domain of the problem, given by the reconstruction space and the
# range of the gradient on the reconstruction space.
domain = odl.ProductSpace(U, V)

# Column vector of three operators defined as:
# 1. Computes ``Ax``
# 2. Computes ``Gx - y``
# 3. Computes ``Ey``
op = odl.BroadcastOperator(
    A * odl.ComponentProjection(domain, 0),
    odl.ReductionOperator(G, odl.ScalingOperator(V, -1)),
    E * odl.ComponentProjection(domain, 1))

# Do not use the f functional, set it to zero.
f = odl.solvers.ZeroFunctional(domain)

# l2-squared data matching
l2_norm = odl.solvers.L2NormSquared(A.range).translated(data)

# parameters
alpha = 4e-1
beta = 1

# The l1-norms scaled by regularization paramters
l1_norm_1 = alpha * odl.solvers.L1Norm(V)
l1_norm_2 = alpha * beta * odl.solvers.L1Norm(W)

# Combine functionals, order must correspond to the operator K
g = odl.solvers.SeparableSum(l2_norm, l1_norm_1, l1_norm_2)

# --- Select solver parameters and solve using PDHG --- #

# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * odl.power_method_opnorm(op)

niter = 300  # Number of iterations
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable

# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow(step=10, indices=0))

# Choose a starting point
x = op.domain.zero()

# Run the algorithm
odl.solvers.pdhg(x, f, g, op, niter=niter, tau=tau, sigma=sigma,
                 callback=callback)

# Display images
x[0].show(title='TGV Reconstruction')
x[1].show(title='Derivatives', force_show=True)
