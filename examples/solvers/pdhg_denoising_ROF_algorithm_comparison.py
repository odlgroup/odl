"""Total variation denoising using the primal-dual hybrid gradient algorithm.

Solves the optimization problem

    min_{x >= 0}  1/2 ||x - g||_2^2 + lam || |grad(x)| ||_1

Where ``grad`` the spatial gradient and ``g`` is given noisy data.

For further details and a description of the solution method used, see
:ref:`chambolle_pock` in the ODL documentation.
"""

import numpy as np
import scipy
import odl

# --- define setting --- #

# Read test image: use only every second pixel, convert integer to float,
# and rotate to get the image upright
image = np.rot90(scipy.misc.ascent()[::2, ::2], 3).astype('float')
shape = image.shape

# Rescale max to 1
image /= image.max()

# Discretized spaces
space = odl.uniform_discr([0, 0], shape, shape)

# Original image
orig = space.element(image)

# Add noise
image += np.random.normal(0, 0.1, shape)

# Data of noisy image
noisy = space.element(image)

# Gradient operator
gradient = odl.Gradient(space, method='forward')

# l2-squared data matching
l2_norm = odl.solvers.L2NormSquared(space).translated(noisy)

# regularization parameter
reg_param = 0.15

# Isotropic TV-regularization: l1-norm of grad(x)
l1_norm = reg_param * odl.solvers.L1Norm(gradient.range)

# Optional: pass callback objects to solver
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow(step=5))

# define objective
obj = l2_norm + l1_norm * gradient

# number of iterations
niter = 400

# --- algorithm 1 --- #


# Matrix of operators
op = odl.BroadcastOperator(odl.IdentityOperator(space), gradient)

# Make separable sum of functionals, order must correspond to the operator K
f = odl.solvers.SeparableSum(l2_norm, l1_norm)

# Non-negativity constraint
g = odl.solvers.IndicatorNonnegativity(op.domain)

# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * odl.power_method_opnorm(op, xstart=noisy)

tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable

# Starting point
x_alg1 = op.domain.zero()

# Run algorithm (and display intermediates)
odl.solvers.primal_dual_hybrid_gradient_solver(
    x_alg1, f, g, op, tau=tau, sigma=sigma, niter=niter, callback=callback)


# --- algorithm 2 and 3 --- #


# Matrix of operators
op = gradient

# Make separable sum of functionals, order must correspond to the operator K
f = l1_norm

# Data fit with non-negativity constraint
g = odl.solvers.FunctionalQuadraticPerturb(
    odl.solvers.IndicatorNonnegativity(op.domain), 0.5, -noisy)

# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
# op_norm = 1.1 * odl.power_method_opnorm(op, xstart=noisy)
op_norm = np.sqrt(8) + 1e-4

tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable

# Starting point
x_alg2 = op.domain.zero()
x_alg3 = op.domain.zero()

# Run algorithm (and display intermediates)
odl.solvers.primal_dual_hybrid_gradient_solver(
    x_alg2, f, g, op, tau=tau, sigma=sigma, niter=niter, gamma_primal=0,
    callback=callback)
odl.solvers.primal_dual_hybrid_gradient_solver(
    x_alg3, f, g, op, tau=tau, sigma=sigma, niter=niter, gamma_primal=0.5,
    callback=callback)

# Display images
orig.show(title='original image')
noisy.show(title='noisy image')
x_alg1.show(title='alg 1, obj:{:.5e}'.format(obj(x_alg1)), force_show=True)
x_alg2.show(title='alg 2, obj:{:.5e}'.format(obj(x_alg2)), force_show=True)
x_alg3.show(title='alg 3, obj:{:.5e}'.format(obj(x_alg3)), force_show=True)
