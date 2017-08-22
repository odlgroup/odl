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

# Matrix of operators
op = gradient

# Set up the functionals

# Isotropic TV-regularization: l1-norm of grad(x)
reg_param = 0.15
l1_norm = reg_param * odl.solvers.L1Norm(gradient.range)

# Make separable sum of functionals, order must correspond to the operator K
f = l1_norm

# Data fit with non-negativity constraint
g = odl.solvers.FunctionalQuadraticPerturb(
    odl.solvers.IndicatorNonnegativity(op.domain), 0.5, -noisy)

# --- Select solver parameters and solve using PDHG --- #

# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
# op_norm = 1.1 * odl.power_method_opnorm(op, xstart=noisy)
op_norm = np.sqrt(8) + 1e-4

niter = 200  # Number of iterations
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable

# Optional: pass callback objects to solver
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow(step=5))

# Starting point
x = op.domain.zero()

# Run algorithm (and display intermediates)
odl.solvers.primal_dual_hybrid_gradient_solver(
    x, f, g, op, tau=tau, sigma=sigma, niter=niter, gamma_primal=0.5,
    callback=callback)

# Display images
orig.show(title='original image')
noisy.show(title='noisy image')
x.show(title='reconstruction', force_show=True)
