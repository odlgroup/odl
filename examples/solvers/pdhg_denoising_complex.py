"""Total variation denoising of complex data using the PDHG solver.

Solves the optimization problem

    min_{x}  1/2 ||x - g||_2^2 + lam || |grad(x)| ||_1

Where ``grad`` the spatial gradient and ``g`` is given noisy data.

For further details and a description of the solution method used, see
https://odlgroup.github.io/odl/guide/pdhg_guide.html in the ODL documentation.
"""

import numpy as np
import scipy.misc
import odl

# Read test image: use only every second pixel, convert integer to float,
# and rotate to get the image upright
image = np.rot90(scipy.misc.ascent()[::1, ::1], 3).astype('float32')
image = image + 1j * image.T
shape = image.shape

# Rescale max to 1
image /= image.real.max()

# Discretized spaces
space = odl.uniform_discr([0, 0], shape, shape, dtype='complex64')

# Original image
orig = space.element(image)

# Add noise
noisy = image + 0.05 * odl.phantom.white_noise(orig.space)

# Gradient operator
gradient = odl.Gradient(space)

# Matrix of operators
op = odl.BroadcastOperator(odl.IdentityOperator(space), gradient)

# Set up the functionals

# l2-squared data matching
l2_norm = odl.solvers.L2NormSquared(space).translated(noisy)

# Isotropic TV-regularization: l1-norm of grad(x)
l1_norm = 0.15 * odl.solvers.L1Norm(gradient.range)

# Make separable sum of functionals, order must correspond to the operator K
f = odl.solvers.ZeroFunctional(op.domain)
g = odl.solvers.SeparableSum(l2_norm, l1_norm)

# --- Select solver parameters and solve using Chambolle-Pock --- #


# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * odl.power_method_opnorm(op, xstart=noisy)

niter = 200  # Number of iterations
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable

# Optional: pass callback objects to solver
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow(step=5))

# Starting point
x = op.domain.zero()

# Run algorithm (and display intermediates)
odl.solvers.pdhg(
    x, f, g, op, niter=niter, tau=tau, sigma=sigma, callback=callback)

# Display images
orig.show(title='original image')
noisy.show(title='noisy image')
x.show(title='reconstruction', force_show=True)
