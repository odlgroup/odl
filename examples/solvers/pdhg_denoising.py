"""Total variation denoising using PDHG.

Solves the optimization problem

    min_{x >= 0}  1/2 ||x - g||_2^2 + lam || |grad(x)| ||_1

Where ``grad`` the spatial gradient and ``g`` is given noisy data.

For further details and a description of the solution method used, see
https://odlgroup.github.io/odl/guide/pdhg_guide.html in the ODL documentation.
"""

import numpy as np
import scipy.misc
import odl

# Read test image: use only every second pixel, convert integer to float,
# and rotate to get the image upright, and rescale to [0, 1]
image = np.rot90(scipy.misc.ascent()[::2, ::2], 3).astype('float')
shape = image.shape
image /= image.max()

# Reconstruction space with pixel size 1
space = odl.uniform_discr([0, 0], shape, shape)

# Noisy version of the image
noisy = image + 0.1 * odl.phantom.white_noise(space)

# Make operator x -> (x, grad(x))
grad = odl.Gradient(space)
L = odl.BroadcastOperator(odl.IdentityOperator(space), grad)

# --- Problem Definition --- #

# Squared L2 norm as data fit
data_fit = odl.solvers.L2NormSquared(space).translated(noisy)

# Anisotropic TV-regularization: L1 norm of grad(x)
regularizer = 0.15 * odl.solvers.L1Norm(grad.range)

# Separable sum of functionals, order corresponding to the operator L
g = odl.solvers.SeparableSum(data_fit, regularizer)

# Non-negativity constraint
f = odl.solvers.IndicatorNonnegativity(L.domain)

# --- Select solver parameters and solve using PDHG --- #

# Estimated operator norm, adding 10 percent safety margin
L_norm = 1.1 * odl.power_method_opnorm(L, xstart=noisy)

niter = 200  # Number of iterations
tau = 1.0 / L_norm  # Step size for the primal variable
sigma = 1.0 / L_norm  # Step size for the dual variable

# Optional: pass callback objects to solver
callback = (odl.solvers.CallbackPrintIteration(step=5) &
            odl.solvers.CallbackShow(space, step=5))

# Starting point
x = L.domain.zero()

# Run algorithm (and display intermediates)
odl.solvers.pdhg(
    x, f, g, L, niter=niter, tau=tau, sigma=sigma, callback=callback
)

# Display images
space.show(image, title='Original Image')
space.show(noisy, title='Noisy Image')
space.show(x, title='Reconstruction', force_show=True)
