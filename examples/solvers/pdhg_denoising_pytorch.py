"""Total variation denoising using PDHG.

Solves the optimization problem

    min_{x >= 0}  1/2 ||x - g||_2^2 + lam || |grad(x)| ||_1

Where ``grad`` the spatial gradient and ``g`` is given noisy data.

For further details and a description of the solution method used, see
https://odlgroup.github.io/odl/guide/pdhg_guide.html in the ODL documentation.
"""

import numpy as np
import torch
import scipy.misc
import odl
import cProfile

# Read test image: use only every second pixel, convert integer to float,
# and rotate to get the image upright
image = np.rot90(scipy.datasets.ascent()[::2, ::2], 3).astype('float')
shape = image.shape

# Rescale max to 1
image /= image.max()

# Discretized spaces
space = odl.uniform_discr([0, 0], shape, shape, impl='pytorch')

# Original image
orig = space.element(image.copy())

orig.data.requires_grad = False

# Add noise
noisy = space.element(image) + 0.1 * odl.phantom.white_noise(orig.space)

noisy.data.requires_grad = False

# Gradient operator
gradient = odl.Gradient(space)

# grad_xmp = gradient(orig)
# grad_xmp.show(title = "Grad-op applied to original")

# Matrix of operators
op = odl.BroadcastOperator(odl.IdentityOperator(space), gradient)

# Set up the functionals

# l2-squared data matching
l2_norm = odl.solvers.L2NormSquared(space).translated(noisy)

# Isotropic TV-regularization: l1-norm of grad(x)
l1_norm = 0.15 * odl.solvers.L1Norm(gradient.range)

# Make separable sum of functionals, order must correspond to the operator K
g = odl.solvers.SeparableSum(l2_norm, l1_norm)

# Non-negativity constraint
f = odl.solvers.IndicatorNonnegativity(op.domain)

# --- Select solver parameters and solve using PDHG --- #

# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * odl.power_method_opnorm(op, xstart=noisy, maxiter=10)
          # 3.2833764101732785
print(f"{op_norm=}")

niter = 200  # Number of iterations
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable

# Optional: pass callback objects to solver
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow(step=5))

# Starting point
x = op.domain.zero()

x.data.requires_grad = False

print("Go solve...")

# Run algorithm (and display intermediates)
def do_running():
  with torch.no_grad():
    odl.solvers.pdhg(x, f, g, op, niter=niter, tau=tau, sigma=sigma,
                 callback=callback)

do_running()
# cProfile.run('do_running()')

# Display images
orig.show(title='Original Image')
noisy.show(title='Noisy Image')
x.show(title='Reconstruction', force_show=True)
