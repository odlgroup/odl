"""Denoising using the Chambolle-Pock solver with TV & entropy-type data term.

Solves the following optimization problem:

    min_{x > 0}  KL(x, g) + lam || |grad(x)| ||_1

where ``KL(x, g)`` is the Kullback-Leibler divergence, ``grad`` is the
spatial gradient, ``|| . ||_1`` is the 1 norm and lam is a regularization
constant.

For details see :ref:`chambolle_pock`, :ref:`proximal_operators`, and
references therein.
"""

import numpy as np
import scipy
import odl


# Read test image:
# convert integer values to float, and rotate to get the image upright
image = np.rot90(scipy.misc.ascent()[::2, ::2], 3).astype('float')
shape = image.shape

# Rescale
image *= 100 / image.max()

# Add noise
noisy_image = np.random.poisson(1 + image)

# Discretized spaces and elements
space = odl.uniform_discr([0, 0], shape, shape)
orig = space.element(image)
noisy = space.element(noisy_image)


# --- Set up the inverse problem --- #


# Gradient operator
gradient = odl.Gradient(space, method='forward')

# Matrix of operators
op = odl.BroadcastOperator(odl.IdentityOperator(space), gradient)


# Proximal operator related to the primal variable

# Non-negativity constraint
g = odl.solvers.IndicatorNonnegativity(op.domain)

# Functionals related to the dual variable

# Kulback-Leibler data matching
kl_divergence = odl.solvers.KullbackLeibler(space, prior=noisy)

# Isotropic TV-regularization: l1-norm of grad(x)
l1_norm = 0.1 * odl.solvers.L1Norm(gradient.range)

# Make separable sum of functionals, order must correspond to the operator K
f = odl.solvers.SeparableSum(kl_divergence, l1_norm)

# Optional: pass callback objects to solver
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow(step=5))


# --- Select solver parameters and solve using Chambolle-Pock --- #


# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * odl.power_method_opnorm(op)
tau = 10.0 / op_norm  # Step size for the primal variable
sigma = 0.1 / op_norm  # Step size for the dual variable

# Starting point
x = op.domain.zero()

# Run algorithm (and display intermediates)
odl.solvers.chambolle_pock_solver(
    x, f, g, op, tau=tau, sigma=sigma, niter=100, callback=callback)

# Display images
orig.show(title='original image')
noisy.show(title='noisy image')
x.show(title='denoised', force_show=True)  # show and hold
