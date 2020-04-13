"""Removal of salt-and-pepper noise using the proximal gradient solvers.

Solves the optimization problem

    min_x || x - g ||_1 + lam || grad(x) ||_2^2

Where ``grad`` is the spatial gradient operator and ``g`` is given noisy data.

The proximal gradient solvers are also known as ISTA and FISTA, respectively.
"""

import odl

# --- Set up problem definition --- #

# Reconstruction space: discretized functions on the rectangle [-20, 20]^2
# with 300 samples per dimension
space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300]
)

# Create noisy phantom
data = odl.phantom.salt_pepper_noise(
    space, odl.phantom.shepp_logan(space, modified=True)
)

# --- Set up the inverse problem --- #

# Create data fit term by translating the L1 norm
l1_norm = odl.solvers.L1Norm(space)
data_fit = l1_norm.translated(data)

# Use squared L2 norm of the gradient for regularization
grad = odl.Gradient(space)
regularizer = 0.05 * odl.solvers.L2NormSquared(grad.range) * grad

# --- Select solver parameters and solve using proximal gradient --- #

# Select step-size that guarantees convergence.
gamma = 0.01

# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration(step=5) &
            odl.solvers.CallbackShow(space, step=5))

# Run the algorithm (ISTA)
x = space.zero()
odl.solvers.proximal_gradient(
    x, f=data_fit, g=regularizer, niter=200, gamma=gamma, callback=callback
)

# Compare to accelerated version (FISTA) which converges much faster
callback.reset()
x_acc = space.zero()
odl.solvers.accelerated_proximal_gradient(
    x_acc, f=data_fit, g=regularizer, niter=50, gamma=gamma, callback=callback
)

# Display images
space.show(data, title='Noisy Image')
space.show(x, title='L1-denoised Image (ISTA)')
space.show(x_acc, title='L1-denoised Image (FISTA)', force_show=True)
