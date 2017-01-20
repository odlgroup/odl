"""Total variation deconvolution using the Chambolle-Pock solver.

Solves the optimization problem

    min_x  1/2 ||A(x) - g||_2^2 + lam || |grad(x)| ||_1

Where ``A`` is a convolution operator, ``grad`` the spatial gradient and ``g``
is given noisy data.

For further details and a description of the solution method used, see
:ref:`chambolle_pock` in the ODL documentation.
"""

import numpy as np
import odl

# Discretization parameters
n = 128

# Discretized spaces
space = odl.uniform_discr([0, 0], [n, n], [n, n])

# Initialize convolution operator by Fourier formula
#     conv(f, g) = F^{-1}[F[f] * F[g]]
# Where F[.] is the Fourier transform and the fourier transform of a guassian
# with standard deviation filter_width is another gaussian with width
# 1 / filter_width
filter_width = 3.0  # standard deviation of the Gaussian filter
ft = odl.trafos.FourierTransform(space)
c = filter_width ** 2 / 4.0 ** 2
gaussian = ft.range.element(lambda x: np.exp(-(x[0] ** 2 + x[1] ** 2) * c))
convolution = ft.inverse * gaussian * ft

# Optional: Run diagnostics to assure the adjoint is properly implemented
# odl.diagnostics.OperatorTest(conv_op).run_tests()

# Create phantom
phantom = odl.phantom.shepp_logan(space, modified=True)

# Create the convolved version of the phantom
data = convolution(phantom)
data.show('Convolved data')

# Set up the Chambolle-Pock solver:

# Initialize gradient operator
gradient = odl.Gradient(space, method='forward')

# Column vector of two operators
op = odl.BroadcastOperator(convolution, gradient)

# Create the functional for unconstrained primal variable
g = odl.solvers.ZeroFunctional(op.domain)

# l2-squared data matching
l2_norm_squared = odl.solvers.L2NormSquared(space).translated(data)

# Isotropic TV-regularization i.e. the l1-norm
l1_norm = 0.0003 * odl.solvers.L1Norm(gradient.range)

# Make separable sum of functionals, order must be the same as in `op`
f = odl.solvers.SeparableSum(l2_norm_squared, l1_norm)


# --- Select solver parameters and solve using Chambolle-Pock --- #


# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * odl.power_method_opnorm(op)

niter = 300  # Number of iterations
tau = 10.0 / op_norm  # Step size for the primal variable
sigma = 0.1 / op_norm  # Step size for the dual variables

# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow(display_step=20))

# Choose a starting point
x = op.domain.zero()

# Run the algorithm
odl.solvers.chambolle_pock_solver(
    x, f, g, op, tau=tau, sigma=sigma, niter=niter, callback=callback)

# Display images
phantom.show(title='original image')
data.show(title='convolved image')
x.show(title='deconvolved image', force_show=True)
