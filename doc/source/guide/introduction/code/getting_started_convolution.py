"""Source code for the getting started example."""

import odl
import scipy


class Convolution(odl.Operator):
    """Operator calculating the convolution of a kernel with a function.

    The operator inherits from ``odl.Operator`` to be able to be used with ODL.
    """

    def __init__(self, kernel):
        """Initialize a convolution operator with a known kernel."""

        # Store the kernel
        self.kernel = kernel

        # Initialize the Operator class by calling its __init__ method.
        # This sets properties such as domain and range and allows the other
        # operator convenience functions to work.
        odl.Operator.__init__(self, domain=kernel.space, range=kernel.space,
                              linear=True)

    def _call(self, x):
        """Implement calling the operator by calling scipy."""
        return scipy.signal.fftconvolve(self.kernel, x, mode='same')

    @property  # making adjoint a property lets users access it as A.adjoint
    def adjoint(self):
        return self  # the adjoint is the same as this operator


# Define the space the problem should be solved on.
# Here the square [-1, 1] x [-1, 1] discretized on a 100x100 grid.
space = odl.uniform_discr([-1, -1], [1, 1], [100, 100])

# Convolution kernel, a small centered rectangle.
kernel = odl.phantom.cuboid(space, [-0.05, -0.05], [0.05, 0.05])

# Create convolution operator
A = Convolution(kernel)

# Create phantom (the "unknown" solution)
phantom = odl.phantom.shepp_logan(space, modified=True)

# Apply convolution to phantom to create data
g = A(phantom)

# Display the results using the show method
kernel.show('kernel')
phantom.show('phantom')
g.show('convolved phantom')

# Landweber

# Need operator norm for step length (omega)
opnorm = odl.power_method_opnorm(A)

f = space.zero()
odl.solvers.landweber(A, f, g, niter=100, omega=1/opnorm**2)
f.show('landweber')

# Conjugate gradient

f = space.zero()
odl.solvers.conjugate_gradient_normal(A, f, g, niter=100)
f.show('conjugate gradient')

# Tikhonov with identity

B = odl.IdentityOperator(space)
a = 0.1
T = A.adjoint * A + a * B.adjoint * B
b = A.adjoint(g)

f = space.zero()
odl.solvers.conjugate_gradient(T, f, b, niter=100)
f.show('Tikhonov identity conjugate gradient')

# Tikhonov with gradient

B = odl.Gradient(space)
a = 0.0001
T = A.adjoint * A + a * B.adjoint * B
b = A.adjoint(g)

f = space.zero()
odl.solvers.conjugate_gradient(T, f, b, niter=100)
f.show('Tikhonov gradient conjugate gradient')

# Douglas-Rachford

# Assemble all operators into a list.
grad = odl.Gradient(space)
lin_ops = [A, grad]
a = 0.001

# Create functionals for the l2 distance and l1 norm.
g_funcs = [odl.solvers.L2NormSquared(space).translated(g),
           a * odl.solvers.L1Norm(grad.range)]

# Functional of the bound constraint 0 <= f <= 1
f = odl.solvers.IndicatorBox(space, 0, 1)

# Find scaling constants so that the solver converges.
# See the douglas_rachford_pd documentation for more information.
opnorm_A = odl.power_method_opnorm(A, xstart=g)
opnorm_grad = odl.power_method_opnorm(grad, xstart=g)
sigma = [1 / opnorm_A**2, 1 / opnorm_grad**2]
tau = 1.0

# Solve using the Douglas-Rachford Primal-Dual method
x = space.zero()
odl.solvers.douglas_rachford_pd(x, f, g_funcs, lin_ops,
                                tau=tau, sigma=sigma, niter=100)
x.show('TV Douglas-Rachford', show=True)
