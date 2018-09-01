"""Create a convolution operator by wrapping a library."""

import odl
import scipy.signal


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
        super(Convolution, self).__init__(
            domain=kernel.space, range=kernel.space, linear=True)

    def _call(self, x):
        """Implement calling the operator by calling scipy."""
        return scipy.signal.fftconvolve(self.kernel, x, mode='same')

    @property
    def adjoint(self):
        """Implement ``self.adjoint``.

        For a convolution operator, the adjoint is given by the convolution
        with a kernel with flipped axes. In particular, if the kernel is
        symmetric the operator is self-adjoint.
        """
        return Convolution(self.kernel[::-1, ::-1])


# Define the space on which the problem should be solved
# Here the square [-1, 1] x [-1, 1] discretized on a 100x100 grid
space = odl.uniform_discr([-1, -1], [1, 1], [100, 100])

# Convolution kernel, a small centered rectangle
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
