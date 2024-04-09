"""Create a convolution operator by wrapping a library."""

import odl
import numpy as np
import torch


class Convolution(odl.Operator):
    """Operator calculating the convolution of a kernel with a function.

    The operator inherits from ``odl.Operator`` to be able to be used with ODL.
    """

    def __init__(self, kernel, domain, range):
        """Initialize a convolution operator with a known kernel."""

        # Store the kernel
        self.kernel = kernel

        # Initialize the Operator class by calling its __init__ method.
        # This sets properties such as domain and range and allows the other
        # operator convenience functions to work.
        super(Convolution, self).__init__(
            domain=domain, range=range, linear=True)

    def _call(self, x):
        """Implement calling the operator by calling PyTorch."""
        return self.range.element(torch.conv2d( input=x.data.unsqueeze(0)
                                              , weight=self.kernel.unsqueeze(0).unsqueeze(0)
                                              , stride=(1,1)
                                              , padding="same"
                                              ).squeeze(0)
                                 )

    @property
    def adjoint(self):
        """Implement ``self.adjoint``.

        For a convolution operator, the adjoint is given by the convolution
        with a kernel with flipped axes. In particular, if the kernel is
        symmetric the operator is self-adjoint.
        """
        return Convolution( torch.flip(self.kernel, dims=(0,1))
                          , domain=self.range, range=self.domain )


# Define the space on which the problem should be solved
# Here the square [-1, 1] x [-1, 1] discretized on a 100x100 grid
space = odl.uniform_discr([-1, -1], [1, 1], [100, 100], impl='pytorch', dtype=np.float32)

# Convolution kernel, a small centered rectangle
kernel = torch.ones((5,5))

# Create convolution operator
A = Convolution(kernel, domain=space, range=space)

# Create phantom (the "unknown" solution)
phantom = odl.phantom.shepp_logan(space, modified=True)

# Apply convolution to phantom to create data
g = A.adjoint(phantom)

# Display the results using the show method
phantom.show('phantom')
g.show('convolved phantom', force_show=True)
