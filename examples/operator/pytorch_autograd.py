"""Differentiation of functions implemented with ODL operators, through the
backpropagation functionality offered by PyTorch."""

import odl
import numpy as np
import torch


class Convolution(odl.Operator):
    """Operator calculating the convolution of a kernel with a function.

    See the convolution example for explanation.

    This operator is implemented directly in terms of PyTorch operations,
    and is therefore differentiable without further ado.
    """

    def __init__(self, kernel, domain, range):
        """Initialize a convolution operator with a known kernel."""

        self.kernel = kernel

        super(Convolution, self).__init__(
            domain=domain, range=range, linear=True)

    def _call(self, x):
        return self.range.element(
                  torch.conv2d( input=x.data.unsqueeze(0)
                              , weight=self.kernel.unsqueeze(0).unsqueeze(0)
                              , stride=(1,1)
                              , padding="same"
                              ).squeeze(0)
         )

    @property
    def adjoint(self):
        return Convolution( torch.flip(self.kernel, dims=(0,1))
                          , domain=self.range, range=self.domain )

class PointwiseSquare_PyTorch(odl.Operator):
    def __init__(self, domain):
        super().__init__(domain=domain, range=domain, linear=False)

    def _call(self, x):
        return x*x


# Define the space on which the problem should be solved
# Here the square [-1, 1] x [-1, 1] discretized on a 100x100 grid
phantom_space = odl.uniform_discr([-1, -1], [1, 1], [100, 100], impl='pytorch', dtype=np.float32)
space = odl.PytorchTensorSpace([100,100], dtype=np.float32)

# Convolution kernel, a Sobel-like edge detector in y direction
kernel = torch.tensor([[-1, 0, 1]
                      ,[-1, 0, 1]
                      ,[-1, 0, 1]], dtype=torch.float32)

# Create composed operator
A = ( PointwiseSquare_PyTorch(domain=space)
     * Convolution(kernel, domain=space, range=space)
    )

# Create phantom, as example input
phantom = odl.phantom.shepp_logan(phantom_space, modified=True)

torch_input = phantom.data.detach().clone()

torch_input.requires_grad = True
odl_input = space.element_type(space, data=torch_input)

# Apply convolution to phantom to create data
g = A(odl_input)
grad = space.element(torch.autograd.grad(torch.sum(g.data), torch_input)[0])

# Alternative version in raw PyTorch
# g_torch = torch.conv2d( input=torch_input.unsqueeze(0)
#                       , weight=kernel.unsqueeze(0).unsqueeze(0)
#                       , padding="same"
#                       ).squeeze(0) ** 2

# grad = space.element(torch.autograd.grad(torch.sum(g_torch), torch_input)[0])

def display(x, label, **kwargs):
    phantom_space.element(x.data).show(label, **kwargs)

# Display the results using the show method
display(odl_input, 'phantom')
display(g, 'convolved phantom')
display(grad, 'autograd', force_show=True)

