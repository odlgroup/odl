# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Example of a deconvolution problem with different solvers (CUDA)."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

# External
import numpy as np
import matplotlib.pyplot as plt

# ODL
import odl
import odl.solvers as solvers
from odlpp import odlpp_cuda


class CudaConvolution(odl.Operator):

    """Calculates the circular convolution of two CUDA vectors."""

    def __init__(self, kernel, adjointkernel=None):
        self.space = kernel.space
        self.kernel = kernel
        self.adjkernel = (adjointkernel if adjointkernel is not None
                          else self.space.element(kernel[::-1].copy()))
        self.norm = float(np.sum(np.abs(self.kernel)))
        super().__init__(self.space, self.space, linear=True)

    def _call(self, rhs, out):
        odlpp_cuda.conv(rhs.ntuple.data,
                        self.kernel.ntuple.data,
                        out.ntuple.data)

    @property
    def adjoint(self):
        return CudaConvolution(self.adjkernel, self.kernel)

    def opnorm(self):  # An upper limit estimate of the operator norm
        return self.norm


# Continuous definition of problem
cont_space = odl.FunctionSpace(odl.Interval(0, 10))

# Complicated functions to check performance
cont_kernel = cont_space.element(lambda x: np.exp(x / 2) * np.cos(x * 1.172))
cont_data = cont_space.element(lambda x: x ** 2 * np.sin(x) ** 2 * (x > 5))

# Discretization
discr_space = odl.uniform_discr_fromspace(cont_space, 5000, impl='cuda')
kernel = discr_space.element(cont_kernel)
data = discr_space.element(cont_data)

# Create operator
conv = CudaConvolution(kernel)

# Dampening parameter for landweber
iterations = 100
omega = 1.0 / conv.opnorm() ** 2

# Display partial
partial = solvers.util.ForEachPartial(
    lambda result: plt.plot(conv(result).asarray()))

# Test CGN
plt.figure()
plt.plot(data)
solvers.conjugate_gradient_normal(conv, discr_space.zero(), data, iterations,
                                  partial)

# Landweber
plt.figure()
plt.plot(data)
solvers.landweber(conv, discr_space.zero(), data, iterations, omega, partial)


plt.show()
