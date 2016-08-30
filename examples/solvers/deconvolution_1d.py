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

"""Example of a deconvolution problem with different solvers (CPU)."""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import odl


class Convolution(odl.Operator):
    def __init__(self, kernel, adjkernel=None):
        self.kernel = kernel
        self.adjkernel = (adjkernel if adjkernel is not None
                          else kernel.space.element(kernel[::-1].copy()))
        self.norm = float(np.sum(np.abs(self.kernel.ntuple)))
        odl.Operator.__init__(self, domain=kernel.space, range=kernel.space,
                              linear=True)

    def _call(self, rhs, out):
        ndimage.convolve(rhs.ntuple.data, self.kernel.ntuple.data,
                         output=out.ntuple.data, mode='wrap')

    @property
    def adjoint(self):
        return Convolution(self.adjkernel, self.kernel)

    def opnorm(self):
        return self.norm


# Continuous definition of the problem
cont_space = odl.FunctionSpace(odl.Interval(0, 10))

# Complicated functions to check performance
cont_kernel = cont_space.element(lambda x: np.exp(x / 2) * np.cos(x * 1.172))
cont_phantom = cont_space.element(lambda x: x ** 2 * np.sin(x) ** 2 * (x > 5))

# Discretization
discr_space = odl.uniform_discr_fromspace(cont_space, 500, impl='numpy')
kernel = discr_space.element(cont_kernel)
phantom = discr_space.element(cont_phantom)

# Create operator
conv = Convolution(kernel)

# Dampening parameter for landweber
iterations = 100
omega = 1 / conv.opnorm() ** 2

# Display callback
callback = odl.solvers.CallbackApply(lambda result: plt.plot(conv(result)))

# Test CGN
plt.figure()
plt.plot(phantom)
odl.solvers.conjugate_gradient_normal(conv, discr_space.zero(), phantom,
                                      iterations, callback)

# Landweber
plt.figure()
plt.plot(phantom)
odl.solvers.landweber(conv, discr_space.zero(), phantom,
                      iterations, omega, callback)

plt.show()
