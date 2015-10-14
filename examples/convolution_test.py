# Copyright 2014, 2015 The ODL development group
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

# pylint: disable=abstract-method

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

# External
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# ODL
import odl
import odl.operator.solvers as solvers
from odl.util.testutils import Timer
import solver_examples


class Convolution(odl.Operator):
    def __init__(self, kernel, adjkernel=None):
        if not isinstance(kernel.space, odl.DiscreteL2):
            raise TypeError("Kernel must be a DiscreteL2 vector")

        self.kernel = kernel
        self.adjkernel = (adjkernel if adjkernel is not None
                          else kernel.space.element(kernel[::-1].copy()))
        self.space = kernel.space
        self.norm = float(np.sum(np.abs(self.kernel.ntuple)))
        super().__init__(self.space, self.space, linear=True)

    def _apply(self, rhs, out):
        ndimage.convolve(rhs.ntuple.data, self.kernel.ntuple.data,
                         output=out.ntuple.data, mode='wrap')

    @property
    def adjoint(self):
        return Convolution(self.adjkernel, self.kernel)

    def opnorm(self):
        return self.norm


# Continuous definition of problem
cont_space = odl.L2(odl.Interval(0, 10))

# Complicated functions to check performance
cont_kernel = cont_space.element(lambda x: np.exp(x/2) * np.cos(x*1.172))
cont_data = cont_space.element(lambda x: x**2 * np.sin(x)**2*(x > 5))

# Discretization
discr_space = odl.l2_uniform_discretization(cont_space, 500, impl='numpy')
kernel = discr_space.element(cont_kernel)
data = discr_space.element(cont_data)

# Create operator
conv = Convolution(kernel)

# Dampening parameter for landweber
iterations = 10
omega = 1/conv.opnorm()**2

# Display partial
partial = solvers.ForEachPartial(lambda result: plt.plot(conv(result)[:]))

# Test CGN
plt.figure()
plt.plot(data)
solvers.conjugate_gradient_normal(conv, discr_space.zero(), data, iterations, partial)

# Landweber
plt.figure()
plt.plot(data)
solvers.landweber(conv, discr_space.zero(), data, iterations, omega, partial)

# testTimingCG
with Timer("Optimized CG"):
    solvers.conjugate_gradient_normal(conv, discr_space.zero(), data, iterations)

with Timer("Base CG"):
    solver_examples.conjugate_gradient_base(conv, discr_space.zero(), data,
                                            iterations)

# Landweber timing
with Timer("Optimized LW"):
    solvers.landweber(conv, discr_space.zero(), data, iterations, omega)

with Timer("Basic LW"):
    solver_examples.landweberBase(conv, discr_space.zero(), data, iterations,
                                  omega)

plt.show()
