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

"""Deconvolution using split Bregman solver."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library

standard_library.install_aliases()
from builtins import super

# External
import numpy as np
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt

# Internal
import odl
from odl.solvers import (l2_gradient, split_bregman_solver)


class Convolution(odl.Operator):
    def __init__(self, space, kernel, adjkernel):
        self.kernel = kernel
        self.adjkernel = adjkernel
        super().__init__(space, space, linear=True)

    def _call(self, rhs, out):
        scipy.ndimage.convolve(rhs,
                               self.kernel,
                               output=out.asarray(),
                               mode='wrap')
        return out

    @property
    def adjoint(self):
        return Convolution(self.domain, self.adjkernel, self.kernel)


def kernel(x):
    mean = [0.0, 0.5]
    std = [0.05, 0.05]
    return np.exp(-(((x[0] - mean[0]) / std[0]) ** 2 + ((x[1] - mean[1]) /
                                                        std[1]) ** 2))


def adjkernel(x):
    return kernel((-x[0], -x[1]))


# Continuous definition of problem
cont_space = odl.FunctionSpace(odl.Rectangle([-1, -1], [1, 1]))
kernel_space = odl.FunctionSpace(cont_space.domain - cont_space.domain)

# Discretization parameters
n = 50
npoints = np.array([n + 1, n + 1])
npoints_kernel = np.array([2 * n + 1, 2 * n + 1])

# Discretized spaces
discr_space = odl.uniform_discr_fromspace(cont_space, npoints)
discr_kernel_space = odl.uniform_discr_fromspace(kernel_space, npoints_kernel)

# Discretize the functions
disc_kernel = discr_kernel_space.element(kernel)
disc_adjkernel = discr_kernel_space.element(adjkernel)

# Load phantom
discr_phantom = odl.util.phantom.shepp_logan(discr_space, modified=True)

# Initialize convolution operator
conv = Convolution(discr_space, disc_kernel, disc_adjkernel)

# Run diagnostics to test the adjoint
# odl.diagnostics.OperatorTest(conv).run_tests()

# Initialize gradient operator
grad = odl.DiscreteGradient(discr_space, method='forward')

# Create data: convoluted image
rhs = conv(discr_phantom)

# Optionally pass partial to the solver to display intermediate results
partial = (odl.solvers.ShowPartial(title='intermediate results') &
           odl.solvers.PrintIterationPartial() &
           odl.solvers.PrintTimingPartial())

# Starting point
x = discr_space.zero()

# Run algorithms
la = 0.05  # Relaxation
mu = 50.  # Data fidelity
H_grad = l2_gradient(conv, rhs)
split_bregman_solver(mu * H_grad, grad, x, la, 1000, 1,
                     isotropic=True,
                     partial=partial)

# Display images
discr_phantom.show(title='original image')
rhs.show(title='convolved image')
x.show(title='deconvolved image')
plt.show()
