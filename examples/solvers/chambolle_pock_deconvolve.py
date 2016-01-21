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

"""Deconvolution using Chambolle-Pock solver."""

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
from odl.solvers import chambolle_pock_solver, f_cc_prox_l2_tv, g_prox_none


class Convolution(odl.Operator):
    def __init__(self, space, kernel, adjkernel):
        self.kernel = kernel
        self.adjkernel = adjkernel

        # Scaling factor needed to account for non-unity cell volume
        # self.scale = disc_kernel.space.cell_volume

        super().__init__(space, space, linear=True)

    def _call(self, rhs, out):
        scipy.ndimage.convolve(rhs,
                               self.kernel,
                               output=out.asarray(),
                               mode='wrap')

        # out *= self.scale

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

# Run diagnosticts to test the adjoint
# odl.diagnostics.OperatorTest(conv).run_tests()

# Initialize gradient operator
grad = odl.DiscreteGradient(discr_space, method='forward')

# Matrix of operators
prod_op = odl.ProductSpaceOperator([[conv], [grad]])

# Operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
# prod_op_norm = 1.1 * odl.operator.oputils.power_method_opnorm(prod_op, 50)
# print(prod_op_norm)
prod_op_norm = 77

# Create data: convoluted image
g = discr_space.element()
conv(discr_phantom, out=g)

# Optionally pass partial to the solver to display intermediate results
fig = plt.figure('intermediate results')
partial = odl.solvers.util.ForEachPartial(
    lambda (result, iter): result.show(title='iteraton:{}'.format(iter),
                                       fig=fig))

# Run algorithms
rec = chambolle_pock_solver(prod_op,
                            f_cc_prox_l2_tv(prod_op.range, g, lam=0.0001),
                            g_prox_none(prod_op.domain),
                            sigma=1 / prod_op_norm,
                            tau=1 / prod_op_norm,
                            niter=400,
                            partial=None)[0]

# Display images
discr_phantom.show(title='original image')
g.show(title='convolved image')
rec.show(title='deconvolved image')
plt.show()
