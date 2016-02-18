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

"""Deconvolution using Chambolle-Pock solver.

Let X and Y be a finite-dimensional vector spaces equipped with an inner
product <.,.> and induced norm ||.||_2, and K a linear mapping from X to Y.

The primal minimization problem reads

    min_{x in X} F(K x) + G(x)

where F : Y -> [0, +infty] and G : X -> [0, +infty] are proper, convex,
possibly non-smooth functionals.

We consider unconstrained problem using l2-data fitting regularized by total
variation:

    F(K x) = 1/2 ||conv(x) - g||_2^2 + lam || |grad(x)| ||_1 ,

    G(x) = 0

where conv denotes the convolution operator, g the image to deconvolve,
||.||_1 denotes the l1-semi-norm, grad the spatial gradient, lam the
regularization parameter, and |.| is the point-wise magnitude across the
components of the vector grad(x). K denotes a column vector of operators K =
(conv, grad)^T.

First we define a convolution operator and create data by convolving
a Shepp-Logan phantom.

In order to use the Chambolle-Pock solver, we have to create the column
operator K, choose a starting point x, create the proximal operator
prox_tau[G] related to the primal variable x, create the proximal operator
prox_sigma[F_cc] related to the dual variable y, where F_cc denotes the
convex conjugate of F, and choose the step sizes tau and sigma such that
tau sigma ||K||_2^2 < 1.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library

standard_library.install_aliases()
from builtins import super

import numpy as np
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt

import odl


# TODO: Use BroadCastOperator instead of ProductSpaceOperator

# Define the forward operator of the inverse problem in question
class Convolution(odl.Operator):
    def __init__(self, space, kernel, adjkernel):
        self.kernel = kernel
        self.adjkernel = adjkernel
        super().__init__(space, space, linear=True)

    def _call(self, rhs, out):
        scipy.ndimage.convolve(rhs, self.kernel, output=out.asarray(),
                               mode='wrap')
        return out

    @property
    def adjoint(self):
        return Convolution(self.domain, self.adjkernel, self.kernel)

# Gaussian kernel for the convolution operator
def kernel(x):
    mean = [0.1, 0.5]
    std = [0.1, 0.1]
    k = np.exp(-(((x[0] - mean[0]) / std[0]) ** 2 +
                 ((x[1] - mean[1]) / std[1]) ** 2))
    return k

# Kernel for the adjoint of the convolution operator
def adjkernel(x):
    return kernel((-x[0], -x[1]))


# Continuous definition of problem
cont_space = odl.FunctionSpace(odl.Rectangle([-1, -1], [1, 1]))

# For the kernel create a space of twice the extent of
kernel_space = odl.FunctionSpace(cont_space.domain - cont_space.domain)
print(cont_space, kernel_space)
# Discretization parameters
n = 50
npoints = np.array([n + 1, n + 1])
npoints_kernel = np.array([2 * n + 1, 2 * n + 1])
print(npoints, npoints_kernel)
# Discretized spaces
discr_space = odl.uniform_discr_fromspace(cont_space, npoints)
discr_kernel_space = odl.uniform_discr_fromspace(kernel_space, npoints_kernel)
print(discr_space, discr_kernel_space)
print(discr_space.cell_sides, discr_kernel_space.cell_sides)
# Discretize the functions
disc_kernel = discr_kernel_space.element(kernel)
disc_adjkernel = discr_kernel_space.element(adjkernel)

# Initialize convolution operator
conv = Convolution(discr_space, disc_kernel, disc_adjkernel)

# Optional: Run diagnostics to assure the adjoint is properly implemented
# odl.diagnostics.OperatorTest(conv_op).run_tests()

# Create phantom
discr_phantom = odl.util.phantom.shepp_logan(discr_space, modified=True)
print(discr_phantom.shape)
# Create vector of convolved phantom
data = conv(discr_phantom)

# Solve the optimization problem

# Initialize gradient operator
grad = odl.Gradient(discr_space, method='forward')

# Column vector of two operators
op = odl.ProductSpaceOperator([[conv],
                               [grad]])

# Starting point
x = op.domain.one()

# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * odl.operator.oputils.power_method_opnorm(op, 100)
print('Norm of the product space operator: {}'.format(op_norm))

# Create proximal operators for unconstrained primal variable
proximal_primal = odl.solvers.proximal_zero(op.domain)

# Create proximal operators for the dual variable

# l2-data matching
prox_convconj_l2 = odl.solvers.proximal_convexconjugate_l2(discr_space, g=data)

# TV-regularization i.e. the l1-norm
prox_convconj_l1 = odl.solvers.proximal_convexconjugate_l1(
    grad.range, lam=0.01)

# Combine proximal operators, order must correspond to the operator K
proximal_dual = odl.solvers.combine_proximals(
    [prox_convconj_l2, prox_convconj_l1])

# Number of iterations
niter = 400

# Step size for the proximal operator for the primal variable x
tau = 1 / op_norm

# Step size for the proximal operator for the dual variable y
sigma= 1 / op_norm

# Optionally pass partial to the solver to display intermediate results
partial = (odl.solvers.util.PrintIterationPartial() &
           odl.solvers.util.PrintTimingPartial() &
           odl.solvers.util.ShowPartial())

# Run algorithms
odl.solvers.chambolle_pock_solver(
    op, x, tau=tau, sigma=sigma, proximal_primal=proximal_primal,
    proximal_dual=proximal_dual,  niter=niter, partial=partial)

# Display images
discr_phantom.show(title='original image')
data.show(title='convolved image')
x.show(title='deconvolved image')
plt.show()
