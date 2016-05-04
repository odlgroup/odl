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

"""Total variation base image denoising using the Chambolle-Pock solver.

Let X and Y be finite-dimensional Hilbert spaces and K a linear mapping
from X to Y with induce norm ||K||. The (primal) minimization problem we
want to solve is

    min_{x in X} F(K x) + G(x)

where the proper, convex, lower-semicontinuous functionals
F : Y -> [0, +infinity] and G : X -> [0, +infinity] are given
by an l2-data fitting term regularized by isotropic total variation

    F(K x) = 1/2 ||x - g||_2^2 + lam || |grad(x)| ||_1

and by the indicator function for the set fo non-negative components of x

   G(x) = {0 if x >=0, infinity if x < 0} ,

respectively. Here, g denotes the image to denoise, ||.||_2 the l2-norm,
||.||_1 the l1-semi-norm, grad the spatial gradient, lam the regularization
parameter, |.| the point-wise magnitude across the vector components of
grad(x), and K is a column vector of operators K = (id, grad)^T with
identity mapping id.

In order to use the Chambolle-Pock solver, we have to create the column
operator K, choose a starting point x, create the proximal operator for G,
create the proximal operator for the convex conjugate of F, choose the
step sizes tau and sigma such that tau sigma ||K||_2^2 < 1, and set the
total number of iterations.

For details see :ref:`chambolle_pock`, :ref:`proximal_operators`, and
references therein.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import matplotlib.pyplot as plt

import odl


# Discretized spaces
space = odl.uniform_discr([0, 0], [100, 100], [256, 256])

# Original image
orig = odl.util.shepp_logan(space, modified=True)
orig.show('Original image')

# Data of noisy image
noisy = orig + odl.util.white_noise(space) * 0.1
noisy.show('Nosy image')

# Gradient operator
gradient = odl.Gradient(space, method='forward')

# Matrix of operators
op = odl.BroadcastOperator(odl.IdentityOperator(space), gradient)

# Proximal operators related to the dual variable

# l2-data matching
prox_convconj_l2 = odl.solvers.proximal_convexconjugate_l2(space, g=noisy)

# TV-regularization: l1-semi norm of grad(x)
prox_convconj_l1 = odl.solvers.proximal_convexconjugate_l1(gradient.range,
                                                           lam=1/30.0)

# Combine proximal operators: the order must match the order of operators in K
proximal_dual = odl.solvers.combine_proximals([prox_convconj_l2,
                                               prox_convconj_l1])

# Proximal operator related to the primal variable

# Non-negativity constraint
proximal_primal = odl.solvers.proximal_nonnegativity(space)


# --- Run algorithms without preconditioner


op_norm_identity = 1.0
op_norm_gradient = 1.5 * odl.power_method_opnorm(gradient, 100, noisy)
print('Operator norms, I: {}, gradient: {}'.format(op_norm_identity,
                                                   op_norm_gradient))

niter = 100

# Create a function to save the partial errors
differences = [(noisy-orig).norm()]


def print_diff(x):
    error = (x-orig).norm()
    differences.append(error)

# Optional: pass partial objects to solver
partial = (odl.solvers.ShowPartial() &
           odl.solvers.PrintIterationPartial() &
           print_diff)

op_norm = op_norm_identity + op_norm_gradient
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable

x = op.domain.zero()  # Starting point
odl.solvers.chambolle_pock_solver(
    op, x, tau=tau, sigma=sigma, proximal_primal=proximal_primal,
    proximal_dual=proximal_dual, niter=niter, partial=partial)


# --- Run algorithm with preconditoner

# Create a function to save the partial errors
differences_precondtioned = [(noisy-orig).norm()]


def print_diff(x):
    error = (x-orig).norm()
    differences_precondtioned.append(error)

# Optional: pass partial objects to solver
partial = (odl.solvers.ShowPartial() &
           odl.solvers.PrintIterationPartial() &
           print_diff)

# Create preconditioning operator
preconditioner_iden = odl.IdentityOperator(op[0].range)
preconditioner_grad = odl.ScalingOperator(op[1].range, 1/op_norm_gradient**2)
preconditioner_dual = odl.DiagonalOperator(preconditioner_iden,
                                           preconditioner_grad)
# Step sizes, we need ||K * preconditioner_dual ** 0.5||_2^2 * sigma * tau < 1
tau = 1.0
sigma = 1.0

x = op.domain.zero()  # Starting point
odl.solvers.chambolle_pock_solver(
    op, x, tau=tau, sigma=sigma, proximal_primal=proximal_primal,
    proximal_dual=proximal_dual, niter=niter, partial=partial,
    preconditioner_dual=preconditioner_dual)

# results
plt.figure()
plt.loglog(np.arange(niter + 1), differences, label='standard')
plt.loglog(np.arange(niter + 1), differences_precondtioned,
           label='with preconditioner')
plt.legend()
