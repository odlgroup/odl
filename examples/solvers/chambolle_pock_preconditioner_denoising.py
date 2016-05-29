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

"""Total variation denoising using the Chambolle-Pock solver.

Solves the optimization problem

    min_{x >= 0}  1/2 ||x - g||_2^2 + lam || |grad(x)| ||_1

Where ``grad`` the spatial gradient and ``g`` is given noisy data.

For further details and a description of the solution method used, see
:ref:`chambolle_pock` in the ODL documentation.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import matplotlib.pyplot as plt
import odl


# Discretized spaces
space = odl.uniform_discr([0, 0], [50, 50], [256, 256])

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
                                                           lam=1/50.0)

# Combine proximal operators: the order must match the order of operators in K
proximal_dual = odl.solvers.combine_proximals([prox_convconj_l2,
                                               prox_convconj_l1])

# Proximal operator related to the primal variable

# Non-negativity constraint
proximal_primal = odl.solvers.proximal_nonnegativity(space)


# Set some general parameters

op_norm_identity = 1.0
op_norm_gradient = 1.0 * odl.power_method_opnorm(gradient, 100, noisy)
print('Operator norms, I: {}, gradient: {}'.format(op_norm_identity,
                                                   op_norm_gradient))

niter = 400

# --- Run algorithms without preconditioner


# Create a function to save the partial errors
partial = odl.solvers.StorePartial(function=lambda x: (x-orig).norm())

# Step sizes
op_norm = max(op_norm_identity, op_norm_gradient)
tau = sigma = 1.0 / op_norm

x = op.domain.zero()  # Starting point
odl.solvers.chambolle_pock_solver(
    op, x, tau=tau, sigma=sigma, proximal_primal=proximal_primal,
    proximal_dual=proximal_dual, niter=niter, partial=partial)


# --- Run algorithm with preconditoner

# Create a function to save the partial errors
partial_precon = odl.solvers.StorePartial(function=lambda x: (x-orig).norm())

# Create preconditioning operator. We precondition by scaling by the squared
# operator norms.
preconditioner_iden = odl.IdentityOperator(op[0].range)
preconditioner_grad = 1/op_norm_gradient**2 * odl.IdentityOperator(op[1].range)
preconditioner_dual = odl.DiagonalOperator(preconditioner_iden,
                                           preconditioner_grad)

# Step sizes, we need ||K * preconditioner_dual ** 0.5||_2^2 * sigma * tau < 1
tau = sigma = 0.8

x_precon = op.domain.zero()  # Starting point
odl.solvers.chambolle_pock_solver(
    op, x_precon, tau=tau, sigma=sigma, proximal_primal=proximal_primal,
    proximal_dual=proximal_dual, niter=niter, partial=partial_precon,
    precond_dual=preconditioner_dual)

# results
x.show('Standard')
x_precon.show('With preconditioner')

plt.figure()
plt.loglog(np.arange(niter), partial, label='Standard')
plt.loglog(np.arange(niter), partial_precon, label='With preconditioner')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Error norm')
plt.title('Convergence')
