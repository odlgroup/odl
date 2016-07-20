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

"""Basic examples on how to use the fucntional class.

This file contains two basic examples on how to use the functional class. It
contains one example of how to use the default functionals, and how they can be
used in connection to solvers. It also contains one example on how to
implement a new functional.
"""

import numpy as np
import odl

# First we show how the default functionals can be used in order to set up and
# solve an optimization problem. The problem we will solve is to minimize
# 1/2 * ||x - g||_2^2 + lam*||x||_1, for some vector g and some constant lam,
# subject to that all components in x are greater than or equal to 0.

# Create a space with dimensiona n.
n = 10
space = odl.rn(n)

# Create parameters. First half of g are ones, second half are minus ones.
g = space.element(np.hstack((np.ones(n/2), -np.ones(n/2))))
lam = 0.5

# Create the L1-norm functional and multiplyit with the constant lam.
lam_l1_func = lam * odl.solvers.L1Norm(space)

# Create the squared L2-norm and translate it with g.
l2_func = 1.0 / 2.0 * odl.solvers.L2NormSquare(space)
trans_l2_func = l2_func.translate(g)

# The problem will be solved using the Chambolle-Pock algorithm. Here we create
# the necessary proximal factories of the conjugate functionals (see the
# Chambolle-Pock algorithm and examples on this for more information).
prox_cc_l2 = trans_l2_func.conjugate_functional.proximal
prox_cc_l1 = lam_l1_func.conjugate_functional.proximal

# Combined the proximals for use in the solver
proximal_dual = odl.solvers.combine_proximals(prox_cc_l2, prox_cc_l1)

# Create the matrix of operators for the Chambolle-Pock solver
op = odl.BroadcastOperator(odl.IdentityOperator(space),
                           odl.IdentityOperator(space))

# Create the proximal operator for the constraint
proximal_primal = odl.solvers.proximal_nonnegativity(op.domain)

# The operator norm is 1, since only identity operators are used
op_norm = 1

# Some solver parameters
niter = 100  # Number of iterations
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable

# Optional: pass callback objects to solver
callback = (odl.solvers.CallbackPrintIteration())

# Starting point, and also updated inplace in the solver
x = op.domain.zero()

# Run the algorithm
odl.solvers.chambolle_pock_solver(
    op, x, tau=tau, sigma=sigma, proximal_primal=proximal_primal,
    proximal_dual=proximal_dual, niter=niter, callback=callback)

# The theoretical optimal solution to this problem is x = (g - lam)_+, where
# ( )+ denotes the positive part of (i.e., (z)_+ = z if z >= 0, 0 otherwise).
print(x.asarray())
