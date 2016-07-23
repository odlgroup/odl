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

"""Basic example on how to use the functional class together with solvers.

This file shows an example of how to set up and solve an optimization problem
using the default functionals. The problem we will solve is to minimize
1/2 * ||x - g||_2^2 + lam*||x||_1, for some vector g and some constant lam,
subject to that all components in x are greater than or equal to 0. The
theoretical optimal solution to this problem is x = (g - lam)_+, where ( )+
denotes the positive part of the element, i.e., (z_i)_+ = z_i if z_i >= 0, and
0 otherwise.
"""

import numpy as np
import odl

# Create a space with dimensiona n=10.
n = 10
space = odl.rn(n)

# Create parameters. First half of g are ones, second half are minus ones.
g = space.element(np.hstack((np.ones(n/2), -np.ones(n/2))))
lam = 0.5

# Note that with the values above, the optimal solution is given by a vector
# with fires half of the elements equal to 0.5, the second half equal to 0.

# Create the L1-norm functional and multiplyit with the constant lam.
lam_l1_func = lam * odl.solvers.L1Norm(space)

# Create the squared L2-norm and translate it with g.
l2_func = 1.0 / 2.0 * odl.solvers.L2NormSquare(space)
trans_l2_func = l2_func.translate(g)

# The problem will be solved using the forward-backward primal-dual algorithm.
# In this setting we let f = nonnegativity contraint, g = l1-norm, and h =
# the squared l2-norm. Here we create necessary proximal and gradient operators
# from the functionals.
prox_f = odl.solvers.proximal_nonnegativity(space)
prox_cc_g = lam_l1_func.conjugate_functional.proximal
L = odl.IdentityOperator(space)
grad_h = trans_l2_func.gradient

# Some solver parameters
niter = 50
tau = 0.5
sigma = 0.5

# Starting point, and also updated inplace in the solver
x_fbpd = space.zero()

# Optional: pass callback objects to solver
callback = (odl.solvers.CallbackPrintIteration())

# Run the algorithm
odl.solvers.forward_backward_pd(x=x_fbpd, prox_f=prox_f, prox_cc_g=[prox_cc_g],
                                L=[L], grad_h=grad_h, tau=tau, sigma=[sigma],
                                niter=niter, callback=callback)

print(x_fbpd.asarray())


# The problem can also be solved using, e.g., the Chambolle-Pock algorithm.
# Here we create the necessary proximal factories of the conjugate functionals
# (see the Chambolle-Pock algorithm and examples on this for more information).
prox_cc_l2 = trans_l2_func.conjugate_functional.proximal
prox_cc_l1 = lam_l1_func.conjugate_functional.proximal

# Combined the proximals for use in the solver
proximal_dual = odl.solvers.combine_proximals(prox_cc_l2, prox_cc_l1)

# Create the matrix of operators for the Chambolle-Pock solver
op = odl.BroadcastOperator(odl.IdentityOperator(space),
                           odl.IdentityOperator(space))

# Create the proximal operator for the constraint
proximal_primal = odl.solvers.proximal_nonnegativity(op.domain)

# The operator norm is sqrt(2), since only identity operators are used
op_norm = np.sqrt(2)

# Some solver parameters
niter = 50  # Number of iterations
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable

# Optional: pass callback objects to solver
callback = (odl.solvers.CallbackPrintIteration())

# Starting point, and also updated inplace in the solver
x_cp = op.domain.zero()

# Run the algorithm
odl.solvers.chambolle_pock_solver(op=op, x=x_cp, tau=tau, sigma=sigma,
                                  proximal_primal=proximal_primal,
                                  proximal_dual=proximal_dual, niter=niter,
                                  callback=callback)

print(x_cp.asarray())
