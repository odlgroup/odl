"""Basic example on how to use the functional class together with solvers.

This file shows an example of how to set up and solve an optimization problem
using the default functionals. The problem we will solve is

    minimize 1/2 * ||x - g||_2^2 + lam*||x||_1,

for some vector g and some constant lam, subject to that all components in x
are greater than or equal to 0. The theoretical optimal solution to this
problem is

    x_opt = (g - lam)_+,

where ( )_+ denotes the positive part of the element, i.e.,
(z_i)_+ = max(z_i, 0).
"""

import numpy as np
import odl

# Create a space with dimensiona n=10.
n = 10
space = odl.rn(n)

# Create parameters.
offset = space.element([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
lam = 0.5

# Note that with the values above, the optimal solution is given by a vector
# with first half of the elements equal to 0.5, the second half equal to 0.


# The problem will be solved using the forward-backward primal-dual algorithm.
# In this setting we let f = nonnegativity contraint, g = l1-norm, L =
# the indentity operator, and h = the squared l2-norm.
f = odl.solvers.IndicatorNonnegativity(space)
g = lam * odl.solvers.L1Norm(space)
L = odl.IdentityOperator(space)
h = 1.0 / 2.0 * odl.solvers.L2NormSquared(space).translated(offset)

# Some solver parameters
niter = 50
tau = 0.5
sigma = 0.5

# Starting point, and also updated inplace in the solver
x = space.element(np.random.randn(n))
print('Initial guess: x = {}'.format(x))

# Optional: pass callback objects to solver
callback = odl.solvers.CallbackPrintIteration()

# Run the algorithm
odl.solvers.forward_backward_pd(x=x, f=f, g=[g],
                                L=[L], h=h, tau=tau, sigma=[sigma],
                                niter=niter, callback=callback)

print('Solution found: x = {}'.format(x))
