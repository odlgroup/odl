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

"""Solves the generalized Heron problem using the Douglas-Rachford solver.

Solves the optimization problem

    min_{x in R^2}  sum_i d(x, Omega_i)

Where d(x, Omega_i) is the distance from x to the set Omega_i. Here, the
Omega_i are given by three rectangles.

This uses the infimal convolution option of the Douglas-Rachford solver since
the problem can be written as:

    min_{x in R^2}  sum_i inf_{z \in Omega_i} ||x - z||
"""

import matplotlib.pyplot as plt
import numpy as np
import odl

# Create the solution space
space = odl.rn(2)

# Define the rectangles by [minimum_corner, maximum_corner]
rectangles = [[[0, 0], [1, 1]],
              [[0, 2], [1, 3]],
              [[2, 2], [3, 3]]]

# The L operators are simply the identity in this case
lin_ops = [odl.IdentityOperator(space)] * len(rectangles)

# The function f in the douglas rachford solver is not needed so we set it
# to the zero function
prox_f = odl.solvers.proximal_zero(space)

# g is the distance function. Here, the l2 distance
prox_cc_g = [odl.solvers.proximal_cconj_l2(space)] * len(rectangles)

# l are the indicator functions on the rectangles.
prox_l = [odl.solvers.proximal_box_constraint(space, minp, maxp)
          for minp, maxp in rectangles]
# We want the proximal of the convex conjugate, so we need to convert to that.
prox_cc_l = [odl.solvers.proximal_cconj(prox) for prox in prox_l]

# Select step size
tau = 1.0 / len(rectangles)
sigma = [1.0] * len(rectangles)

# The lam parameter can be used to accelerate the convergence rate
lam = lambda n: 1.0 + 1.0 / (n + 1)


def print_objective(x):
    """Calculates the objective value and prints it."""
    value = 0
    for minp, maxp in rectangles:
        x_proj = np.minimum(np.maximum(x, minp), maxp)
        value += (x - x_proj).norm()
    print('point = [{:.4f}, {:.4f}], value = {:.4f}'.format(x[0], x[1], value))

# Solve
x = space.zero()
odl.solvers.douglas_rachford_pd(x, prox_f, prox_cc_g, lin_ops,
                                tau=tau, sigma=sigma, niter=20, lam=lam,
                                callback=print_objective, prox_cc_l=prox_cc_l)

# plot the result
for minp, maxp in rectangles:
    xp = [minp[0], maxp[0], maxp[0], minp[0], minp[0]]
    yp = [minp[1], minp[1], maxp[1], maxp[1], minp[1]]
    plt.plot(xp, yp)

plt.scatter(x[0], x[1])

plt.xlim(-1, 4)
plt.ylim(-1, 4)
plt.show()
