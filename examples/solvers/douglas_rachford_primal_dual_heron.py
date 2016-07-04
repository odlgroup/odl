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

    min_{x \in R^2}  sum_i^n d(x, Omega_i)

Where d(x, \Omega_i) is the distance from x to the set Omega_i. Here, the
Omega_i are given by three rectangles.

This uses the infimal convolution option of the Douglas-Rachford solver since
the problem can be written:

    min_{x \in R^2}  sum_i^n inf_{z \in Omega_i} ||x - z||
"""

import numpy as np
import odl
import odl.solvers as odls

# Create the solution space
space = odl.rn(2)

# Define the rectangles by [minimum_corner, maximum_corner]
rectangles = [[[0, 0], [1, 1]],
              [[0, 2], [1, 3]],
              [[2, 2], [3, 3]]]

# The L operators are simply the identity in this case
L = [odl.IdentityOperator(space)] * len(rectangles)

# f is the zero function
prox_f = odls.proximal_zero(space)

# g is the distance function. Here, the l2 distance
prox_cc_g = [odls.proximal_cconj_l2(space)] * len(rectangles)

# l are the indicator functions on the rectangles. We want the proximal of the
# convex conjugate, so we need to convert to that.
prox_cc_l = [odls.proximal_cconj(odls.proximal_box_constraint(space,
                                                              minp, maxp))
             for minp, maxp in rectangles]

# Select step size
tau = 1.0 / len(rectangles)


def print_objective(x):
    """Calculates the objective value and prints it."""
    value = 0
    for minp, maxp in rectangles:
        x_inside = np.minimum(np.maximum(x, minp), maxp)
        value += (x - x_inside).norm()
    print('point = [{:.4f}, {:.4f}], value = {:.4f}'.format(x[0], x[1], value))

# Solve
x = space.zero()
odls.douglas_rachford_pd(x, prox_f, prox_cc_g, L,
                         tau=tau, sigma=[1.0] * len(rectangles), lam=1.0,
                         niter=20, callback=print_objective,
                         prox_cc_l=prox_cc_l)
