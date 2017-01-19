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

The generalized Heron problem is defined as

    min_{x in R^2}  sum_i d(x, Omega_i),

where d(x, Omega_i) is the distance from x to the set Omega_i. Here, the
Omega_i are given by three rectangles.

This uses the infimal convolution option of the Douglas-Rachford solver since
the problem can be written as:

    min_{x in R^2}  sum_i inf_{z \in Omega_i} ||x - z||.
"""

import odl

# Create the solution space
space = odl.rn(2)

# Create objective functional
f = odl.solvers.RosenbrockFunctional(space)

# Define a line search method
line_search = odl.solvers.BacktrackingLineSearch(f)


# Create callback to show iterates
callback = odl.solvers.CallbackShowConvergence(f, logy=True)

# Solve problem
x = space.zero()
odl.solvers.conjugate_gradient_nonlinear(f, x, line_search=line_search,
                                         callback=callback)
