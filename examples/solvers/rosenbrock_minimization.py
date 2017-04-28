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

"""Minimize the Rosenbrock functional.

This example shows how this can be done using a variety of solution methods.
"""

import odl

# Create the solution space
space = odl.rn(2)

# Create objective functional
f = odl.solvers.RosenbrockFunctional(space)

# Define a line search method
line_search = odl.solvers.BacktrackingLineSearch(f)


# Solve problem using steepest descent
callback = odl.solvers.CallbackShowConvergence(f, logx=True, logy=True,
                                               color='b')
x = space.zero()
odl.solvers.steepest_descent(f, x, line_search=line_search,
                             callback=callback)

# Solve problem using nonlinear conjugate gradient
callback = odl.solvers.CallbackShowConvergence(f, logx=True, logy=True,
                                               color='g')
x = space.zero()
odl.solvers.conjugate_gradient_nonlinear(f, x, line_search=line_search,
                                         callback=callback)

# Solve problem using bfgs
callback = odl.solvers.CallbackShowConvergence(f, logx=True, logy=True,
                                               color='r')
x = space.zero()
odl.solvers.bfgs_method(f, x, line_search=line_search,
                        callback=callback)

# Solve problem using newtons method
callback = odl.solvers.CallbackShowConvergence(f, logx=True, logy=True,
                                               color='k')
x = space.zero()
odl.solvers.newtons_method(f, x, line_search=line_search,
                           callback=callback)
