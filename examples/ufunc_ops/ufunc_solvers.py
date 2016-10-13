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

"""Examples of using the UFunc functionals in ODL in optimization.

Here, we minimize the logarithm of the rosenbrock function:

    min_x log(rosenbrock(x) + 0.1)
"""

from __future__ import print_function
import odl

# Create space and functionals
r2 = odl.rn(2)
rosenbrock = odl.solvers.example_funcs.RosenbrockFunctional(r2, scale=2.0)
log = odl.ufunc_ops.log()

# Create optimization function by composing log with rosenbrock, add 0.1 to
# avoid singularity at 0
opt_fun = log * (rosenbrock + 0.1)

# Select solution method, steepest descent with line search
line_search = odl.solvers.BacktrackingLineSearch(opt_fun)

x = opt_fun.domain.zero()
odl.solvers.steepest_descent(opt_fun.gradient, x, niter=100,
                             line_search=line_search)

print('optimization result={}. Should be [1, 1]'.format(x))
