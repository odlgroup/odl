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

"""Example of interfacing with scipy solvers.

This example solves the problem

    -Laplacian(x) = b

where b is a gaussian peak at the origin.
"""

import numpy as np
import scipy.sparse.linalg as sl
import odl

# Create discrete space, a square from [-1, 1] x [-1, 1] with (11 x 11) points
space = odl.uniform_discr([-1, -1], [1, 1], [11, 11])

# Create odl operator for negative laplacian
laplacian = -odl.Laplacian(space)

# Create right hand side, a gaussian around the point (0, 0)
rhs = space.element(lambda x: np.exp(-(x[0]**2 + x[1]**2) / 0.1**2))

# Convert laplacian to scipy operator
scipy_laplacian = odl.operator.oputils.as_scipy_operator(laplacian)

# Convert to array and flatten
rhs_arr = rhs.asarray().ravel(space.order)

# Solve using scipy
result, info = sl.cg(scipy_laplacian, rhs_arr)

# Other options include
# result, info = sl.cgs(scipy_laplacian, rhs_arr)
# result, info = sl.gmres(scipy_op, rhs_arr)
# result, info = sl.lgmres(scipy_op, rhs_arr)
# result, info = sl.bicg(scipy_op, rhs_arr)
# result, info = sl.bicgstab(scipy_op, rhs_arr)

# Convert back to odl and display result
result_odl = space.element(result)
result_odl.show('result')
(rhs - laplacian(result_odl)).show('residual')
