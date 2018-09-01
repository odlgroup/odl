"""Example of interfacing with scipy solvers.

This example solves the problem

    -Laplacian(x) = b

where b is a gaussian peak at the origin.
"""

import numpy as np
import scipy.sparse.linalg as scipy_solvers
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
rhs_arr = rhs.asarray().ravel()

# Solve using scipy
result, info = scipy_solvers.cg(scipy_laplacian, rhs_arr)

# Other options include
# result, info = scipy_solvers.cgs(scipy_laplacian, rhs_arr)
# result, info = scipy_solvers.gmres(scipy_op, rhs_arr)
# result, info = scipy_solvers.lgmres(scipy_op, rhs_arr)
# result, info = scipy_solvers.bicg(scipy_op, rhs_arr)
# result, info = scipy_solvers.bicgstab(scipy_op, rhs_arr)

# Convert back to odl and display result
result_odl = space.element(result.reshape(space.shape))  # result is flat
result_odl.show('Result')
(rhs - laplacian(result_odl)).show('Residual', force_show=True)
