"""Examples of using the ufunc functionals in ODL in optimization.

Here, we minimize the logarithm of the rosenbrock function:

    min_x log(rosenbrock(x) + 0.1)
"""

import odl

# Create space and functionals
r2 = odl.rn(2)
rosenbrock = odl.solvers.RosenbrockFunctional(r2, scale=2.0)
log = odl.ufunc_ops.log()

# Create goal functional by composing log with rosenbrock and add 0.1 to
# avoid singularity at 0
opt_fun = log * (rosenbrock + 0.1)

# Solve problem using steepest descent with backtracking line search,
# starting in the point x = [0, 0]
line_search = odl.solvers.BacktrackingLineSearch(opt_fun)

x = opt_fun.domain.zero()
odl.solvers.steepest_descent(opt_fun, x, maxiter=100,
                             line_search=line_search)

print('optimization result={}. Should be [1, 1]'.format(x))
