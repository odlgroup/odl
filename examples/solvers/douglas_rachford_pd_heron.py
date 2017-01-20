"""Solves the generalized Heron problem using the Douglas-Rachford solver.

The generalized Heron problem is defined as

    min_{x in R^2}  sum_i d(x, Omega_i),

where d(x, Omega_i) is the distance from x to the set Omega_i. Here, the
Omega_i are given by three rectangles.

This uses the infimal convolution option of the Douglas-Rachford solver since
the problem can be written as:

    min_{x in R^2}  sum_i inf_{z \in Omega_i} ||x - z||.
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
f = odl.solvers.ZeroFunctional(space)

# g is the distance function `d(x, Omega_i)`. Here, the l2 distance.
g = [odl.solvers.L2Norm(space)] * len(rectangles)

# l are the indicator functions on the rectangles.
l = [odl.solvers.IndicatorBox(space, minp, maxp) for minp, maxp in rectangles]

# Select step size
tau = 1.0 / len(rectangles)
sigma = [1.0] * len(rectangles)


# The lam parameter can be used to accelerate the convergence rate
def lam(n):
    return 1.0 + 1.0 / (n + 1)


def print_objective(x):
    """Calculate the objective value and prints it."""
    value = 0
    for minp, maxp in rectangles:
        x_proj = np.minimum(np.maximum(x, minp), maxp)
        value += (x - x_proj).norm()
    print('point = [{:.4f}, {:.4f}], value = {:.4f}'.format(x[0], x[1], value))

# Solve
x = space.zero()
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                tau=tau, sigma=sigma, niter=20, lam=lam,
                                callback=print_objective, l=l)

# plot the result
for minp, maxp in rectangles:
    xp = [minp[0], maxp[0], maxp[0], minp[0], minp[0]]
    yp = [minp[1], minp[1], maxp[1], maxp[1], minp[1]]
    plt.plot(xp, yp)

plt.scatter(x[0], x[1])

plt.xlim(-1, 4)
plt.ylim(-1, 4)
plt.show()
