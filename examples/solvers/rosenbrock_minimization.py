"""Minimize the Rosenbrock functional.

This example shows how this can be done using a variety of solution methods.
"""

import odl
from matplotlib import pyplot as plt

space = odl.rn(2)
f = odl.solvers.RosenbrockFunctional(space)
line_search = odl.solvers.BacktrackingLineSearch(f)

# --- Steepest Descent --- #

callback = odl.solvers.CallbackShowConvergence(
    f, logx=True, logy=True, color='b'
)
x = space.zero()
odl.solvers.steepest_descent(
    f, x, line_search=line_search, callback=callback
)
legend_artists = [callback.ax.collections[-1]]
legend_labels = ['SD']

# --- Nonlinear CG --- #

callback = odl.solvers.CallbackShowConvergence(
    f, logx=True, logy=True, color='g'
)
x = space.zero()
odl.solvers.conjugate_gradient_nonlinear(
    f, x, line_search=line_search, callback=callback
)
legend_artists.append(callback.ax.collections[-1])
legend_labels.append('CG')

# --- Broyden's Method --- #

callback = odl.solvers.CallbackShowConvergence(
    f, logx=True, logy=True, color='m'
)
x = space.zero()
odl.solvers.broydens_method(
    f, x, line_search=line_search, callback=callback
)
legend_artists.append(callback.ax.collections[-1])
legend_labels.append('Broyden')

# --- BFGS --- #

callback = odl.solvers.CallbackShowConvergence(
    f, logx=True, logy=True, color='r'
)
x = space.zero()
odl.solvers.bfgs_method(
    f, x, line_search=line_search, callback=callback
)
legend_artists.append(callback.ax.collections[-1])
legend_labels.append('BFGS')

# --- Newton's Method --- #

callback = odl.solvers.CallbackShowConvergence(
    f, logx=True, logy=True, color='k'
)
x = space.zero()
odl.solvers.newtons_method(
    f, x, line_search=line_search, callback=callback
)
legend_artists.append(callback.ax.collections[-1])
legend_labels.append('Newton')

# --- Add legend to plots and show it --- #

plt.legend(legend_artists, legend_labels)
plt.show()
