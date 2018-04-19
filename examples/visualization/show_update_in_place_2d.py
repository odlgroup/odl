"""Example on using show and updating the figure in real time in 2d.

This example uses the update_in_place option, which can improve performance.
"""

import matplotlib.pyplot as plt

import odl

n = 100
m = 200
space = odl.uniform_discr([0, 0], [1, 1], [n, n])
phantom = odl.phantom.shepp_logan(space, modified=True)

# Create a figure by saving the result of show
fig = None

# Reuse the figure indefinitely, values are overwritten.
for i in range(m):
    fig = (phantom * i).show(fig=fig, clim=[0, m], update_in_place=True)

plt.show()
