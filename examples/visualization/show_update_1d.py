"""Example on using show and updating the figure in real time in 1d."""

import odl
import matplotlib.pyplot as plt
import numpy as np

n = 100
m = 20
space = odl.uniform_discr(0, 5, n)
elem = space.element(np.sin)

# Pre-create a plot and set some property, here the plot limits in the y axis.
fig = plt.figure()
plt.ylim(-m, m)

# Reuse the figure indefinitely
for i in range(m):
    fig = (elem * i).show(fig=fig)
    plt.pause(0.1)

plt.show()
