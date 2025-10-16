"""Example for `DiscretizedSpaceElement.show` in 1D.

Notes
-----
The behaviour of blocking shows etc in matplotlib is experimental and can cause
issues with this example.
"""

import matplotlib.pyplot as plt
import numpy as np

import odl

space = odl.uniform_discr(0, 5, 100)
elem = space.element(lambda x : np.sin(x))

# Get figure object
fig = elem.show(title='Sine Functions')
# Plot into the same figure
fig = (elem / 2).show(fig=fig)

# Plotting is deferred until show() is called
plt.show()

# "Instant" plotting can be forced
elem.show(force_show=True)
