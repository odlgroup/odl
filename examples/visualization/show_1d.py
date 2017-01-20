"""Example for `DiscreteLpElement.show` in 1D.

Notes
-----
The behaviour of blocking shows etc in matplotlib is experimental and can cause
issues with this example.
"""

import matplotlib.pyplot as plt
import odl
import numpy as np

space = odl.uniform_discr(0, 5, 100)
elem = space.element(np.sin)

# Get figure object
fig = elem.show(title='Sine functions')
# Plot into the same figure
fig = (elem / 2).show(fig=fig)

# Plotting is deferred until show() is called
plt.show()

# "Instant" plotting can be forced
elem.show(force_show=True)
