"""Example for `DiscretizedSpaceElement.show` in 1D.

Notes
-----
The behaviour of blocking shows etc in matplotlib is experimental and can cause
issues with this example.
"""

import numpy as np

import odl

space = odl.uniform_discr(0, 5, 100)
elem = space.element(np.sin)

# Get figure object
fig = space.show(elem, title='Sine Functions')
# Plot into the same figure
fig = space.show(elem / 2, fig=fig)

# "Instant" plotting can be forced
space.show(elem, force_show=True)
