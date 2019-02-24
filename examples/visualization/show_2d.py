"""Example for `DiscretizedSpaceElement.show` in 2D.

Notes
-----
The behaviour of blocking shows etc in matplotlib is experimental and can cause
issues with this example.
"""

import odl

space = odl.uniform_discr([0, 0], [1, 1], [100, 100])
phantom = odl.phantom.shepp_logan(space, modified=True)

# Show all data
space.show(phantom)

# We can show subsets by index
space.show(phantom, indices=[None, 50])

# Or we can show by coordinate
space.show(phantom, coords=[None, 0.5])

# We can also show subsets
space.show(phantom, coords=[[None, 0.5], None], force_show=True)
