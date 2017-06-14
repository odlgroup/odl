"""Example for `DiscreteLpElement.show` in 2D.

Notes
-----
The behaviour of blocking shows etc in matplotlib is experimental and can cause
issues with this example.
"""

import odl

space = odl.uniform_discr([0, 0], [1, 1], [100, 100])
phantom = odl.phantom.shepp_logan(space, modified=True)

# Show all data
phantom.show(force_show=True)

# We can show subsets by index
phantom.show(indices=[None, 50])

# Or we can show by coordinate
phantom.show(coords=[None, 0.5])

# We can also show subsets
phantom.show(coords=[[None, 0.5], None])
