"""Example for `DiscreteLpElement.show` in 2D with complex data.

Notes
-----
The behaviour of blocking shows etc in matplotlib is experimental and can cause
issues with this example.
"""

import odl

space = odl.uniform_discr([0, 0], [1, 1], [100, 100], dtype='complex')
phantom = odl.phantom.shepp_logan(space, modified=True) * (1 + 0.5j)
phantom.show(force_show=True)
