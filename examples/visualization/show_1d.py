# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

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
