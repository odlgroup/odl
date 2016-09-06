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
phantom.show(show=True)

# We can show subsets by index
phantom.show(indices=[slice(None), 50])

# Or we can show by coordinate
phantom.show(coords=[None, 0.5])
