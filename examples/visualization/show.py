# Copyright 2014, 2015 The ODL development group
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

"""Examples on using the vector.show() syntax

NOTES
-----
The behaviour of blocking shows etc in matplotlib is experimental and can cause
issues with these examples.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# Internal
import matplotlib.pyplot as plt
import odl

spc = odl.uniform_discr([0, 0], [1, 1], [100, 100])
phantom = odl.util.shepp_logan(spc)

phantom.show()
phantom.ufunc.exp().show()

# Plotting is defered untill show() is called
plt.show()

# Can also force "instant" plotting
phantom.show(show=True)
