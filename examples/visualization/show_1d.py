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

"""Examples on using the vector.show() syntax

NOTES
-----
The behaviour of blocking shows etc in matplotlib is experimental and can cause
issues with these examples.
"""

import matplotlib.pyplot as plt
import odl
import numpy as np

spc = odl.uniform_discr(0, 5, 100)
vec = spc.element(np.sin(spc.points()))

vec.show()
(vec * 2).show()

# Plotting is deferred until show() is called
plt.show()

# Can also force "instant" plotting
vec.show(show=True)
