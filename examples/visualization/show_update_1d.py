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

"""Example on using show and updating the figure in real time in 1d."""

import odl
import matplotlib.pyplot as plt
import numpy as np

n = 100
m = 20
space = odl.uniform_discr(0, 5, n)
elem = space.element(np.sin)

# Pre-create a plot and set some property, here the plot limits in the y axis.
fig = plt.figure()
plt.ylim(-m, m)

# Reuse the figure indefinitely
for i in range(m):
    fig = (elem * i).show(fig=fig)
    plt.pause(0.1)

plt.show()
