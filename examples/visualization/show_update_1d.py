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

"""Example on using show and updating the figure in real time in 1d."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
import odl
import matplotlib.pyplot as plt
import numpy as np

n = 100
m = 20
spc = odl.uniform_discr(0, 5, n)
vec = spc.element(np.sin(spc.points()))

# Pre-create a plot and set some property
fig = plt.figure()
plt.ylim(-m, m)

# Reuse the figure indefinitely
for i in range(m):
    fig = (vec * i).show(fig=fig)

plt.show()
