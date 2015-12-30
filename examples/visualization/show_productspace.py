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

""" Example on using show with ProductSpace's """

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
import odl
import numpy as np

n = 100
m = 7
spc = odl.uniform_discr([0, 0], [1, 1], [n, n])
pspace = odl.ProductSpace(spc, m)

vec = pspace.element([odl.util.shepp_logan(spc) * i for i in range(1, m+1)])

# By default 4 uniformly spaced elements are shown
vec.show(title='Default')

# User can also define a slice or by indexing
vec.show(indices=[0, 1], show=True,
         title='Show first 2 elements')

vec.show(indices=np.s_[::3], show=True,
         title='Show every third element')

# Slices propagate (as in numpy)
vec.show(indices=np.s_[2, :, n//2], show=True,
         title='Show second element, then slice by [:, n//2]')
