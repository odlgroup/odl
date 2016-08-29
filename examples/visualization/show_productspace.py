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

"""Example on using show with ProductSpace's."""

import odl
import numpy as np

n = 100
m = 7
spc = odl.uniform_discr([0, 0], [1, 1], [n, n])
pspace = odl.ProductSpace(spc, m)

# Making a product space element where each component consists of a
# Shepp-Logan phantom multiplied by the constant i, where i is the
# index of the product space component.
vec = pspace.element([odl.phantom.shepp_logan(spc, modified=True) * i
                      for i in range(m)])

# By default 4 uniformly spaced elements are shown. Since there are 7 in
# total, the shown components are 0, 2, 4 and 6
vec.show(title='Default')

# One can also use indexing by a list of indices or a slice.
vec.show(indices=[0, 1], show=True,
         title='Show first 2 elements')

vec.show(indices=np.s_[::3], show=True,
         title='Show every third element')

# Slices propagate (as in numpy): the first index in the slice applies to
# the product space components, the other dimensions are applied to each
# component. Here we take the second vector component and slice in the
# middle along the second axis.
vec.show(indices=np.s_[2, :, n // 2], show=True,
         title='Show second element, then slice by [:, n//2]')
