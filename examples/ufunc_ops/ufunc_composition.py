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

"""Examples of using the UFunc (Universal Function) functionals in ODL.

Here we demonstrate how the functionals can be composed with other operators
and functionals in order to achieve more complicated functions.

We create the L2-norm squared in two ways, first using the built in
L2NormSquared functional, and also by composing the square ufunc with the
L2Norm functional.
"""

import odl

# Create square functional. It's domain is by default the real numbers.
square = odl.ufunc_ops.square()

# Create L2 norm functionals
space = odl.rn(3)
l2_norm = odl.solvers.L2Norm(space)
l2_norm_squared_comp = square * odl.solvers.L2Norm(space)
l2_norm_squared_raw = odl.solvers.L2NormSquared(space)

# Evaluate in a point and see that the results are equal
x = [1, 2, 3]

print('composed      ||x||_1^2 = {}'.format(l2_norm_squared_comp(x)))
print('raw           ||x||_1^2 = {}'.format(l2_norm_squared_raw(x)))

# The usual properties like gradients follow as expected
print('composed grad ||x||_1^2 = {}'.format(l2_norm_squared_comp.gradient(x)))
print('raw      grad ||x||_1^2 = {}'.format(l2_norm_squared_raw.gradient(x)))
