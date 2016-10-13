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

"""Examples of using the trigonometric UFunc functionals in ODL."""

from __future__ import print_function
import odl

# Trigonometric functions can be computed, along with their gradient

cos = odl.ufunc_ops.cos()
sin = odl.ufunc_ops.sin()

print('cos(0)={}, cos.gradient(0.2)={}, -sin(0.2)={}'.format(
    cos(0), cos.gradient(0.2), -sin(0.2)))
