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

from __future__ import print_function, division
from future import standard_library
standard_library.install_aliases()
from builtins import super

import odl

"""An example of a very simple operator on Rn."""


class AddOp(odl.Operator):
    def __init__(self, n, x):
        super().__init__(odl.Rn(n), odl.Rn(n))
        self.x = x

    def _apply(self, rhs, out):
        out[:] = rhs.data + self.x

n = 3
rn = odl.Rn(n)
x = rn.element([1, 2, 3])

op = AddOp(n, 10)

print(op(x))
