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

# pylint: disable=abstract-method

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

from odl.sets.space import LinearSpace
from odl.sets.domain import RealNumbers

"""An example of a very simple space, the real numbers."""


class Reals(LinearSpace):
    """The real numbers
    """

    def __init__(self):
        self._field = RealNumbers()

    def _inner(self, x, y):
        return x.__val__ * y.__val__

    def _lincomb(self, z, a, x, b, y):
        z.__val__ = a*x.__val__ + b*y.__val__

    def _multiply(self, z, x, y):
        z.__val__ = y.__val__ * x.__val__

    @property
    def field(self):
        return self._field

    def equals(self, other):
        return isinstance(other, Reals)

    def element(self, value=0):
        return Reals.Vector(self, value)

    class Vector(LinearSpace.Vector):
        """Real vectors are floats
        """

        __val__ = None
        def __init__(self, space, v):
            LinearSpace.Vector.__init__(self, space)
            self.__val__ = v

        def __float__(self):
            return self.__val__.__float__()

        def __str__(self):
            return str(self.__val__)

if __name__ == '__main__':
    R = Reals()
    x = R.element(5.0)
    y = R.element(10.0)

    print(x+y)
    print(x*y)
    print(x-y)
    print(x)
    print(y)
