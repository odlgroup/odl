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

"""An example of a very simple space, the real numbers."""


import odl


class Reals(odl.LinearSpace):
    """The real numbers."""

    def __init__(self):
        odl.LinearSpace.__init__(self, field=odl.RealNumbers())

    def _inner(self, x1, x2):
        return x1.__val__ * x2.__val__

    def _lincomb(self, a, x1, b, x2, out):
        out.__val__ = a * x1.__val__ + b * x2.__val__

    def _multiply(self, x1, x2, out):
        out.__val__ = x1.__val__ * x2.__val__

    def __eq__(self, other):
        return isinstance(other, Reals)

    def element(self, value=0):
        return RealNumber(self, value)


class RealNumber(odl.LinearSpaceVector):
    """Real vectors are floats."""

    __val__ = None

    def __init__(self, space, v):
        odl.LinearSpaceVector.__init__(self, space=space)
        self.__val__ = v

    def __float__(self):
        return self.__val__.__float__()

    def __str__(self):
        return str(self.__val__)


R = Reals()
x = R.element(5.0)
y = R.element(10.0)

print(x)
print(y)
print(x + y)
print(x * y)
print(x - y)
print(3.14 * x)
