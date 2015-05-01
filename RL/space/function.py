# Copyright 2014, 2015 Holger Kohr, Jonas Adler
#
# This file is part of RL.
#
# RL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RL.  If not, see <http://www.gnu.org/licenses/>.


# Imports for common Python 2/3 codebase
from __future__ import unicode_literals, print_function, division
from __future__ import absolute_import
from future import standard_library

# RL imports
import RL.operator.functional as fun
from RL.space.space import HilbertSpace, Algebra
import RL.space.set as sets
from RL.utility.utility import errfmt

standard_library.install_aliases()


# Example of a space:
class FunctionSpace(Algebra):
    """ The space of scalar valued functions on some domain
    """

    def __init__(self, domain, field=None):
        if not isinstance(domain, sets.AbstractSet):
            raise TypeError("domain ({}) is not a set".format(domain))

        self.domain = domain
        self._field = field if field is not None else sets.RealNumbers()

    def linCombImpl(self, a, x, b, y):
        return a*x + b*y  # Use operator overloading

    def multiplyImpl(self, x, y):
        return self.makeVector(lambda *args: x(*args)*y(*args))

    @property
    def field(self):
        return self._field

    @property
    def dimension(self):
        raise NotImplementedError("TODO: infinite")

    def equals(self, other):
        return isinstance(other, FunctionSpace) and self.domain == other.domain

    def empty(self):
        return self.makeVector(lambda *args: 0)

    def zero(self):
        return self.makeVector(lambda *args: 0)

    def makeVector(self, *args, **kwargs):
        return FunctionSpace.Vector(self, *args, **kwargs)

    class Vector(HilbertSpace.Vector, Algebra.Vector, fun.Functional):
        """ L2 Vectors are functions from the domain
        """

        def __init__(self, space, function):
            HilbertSpace.Vector.__init__(self, space)
            self.function = function

        def applyImpl(self, rhs):
            return self.function(rhs)

        def assign(self, other):
            self.function = other.function

        @property
        def domain(self):
            return self.space.domain

        @property
        def range(self):
            return self.space.field


class L2(FunctionSpace, HilbertSpace):
    """The space of square integrable functions on some domain
    """

    def __init__(self, domain):
        FunctionSpace.__init__(self, domain)

    def innerImpl(self, v1, v2):
        raise NotImplementedError(errfmt('''
        You cannot calculate inner products in non-discretized spaces'''))

    def equals(self, other):
        return isinstance(other, L2) and FunctionSpace.equals(self, other)
