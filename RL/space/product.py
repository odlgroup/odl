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
try:
    from builtins import str, zip, range, super
except ImportError:  # Versions < 0.14 of python-future
    from future.builtins import str, zip, range, super
from future import standard_library

# 
from itertools import repeat

# RL imports
from RL.space.space import HilbertSpace, NormedSpace, LinearSpace
from RL.utility.utility import errfmt

standard_library.install_aliases()


class ProductSpace(HilbertSpace):
    """Product space (X1 x X2 x ... x Xn)
    """

    def __init__(self, *spaces):        
        """ Creates a Cartesian product of an arbitrary set of spaces.

        For example:

        `ProductSpace(Reals(), Reals())` is mathematically equivalent to `RN(2)`

        Note that the later is obviously more efficient.

        Args:
            spaces    (multiple) LinearSpace     One or more instances of linear spaces
        """
        if len(spaces) == 0:
            raise TypeError("Empty product not allowed")

        if not all(spaces[0].field == y.field for y in spaces):
            raise TypeError("All spaces must have the same field")

        self.spaces = spaces
        self._nProducts = len(self.spaces)
        self._field = spaces[0].field  # X_n has same field

    def zero(self):
        return self.makeVector(*[space.zero() for space in self.spaces])

    def empty(self):
        return self.makeVector(*[space.empty() for space in self.spaces])

    def innerImpl(self, x, y):
        return (sum(space.innerImpl(xp, yp)
                    for space, xp, yp in zip(self.spaces, x.parts, y.parts)))

    def linCombImpl(self, z, a, x, b, y):
        for space, zp, xp, yp in zip(self.spaces, z.parts, x.parts, y.parts):
            space.linCombImpl(zp, a, xp, b, yp)

    @property
    def field(self):
        return self._field

    def equals(self, other):
        return (isinstance(other, ProductSpace) and
                self._nProducts == other._nProducts and
                all(x.equals(y) for x, y in zip(self.spaces, other.spaces)))

    def makeVector(self, *args):
        return ProductSpace.Vector(self, *args)

    def __len__(self):
        """ Get the number of parts of this product space
        """
        return self._nProducts

    def __getitem__(self, index):
        """ Get the i:th part of this product space
        """
        return self.spaces[index]

    def __str__(self):
        return ('ProductSpace(' +
                ', '.join(str(space) for space in self.spaces) + ')')

    class Vector(HilbertSpace.Vector):
        def __init__(self, space, *args):
            super().__init__(space)

            if not isinstance(args[0], HilbertSpace.Vector):
                # Delegate constructors
                self.parts = (tuple(space.makeVector(arg)
                                    for arg, space in zip(args, space.spaces)))
            else:  # Construct from existing tuple
                if any(part.space != space
                       for part, space in zip(args, space.spaces)):
                    raise TypeError(errfmt('''
                    The spaces of all parts must correspond to this
                    space's parts'''))

                self.parts = args

        def __len__(self):
            return self.space._nProducts

        def __getitem__(self, index):
            return self.parts[index]

        def __str__(self):
            return (self.space.__str__() +
                    '::Vector(' + ', '.join(str(part) for part in self.parts) +
                    ')')

        def __repr__(self):
            return (self.space.__repr__() + '::Vector(' +
                    ', '.join(part.__repr__() for part in self.parts) + ')')


class PowerSpace(ProductSpace):
    """Product space with the same underlying space (X x X x ... x X)
    """

    def __init__(self, underlying_space, nProducts):
        """ Creates a Cartesian "power" of a space. For example,

        `PowerSpace(Reals(),10)` is mathematically the same as `RN(10)`

        Note that the later is obviously more efficient.

        Args:
            underlying_space    LinearSpace     An instance of some linear space
            nProducts           Integer         The number of times the underlying space should be replicated
        """
        super().__init__(*([underlying_space]*nProducts))