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

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from builtins import super
from future import standard_library
standard_library.install_aliases()

# ODL imports
from odl.space.fspace import FunctionSpace
from odl.set.sets import RealNumbers


__all__ = ('L2',)


class L2(FunctionSpace):
    """The space of square integrable functions on some domain."""

    def __init__(self, domain, field=RealNumbers()):
        super().__init__(domain, field)

    def _inner(self, v1, v2):
        """Inner product, not computable in continuous spaces."""
        raise NotImplementedError('inner product not computable in the'
                                  'non-discretized space {}.'.format(self))

    def __str__(self):
        if isinstance(self.field, RealNumbers):
            return 'L2({})'.format(self.domain)
        else:
            return 'L2({}, {})'.format(self.domain, self.field)

    def __repr__(self):
        if isinstance(self.field, RealNumbers):
            return 'L2({!r})'.format(self.domain)
        else:
            return 'L2({!r}, {!r})'.format(self.domain, self.field)

    class Vector(FunctionSpace.Vector):
        """Representation of an `L2` element."""

if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
