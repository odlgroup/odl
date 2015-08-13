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
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from builtins import super
from future import standard_library
standard_library.install_aliases()

# ODL imports
from odl.space.function import FunctionSpace
from odl.space.set import RealNumbers
from odl.space.space import HilbertSpace
from odl.utility.utility import errfmt


class L2(FunctionSpace, HilbertSpace):
    """The space of square integrable functions on some domain."""

    def __init__(self, domain, field=RealNumbers()):
        super().__init__(domain, field)

    def _inner(self, v1, v2):
        """ TODO: remove?
        """
        raise NotImplementedError(errfmt('''
        You cannot calculate inner products in non-discretized spaces'''))

    def equals(self, other):
        """ Verify that other is equal to this space as a FunctionSpace
        and also a L2 space.
        """
        return isinstance(other, L2) and FunctionSpace.equals(self, other)

    def __str__(self):
        if isinstance(self.field, RealNumbers):
            return "L2(" + str(self.domain) + ")"
        else:
            return "L2(" + str(self.domain) + ", " + str(self.field) + ")"

    def __repr__(self):
        if isinstance(self.field, RealNumbers):
            return "L2(" + repr(self.domain) + ")"
        else:
            return "L2(" + repr(self.domain) + ", " + repr(self.field) + ")"

    class Vector(FunctionSpace.Vector, HilbertSpace.Vector):
        """ A Vector in a L2-space

        FunctionSpace-Vectors are themselves also Functionals, and inherit
        a large set of features from them.

        Parameters
        ----------

        space : FunctionSpace
            Instance of FunctionSpace this vector lives in
        function : Function from space.domain to space.field
            The function that should be converted/reinterpreted as a vector.
        """
