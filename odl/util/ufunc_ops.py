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

"""Ufunc operators for ODL vectors."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import re
from odl.space import ProductSpace
from odl.operator import Operator
from odl.util.ufuncs import UFUNCS


__all__ = ()


class UfuncOperator(Operator):

    """Base class for all ufunc operators."""


def ufunc_class_factory(name, nargin, nargout, docstring):
    """Create a UfuncOperator from a given specification."""

    def __init__(self, space):
        """Initialize an instance."""
        self.space = space

        if nargin == 1:
            domain = space
        else:
            domain = ProductSpace(space, nargin)

        if nargout == 1:
            range = space
        else:
            range = ProductSpace(space, nargin)

        UfuncOperator.__init__(self, domain=domain, range=range, linear=False)

    def _call(self, x, out):
        """return ``self(x)``."""
        return getattr(x.ufunc, name)(out=out)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(name, self.space)

    newclass = type(name, (UfuncOperator,),
                    {"__init__": __init__,
                     "_call": _call,
                     "__repr__": __repr__,
                     "__doc__": docstring})
    return newclass

# Create an operator for each ufunc
for name, nargin, nargout, docstring in UFUNCS:
    globals()[name] = ufunc_class_factory(name, nargin, nargout, docstring)
    __all__ += (name,)

if __name__ == '__main__':
    import odl
    r3 = odl.rn(3)

    s = sin(r3)

    print(s([-1, 1, 2]))