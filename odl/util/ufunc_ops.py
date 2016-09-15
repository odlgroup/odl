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
from odl.space import ProductSpace, fn
from odl.operator import Operator
from odl.util.ufuncs import UFUNCS

__all__ = ()


class UfuncOperator(Operator):

    """Base class for all ufunc operators."""


RAW_EXAMPLES_DOCSTRING = """
Examples
--------
>>> space = {space}
>>> op = {name}(r3)
>>> op({arg})
{result}
"""


def ufunc_class_factory(name, nargin, nargout, docstring):
    """Create a UfuncOperator from a given specification."""

    assert 0 <= nargin <= 2

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
        if nargin == 1:
            return getattr(x.ufunc, name)(out=out)
        elif nargin == 2:
            return getattr(x[0].ufunc, name)(*x[1:], out=out)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(name, self.space)

    # Create example
    if 'shift' in name or 'bitwise' in name or name == 'invert':
        dtype = int
    else:
        dtype = float

    space = fn(3, dtype=dtype)
    if nargin == 1:
        vec = space.element([-1, 1, 2])
        arg = '{}'.format(vec)
        with np.errstate(all='ignore'):
            result = getattr(vec.ufunc, name)()
    else:
        vec = space.element([-1, 1, 2])
        vec2 = space.element([3, 4, 5])
        arg = '[{}, {}]'.format(vec, vec2)
        with np.errstate(all='ignore'):
            result = getattr(vec.ufunc, name)(vec2)

    examples_docstring = RAW_EXAMPLES_DOCSTRING.format(space=space, name=name,
                                                       arg=arg, result=result)
    full_docstring = docstring + examples_docstring

    newclass = type(name, (UfuncOperator,),
                    {"__init__": __init__,
                     "_call": _call,
                     "__repr__": __repr__,
                     "__doc__": full_docstring})

    return newclass

# Create an operator for each ufunc
for name, nargin, nargout, docstring in UFUNCS:
    globals()[name] = ufunc_class_factory(name, nargin, nargout, docstring)
    __all__ += (name,)

if __name__ == '__main__':
    import odl
    z3 = odl.fn(3, dtype=int)
    ba = bitwise_and(z3)
    print(ba([[1, 2, 3], [4, 5, 6]]))

    r3 = odl.rn(3)
    s = sin(r3)
    print(s([1, 2, 3]))