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
from odl.set import LinearSpace, RealNumbers, Field
from odl.space import ProductSpace, fn
from odl.space.base_ntuples import FnBase
from odl.operator import Operator, MultiplyOperator
from odl.util.utility import is_int_dtype
from odl.util.ufuncs import UFUNCS
from odl.solvers import (Functional, ScalingFunctional, FunctionalQuotient,
                         ConstantFunctional, IdentityFunctional)

__all__ = ()


def _is_integer_only_ufunc(name):
    return 'shift' in name or 'bitwise' in name or name == 'invert'

LINEAR_UFUNCS = ['negative', 'rad2deg', 'deg2rad', 'add', 'subtract']


RAW_EXAMPLES_DOCSTRING = """
Examples
--------
>>> import odl
>>> space = odl.{space}
>>> op = {name}(space)
>>> print(op({arg}))
{result}
"""


def gradient_factory(name):
    """Create gradient `Functional` for some ufuncs."""

    if name == 'sin':
        def gradient(self):
            """Return the gradient operator."""
            return cos(self.domain)
    elif name == 'cos':
        def gradient(self):
            """Return the gradient operator."""
            return -sin(self.domain)
    elif name == 'tan':
        def gradient(self):
            """Return the gradient operator."""
            return 1 + square(self.domain) * self
    elif name == 'sqrt':
        def gradient(self):
            """Return the gradient operator."""
            return FunctionalQuotient(ConstantFunctional(self.domain, 0.5),
                                      self)
    elif name == 'square':
        def gradient(self):
            """Return the gradient operator."""
            return ScalingFunctional(self.domain, 2.0)
    elif name == 'log':
        def gradient(self):
            """Return the gradient operator."""
            return FunctionalQuotient(ConstantFunctional(self.domain, 1.0),
                                      IdentityFunctional(self.domain))
    elif name == 'exp':
        def gradient(self):
            """Return the gradient operator."""
            return self
    else:
        # Fallback to default
        gradient = Functional.gradient

    return gradient


def derivative_factory(name):
    """Create derivative function for some ufuncs."""

    if name == 'sin':
        def derivative(self, point):
            """Return the derivative operator."""
            return MultiplyOperator(cos(self.domain)(point))
    elif name == 'cos':
        def derivative(self, point):
            """Return the derivative operator."""
            point = self.domain.element(point)
            return MultiplyOperator(-sin(self.domain)(point))
    elif name == 'tan':
        def derivative(self, point):
            """Return the derivative operator."""
            return MultiplyOperator(1 + self(point) ** 2)
    elif name == 'sqrt':
        def derivative(self, point):
            """Return the derivative operator."""
            return MultiplyOperator(0.5 / self(point))
    elif name == 'square':
        def derivative(self, point):
            """Return the derivative operator."""
            point = self.domain.element(point)
            return MultiplyOperator(2.0 * point)
    elif name == 'log':
        def derivative(self, point):
            """Return the derivative operator."""
            point = self.domain.element(point)
            return MultiplyOperator(1.0 / point)
    elif name == 'exp':
        def derivative(self, point):
            """Return the derivative operator."""
            return MultiplyOperator(self(point))
    else:
        # Fallback to default
        derivative = Operator.derivative

    return derivative


def ufunc_class_factory(name, nargin, nargout, docstring):
    """Create a Ufunc `Operator` from a given specification."""

    assert 0 <= nargin <= 2

    def __init__(self, space):
        """Initialize an instance.

        Parameters
        ----------
        space : `FnBase`
            The domain of the operator.
        """
        if not isinstance(space, LinearSpace):
            raise TypeError('`space` {!r} not a `LinearSpace`'.format(space))

        if _is_integer_only_ufunc(name) and not is_int_dtype(space.dtype):
            raise ValueError("UFunc '{}' only defined with integral dtype"
                             "".format(name))

        if nargin == 1:
            domain = space
        else:
            domain = ProductSpace(space, nargin)

        if nargout == 1:
            range = space
        else:
            range = ProductSpace(space, nargout)

        linear = name in LINEAR_UFUNCS
        Operator.__init__(self, domain=domain, range=range, linear=linear)

    def _call(self, x, out=None):
        """return ``self(x)``."""
        if out is None:
            if nargin == 1:
                return getattr(x.ufunc, name)()
            else:
                return getattr(x[0].ufunc, name)(*x[1:])
        else:
            if nargin == 1:
                return getattr(x.ufunc, name)(out=out)
            else:
                return getattr(x[0].ufunc, name)(*x[1:], out=out)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(name, self.domain)

    # Create example (also functions as doctest)
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

    if nargout == 2:
        result = '{{{}, {}}}'.format(result[0], result[1])

    examples_docstring = RAW_EXAMPLES_DOCSTRING.format(space=space, name=name,
                                                       arg=arg, result=result)
    full_docstring = docstring + examples_docstring

    attributes = {"__init__": __init__,
                  "_call": _call,
                  "derivative": derivative_factory(name),
                  "__repr__": __repr__,
                  "__doc__": full_docstring}

    full_name = name + '_op'

    return type(full_name, (Operator,), attributes)


def ufunc_functional_factory(name, nargin, nargout, docstring):
    """Create a UFunc `Functional` from a given specification."""

    assert 0 <= nargin <= 2

    def __init__(self, field):
        """Initialize an instance.

        Parameters
        ----------
        field : `Field`
            The domain of the functional.
        """
        if not isinstance(field, Field):
            raise TypeError('`field` {!r} not a `Field`'.format(space))

        if _is_integer_only_ufunc(name):
            raise ValueError("UFunc '{}' only defined with integral dtype"
                             "".format(name))

        linear = name in LINEAR_UFUNCS
        Functional.__init__(self, space=field, linear=linear)

    def _call(self, x):
        """return ``self(x)``."""
        if nargin == 1:
            return getattr(np, name)(x)
        else:
            return getattr(np, name)(*x)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(name, self.domain)

    # Create example (also functions as doctest)

    space = RealNumbers()
    if nargin == 1:
        val = 1.0
        arg = '{}'.format(val)
        with np.errstate(all='ignore'):
            result = getattr(np, name)(val)
    else:
        val1 = 1.0
        val2 = 2.0
        arg = '[{}, {}]'.format(val1, val2)
        with np.errstate(all='ignore'):
            result = getattr(np, name)(val1, val2)

    if nargout == 2:
        result = '{{{}, {}}}'.format(result[0], result[1])

    examples_docstring = RAW_EXAMPLES_DOCSTRING.format(space=space, name=name,
                                                       arg=arg, result=result)
    full_docstring = docstring + examples_docstring

    attributes = {"__init__": __init__,
                  "_call": _call,
                  "gradient": property(gradient_factory(name)),
                  "__repr__": __repr__,
                  "__doc__": full_docstring}

    full_name = name + '_op'

    return type(full_name, (Functional,), attributes)


# Create an operator for each ufunc
for name, nargin, nargout, docstring in UFUNCS:
    def indirection(name):
        def ufunc_factory(domain=RealNumbers()):
            if isinstance(domain, Field):
                return globals()[name + '_func'](domain)
            elif isinstance(domain, FnBase):
                return globals()[name + '_op'](domain)
            else:
                raise ValueError('UFunc not available for {}'.format(domain))
        return ufunc_factory

    if not _is_integer_only_ufunc(name):
        globals()[name + '_op'] = ufunc_class_factory(name, nargin, nargout, docstring)
        globals()[name + '_func'] = ufunc_functional_factory(name, nargin, nargout, docstring)
        globals()[name] = indirection(name)
        __all__ += (name,)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
