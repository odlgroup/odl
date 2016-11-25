# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Ufunc operators for ODL vectors."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
from odl.set import LinearSpace, RealNumbers, Field
from odl.space import ProductSpace, tensor_space
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
>>> space = odl.{space!r}
>>> op = odl.ufunc_ops.{name}(space)
>>> print(op({arg}))
{result!s}
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
            return reciprocal(self.domain)
    elif name == 'exp':
        def gradient(self):
            """Return the gradient operator."""
            return self
    elif name == 'reciprocal':
        def gradient(self):
            """Return the gradient operator."""
            return FunctionalQuotient(ConstantFunctional(self.domain, -1.0),
                                      square(self.domain))
    elif name == 'sinh':
        def gradient(self):
            """Return the gradient operator."""
            return cosh(self.domain)
    elif name == 'cosh':
        def gradient(self):
            """Return the gradient operator."""
            return sinh(self.domain)
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
    elif name == 'reciprocal':
        def derivative(self, point):
            """Return the derivative operator."""
            point = self.domain.element(point)
            return MultiplyOperator(-self(point) ** 2)
    elif name == 'sinh':
        def derivative(self, point):
            """Return the derivative operator."""
            point = self.domain.element(point)
            return MultiplyOperator(cosh(self.domain)(point))
    elif name == 'cosh':
        def derivative(self, point):
            """Return the derivative operator."""
            return MultiplyOperator(sinh(self.domain)(point))
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
        space : `TensorSpace`
            The domain of the operator.
        """
        if not isinstance(space, LinearSpace):
            raise TypeError('`space` {!r} not a `LinearSpace`'.format(space))

        if _is_integer_only_ufunc(name) and not is_int_dtype(space.dtype):
            raise ValueError("ufunc '{}' only defined with integral dtype"
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
        """Return ``self(x)``."""
        if out is None:
            if nargin == 1:
                return getattr(x.ufuncs, name)()
            else:
                return getattr(x[0].ufuncs, name)(*x[1:])
        else:
            if nargin == 1:
                return getattr(x.ufuncs, name)(out=out)
            else:
                return getattr(x[0].ufuncs, name)(*x[1:], out=out)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(name, self.domain)

    # Create example (also functions as doctest)
    if 'shift' in name or 'bitwise' in name or name == 'invert':
        dtype = int
    else:
        dtype = float

    space = tensor_space(3, dtype=dtype)
    if nargin == 1:
        vec = space.element([-1, 1, 2])
        arg = '{}'.format(vec)
        with np.errstate(all='ignore'):
            result = getattr(vec.ufuncs, name)()
    else:
        vec = space.element([-1, 1, 2])
        vec2 = space.element([3, 4, 5])
        arg = '[{}, {}]'.format(vec, vec2)
        with np.errstate(all='ignore'):
            result = getattr(vec.ufuncs, name)(vec2)

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
    """Create a ufunc `Functional` from a given specification."""

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
            raise ValueError("ufunc '{}' only defined with integral dtype"
                             "".format(name))

        linear = name in LINEAR_UFUNCS
        Functional.__init__(self, space=field, linear=linear)

    def _call(self, x):
        """Return ``self(x)``."""
        if nargin == 1:
            return getattr(np, name)(x)
        else:
            return getattr(np, name)(*x)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(name, self.domain)

    # Create example (also functions as doctest)

    if nargin != 1:
        raise NotImplementedError('Currently not suppored')

    if nargout != 1:
        raise NotImplementedError('Currently not suppored')

    space = RealNumbers()
    val = 1.0
    arg = '{}'.format(val)
    with np.errstate(all='ignore'):
        result = np.float64(getattr(np, name)(val))

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


RAW_UFUNC_FACTORY_DOCSTRING = """{docstring}
Notes
-----
This creates a `Operator`/`Functional` that applies a ufunc pointwise.

Examples
--------
{operator_example}
{functional_example}
"""

RAW_UFUNC_FACTORY_FUNCTIONAL_DOCSTRING = """
Create functional with domain/range as real numbers:

>>> func = odl.ufunc_ops.{name}()
"""

RAW_UFUNC_FACTORY_OPERATOR_DOCSTRING = """
Create operator that acts pointwise on a `TensorSpace`

>>> space = odl.rn(3)
>>> op = odl.ufunc_ops.{name}(space)
"""


# Create an operator for each ufunc
for name, nargin, nargout, docstring in UFUNCS:
    def indirection(name, docstring):
        # Indirection is needed since name should be saved but is changed
        # in the loop.

        def ufunc_factory(domain=RealNumbers()):
            # Create a `Operator` or `Functional` depending on arguments
            try:
                if isinstance(domain, Field):
                    return globals()[name + '_func'](domain)
                else:
                    return globals()[name + '_op'](domain)
            except KeyError:
                raise ValueError('ufunc not available for {}'.format(domain))
        return ufunc_factory

    globals()[name + '_op'] = ufunc_class_factory(name, nargin,
                                                  nargout, docstring)
    if not _is_integer_only_ufunc(name):
        operator_example = RAW_UFUNC_FACTORY_OPERATOR_DOCSTRING.format(
            name=name)
    else:
        operator_example = ""

    if not _is_integer_only_ufunc(name) and nargin == 1 and nargout == 1:
        globals()[name + '_func'] = ufunc_functional_factory(
            name, nargin, nargout, docstring)
        functional_example = RAW_UFUNC_FACTORY_FUNCTIONAL_DOCSTRING.format(
            name=name)
    else:
        functional_example = ""

    ufunc_factory = indirection(name, docstring)

    ufunc_factory.__doc__ = RAW_UFUNC_FACTORY_DOCSTRING.format(
        docstring=docstring, name=name,
        functional_example=functional_example,
        operator_example=operator_example)

    globals()[name] = ufunc_factory
    __all__ += (name,)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
