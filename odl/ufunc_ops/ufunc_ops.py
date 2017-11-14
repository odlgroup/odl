# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Ufunc operators for ODL vectors."""

from __future__ import print_function, division, absolute_import
import numpy as np

from odl.set import LinearSpace, RealNumbers, Field
from odl.space import ProductSpace, tensor_space
from odl.operator import Operator, MultiplyOperator
from odl.solvers import (Functional, ScalingFunctional, FunctionalQuotient,
                         ConstantFunctional)
from odl.util.ufuncs import UFUNCS

__all__ = ()

SUPP_TYPECODES = '?bhilqpBHILQPefdgFDG'
SUPP_TYPECODES_TO_DTYPES = {tc: np.dtype(tc) for tc in SUPP_TYPECODES}


def find_min_signature(ufunc, dtypes_in):
    """Determine the minimum matching ufunc signature for given dtypes.

    Parameters
    ----------
    ufunc : str or numpy.ufunc
        Ufunc whose signatures are to be considered.
    dtypes_in :
        Sequence of objects specifying input dtypes. Its length must match
        the number of inputs of ``ufunc``, and its entries must be understood
        by `numpy.dtype`.

    Returns
    -------
    signature : str
        Minimum matching ufunc signature, see, e.g., ``np.add.types``
        for examples.

    Raises
    ------
    TypeError
        If no valid signature is found.
    """
    if not isinstance(ufunc, np.ufunc):
        ufunc = getattr(np, str(ufunc))

    dtypes_in = [np.dtype(dt_in) for dt_in in dtypes_in]
    tcs_in = [dt.base.char for dt in dtypes_in]

    if len(tcs_in) != ufunc.nin:
        raise ValueError('expected {} input dtype(s) for {}, got {}'
                         ''.format(ufunc.nin, ufunc, len(tcs_in)))

    valid_sigs = []
    for sig in ufunc.types:
        sig_tcs_in, sig_tcs_out = sig.split('->')
        if all(np.dtype(tc_in) <= np.dtype(sig_tc_in) and
               sig_tc_in in SUPP_TYPECODES
               for tc_in, sig_tc_in in zip(tcs_in, sig_tcs_in)):
            valid_sigs.append(sig)

    if not valid_sigs:
        raise TypeError('no valid signature found for {} and input dtypes {}'
                        ''.format(ufunc, tuple(dt.name for dt in dtypes_in)))

    def in_dtypes(sig):
        """Comparison key function for input dtypes of a signature."""
        sig_tcs_in = sig.split('->')[0]
        return tuple(np.dtype(tc) for tc in sig_tcs_in)

    return min(valid_sigs, key=in_dtypes)


def dtypes_out(ufunc, dtypes_in):
    """Return the result dtype(s) of ``ufunc`` with inputs of given dtypes."""
    sig = find_min_signature(ufunc, dtypes_in)
    tcs_out = sig.split('->')[1]
    return tuple(np.dtype(tc) for tc in tcs_out)


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

        if nargin == 1:
            domain = space0 = space
            dtypes = [space.dtype]
        elif nargin == len(space) == 2 and isinstance(space, ProductSpace):
            domain = space
            space0 = space[0]
            dtypes = [space[0].dtype, space[1].dtype]
        else:
            domain = ProductSpace(space, nargin)
            space0 = space
            dtypes = [space.dtype, space.dtype]

        dts_out = dtypes_out(name, dtypes)

        if nargout == 1:
            range = space0.astype(dts_out[0])
        else:
            range = ProductSpace(space0.astype(dts_out[0]),
                                 space0.astype(dts_out[1]))

        linear = name in LINEAR_UFUNCS
        Operator.__init__(self, domain=domain, range=range, linear=linear)

    def _call(self, x, out=None):
        """Return ``self(x)``."""
        # TODO: use `__array_ufunc__` when implemented on `ProductSpace`,
        # or try both
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
        result_space = ProductSpace(vec.space, 2)
        result = repr(result_space.element(result))

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
    from odl.util.testutils import run_doctests
    run_doctests()
