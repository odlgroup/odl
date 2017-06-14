# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Ufuncs for ODL vectors.

These functions are internal and should only be used as methods on
`NtuplesBaseVector` type spaces.

See `numpy.ufuncs
<http://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_
for more information.

Notes
-----
The default implementation of these methods make heavy use of the
``NtuplesBaseVector.__array__`` to extract a `numpy.ndarray` from the vector,
and then apply a ufunc to it. Afterwards, ``NtuplesBaseVector.__array_wrap__``
is used to re-wrap the data into the appropriate space.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import re


__all__ = ('NtuplesBaseUfuncs', 'NumpyNtuplesUfuncs',
           'DiscreteLpUfuncs', 'ProductSpaceUfuncs')


# Some are ignored since they don't cooperate with dtypes, needs fix
RAW_UFUNCS = ['absolute', 'add', 'arccos', 'arccosh', 'arcsin', 'arcsinh',
              'arctan', 'arctan2', 'arctanh', 'bitwise_and', 'bitwise_or',
              'bitwise_xor', 'ceil', 'conj', 'copysign', 'cos', 'cosh',
              'deg2rad', 'divide', 'equal', 'exp', 'exp2', 'expm1', 'floor',
              'floor_divide', 'fmax', 'fmin', 'fmod', 'fmod', 'greater',
              'greater_equal', 'hypot', 'invert', 'isfinite', 'isinf', 'isnan',
              'left_shift', 'less', 'less_equal', 'log', 'log10', 'log1p',
              'log2', 'logaddexp', 'logaddexp2', 'logical_and', 'logical_not',
              'logical_or', 'logical_xor', 'maximum', 'minimum', 'mod', 'modf',
              'multiply', 'negative', 'not_equal', 'power',
              'rad2deg', 'reciprocal', 'remainder', 'right_shift', 'rint',
              'sign', 'signbit', 'sin', 'sinh', 'sqrt', 'square', 'subtract',
              'tan', 'tanh', 'true_divide', 'trunc']
# ,'isreal', 'iscomplex', 'ldexp', 'frexp'

# Add some standardized information
UFUNCS = []
for name in RAW_UFUNCS:
    ufunc = getattr(np, name)
    n_in, n_out = ufunc.nin, ufunc.nout
    descr = ufunc.__doc__.splitlines()[2]
    # Numpy occasionally uses single tics for doc, we only use them for links
    descr = re.sub('`+', '``', descr)
    doc = descr + """

See Also
--------
numpy.{}
""".format(name)
    UFUNCS.append((name, n_in, n_out, doc))


# Wrap all numpy ufuncs

def wrap_ufunc_base(name, n_in, n_out, doc):
    """Add ufunc methods to `NtuplesBaseUfuncs`."""
    ufunc = getattr(np, name)
    if n_in == 1:
        if n_out == 1:
            def wrapper(self, out=None):
                return self.vector.__array_ufunc__(
                    ufunc, '__call__', self.vector, out=(out,))

        elif n_out == 2:
            def wrapper(self, out1=None, out2=None):
                return self.vector.__array_ufunc__(
                    ufunc, '__call__', self.vector, out=(out1, out2))

        else:
            raise NotImplementedError

    elif n_in == 2:
        if n_out == 1:
            def wrapper(self, x2, out=None):
                return self.vector.__array_ufunc__(
                    ufunc, '__call__', self.vector, x2, out=(out,))

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    wrapper.__name__ = name
    wrapper.__doc__ = doc
    return wrapper


class NtuplesBaseUfuncs(object):

    """Ufuncs for `NtuplesBaseVector` objects.

    Internal object, should not be created except in `NtuplesBaseVector`.
    """

    def __init__(self, vector):
        """Create ufunc wrapper for vector."""
        self.vector = vector

    # Reductions for backwards compatibility
    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        """Return the sum of ``self``.

        See Also
        --------
        numpy.sum
        prod
        """
        return self.vector.__array_ufunc__(
            np.add, 'reduce', self.vector,
            axis=axis, dtype=dtype, out=(out,), keepdims=keepdims)

    def prod(self, axis=None, dtype=None, out=None, keepdims=False):
        """Return the product of ``self``.

        See Also
        --------
        numpy.prod
        sum
        """
        return self.vector.__array_ufunc__(
            np.multiply, 'reduce', self.vector,
            axis=axis, dtype=dtype, out=(out,), keepdims=keepdims)

    def min(self, axis=None, dtype=None, out=None, keepdims=False):
        """Return the minimum of ``self``.

        See Also
        --------
        numpy.amin
        max
        """
        return self.vector.__array_ufunc__(
            np.minimum, 'reduce', self.vector,
            axis=axis, dtype=dtype, out=(out,), keepdims=keepdims)

    def max(self, axis=None, dtype=None, out=None, keepdims=False):
        """Return the maximum of ``self``.

        See Also
        --------
        numpy.amax
        min
        """
        return self.vector.__array_ufunc__(
            np.maximum, 'reduce', self.vector,
            axis=axis, dtype=dtype, out=(out,), keepdims=keepdims)


# Add ufunc methods to ufunc class
for name, n_in, n_out, doc in UFUNCS:
    method = wrap_ufunc_base(name, n_in, n_out, doc)
    setattr(NtuplesBaseUfuncs, name, method)


class NumpyNtuplesUfuncs(NtuplesBaseUfuncs):

    """Ufuncs for `NumpyNtuplesVector` objects.

    Internal object, should not be created except in `NumpyNtuplesVector`.
    """


# Add ufunc methods to ufunc class
for name, n_in, n_out, doc in UFUNCS:
    # The base implementation is already optimal for Numpy-based vectors
    method = wrap_ufunc_base(name, n_in, n_out, doc)
    setattr(NumpyNtuplesUfuncs, name, method)


# Optimized implementation of ufuncs since we can use the out parameter
# as well as the data parameter to avoid one call to asarray() when using a
# NumpyNtuplesVector
def wrap_ufunc_discretelp(name, n_in, n_out, doc):
    """Add ufunc methods to `DiscreteLpUfuncs`."""

    if n_in == 1:
        if n_out == 0:
            def wrapper(self):
                method = getattr(self.vector.ntuple.ufuncs, name)
                return self.vector.space.element(method())

        elif n_out == 1:
            def wrapper(self, out=None):
                method = getattr(self.vector.ntuple.ufuncs, name)
                if out is None:
                    return self.vector.space.element(method())
                else:
                    method(out=out.ntuple)
                    return out

        elif n_out == 2:
            def wrapper(self, out1=None, out2=None):
                method = getattr(self.vector.ntuple.ufuncs, name)
                if out1 is None:
                    out1 = self.vector.space.element()
                if out2 is None:
                    out2 = self.vector.space.element()

                y1, y2 = method(out1.ntuple, out2.ntuple)
                return out1, out2

        else:
            raise NotImplementedError

    elif n_in == 2:
        if n_out == 1:
            def wrapper(self, x2, out=None):
                if x2 in self.vector.space:
                    x2 = x2.ntuple

                try:
                    # Try to reshape to linear data
                    x2 = x2.reshape(self.vector.size,
                                    order=self.vector.space.order)
                except AttributeError:
                    pass

                method = getattr(self.vector.ntuple.ufuncs, name)
                if out is None:
                    return self.vector.space.element(method(x2))
                else:
                    method(x2, out.ntuple)
                    return out

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    wrapper.__name__ = name
    wrapper.__doc__ = doc
    return wrapper


class DiscreteLpUfuncs(NtuplesBaseUfuncs):

    """Ufuncs for `DiscreteLpElement` objects.

    Internal object, should not be created except in `DiscreteLpElement`.
    """


# Add ufunc methods to ufunc class
for name, n_in, n_out, doc in UFUNCS:
    method = wrap_ufunc_discretelp(name, n_in, n_out, doc)
    setattr(DiscreteLpUfuncs, name, method)


# Ufuncs for product space elements
def wrap_ufunc_productspace(name, n_in, n_out, doc):
    """Add ufunc methods to `ProductSpaceElement`."""

    if n_in == 1:
        if n_out == 0:
            def wrapper(self):
                result = [getattr(x.ufuncs, name)() for x in self.vector]
                return self.vector.space.element(result)

        elif n_out == 1:
            def wrapper(self, out=None):
                if out is None:
                    result = [getattr(x.ufuncs, name)() for x in self.vector]
                    return self.vector.space.element(result)
                else:
                    for x, out_x in zip(self.vector, out):
                        getattr(x.ufuncs, name)(out=out_x)
                    return out

        elif n_out == 2:
            def wrapper(self, out1=None, out2=None):
                if out1 is None:
                    out1 = self.vector.space.element()
                if out2 is None:
                    out2 = self.vector.space.element()
                for x, out1_x, out2_x in zip(self.vector, out1, out2):
                    getattr(x.ufuncs, name)(out1=out1_x, out2=out2_x)
                return out1, out2

        else:
            raise NotImplementedError

    elif n_in == 2:
        if n_out == 1:
            def wrapper(self, x2, out=None):
                if x2 in self.vector.space:
                    if out is None:
                        result = [getattr(x.ufuncs, name)(x2p)
                                  for x, x2p in zip(self.vector, x2)]
                        return self.vector.space.element(result)
                    else:
                        for x, x2p, outp in zip(self.vector, x2, out):
                            getattr(x.ufuncs, name)(x2p, out=outp)
                        return out
                else:
                    if out is None:
                        result = [getattr(x.ufuncs, name)(x2)
                                  for x in self.vector]
                        return self.vector.space.element(result)
                    else:
                        for x, outp in zip(self.vector, out):
                            getattr(x.ufuncs, name)(x2, out=outp)
                        return out

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    wrapper.__name__ = name
    wrapper.__doc__ = doc
    return wrapper


class ProductSpaceUfuncs(object):

    """Ufuncs for `ProductSpaceElement` objects.

    Internal object, should not be created except in `ProductSpaceElement`.
    """
    def __init__(self, vector):
        """Create ufunc wrapper for vector."""
        self.vector = vector


# Add ufunc methods to ufunc class
for name, n_in, n_out, doc in UFUNCS:
    method = wrap_ufunc_productspace(name, n_in, n_out, doc)
    setattr(ProductSpaceUfuncs, name, method)
