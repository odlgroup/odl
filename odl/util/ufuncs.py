# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Ufuncs for ODL vectors.

These functions are internal and should only be used as methods on
`GeneralizedTensor` type spaces.

See `numpy.ufuncs
<http://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_
for more information.

Notes
-----
The default implementation of these methods make heavy use of the
``GeneralizedTensor.__array__`` to extract a `numpy.ndarray` from the vector,
and then apply a ufunc to it. Afterwards, ``GeneralizedTensor.__array_wrap__``
is used to re-wrap the data into the appropriate space.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import re


__all__ = ('TensorSetUfuncs', 'NumpyTensorSetUfuncs',
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

RAW_REDUCTIONS = [('sum', 'sum', 'Sum of array elements.'),
                  ('prod', 'prod', 'Product of array elements.'),
                  ('min', 'amin', 'Minimum value in array.'),
                  ('max', 'amax', 'Maximum value in array.')]

REDUCTIONS = []
for name, numpyname, descr in RAW_REDUCTIONS:
    doc = descr + """

See Also
--------
numpy.{}
""".format(numpyname)
    REDUCTIONS += [(name, doc)]


# Wrap all numpy ufuncs

def wrap_ufunc_base(name, n_in, n_out, doc):
    """Return ufunc wrapper for implementation-agnostic ufunc classes."""
    wrapped = getattr(np, name)
    if n_in == 1:
        if n_out == 0:
            def wrapper(self):
                return wrapped(self.vector)

        elif n_out == 1:
            def wrapper(self, out=None):
                if out is None:
                    out = self.vector.space.element()

                out[:] = wrapped(self.vector)
                return out

        elif n_out == 2:
            def wrapper(self, out1=None, out2=None):
                if out1 is None:
                    out1 = self.vector.space.element()
                if out2 is None:
                    out2 = self.vector.space.element()

                [y1, y2] = wrapped(self.vector)
                out1[:] = y1
                out2[:] = y2
                return out1, out2

        else:
            raise NotImplementedError

    elif n_in == 2:
        if n_out == 1:
            def wrapper(self, x2, out=None):
                if out is None:
                    return wrapped(self.vector, x2)
                else:
                    out[:] = wrapped(self.vector, x2)
                    return out

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    wrapper.__name__ = name
    wrapper.__doc__ = doc
    return wrapper


# Wrap reductions
def wrap_reduction_base(name, doc):
    """Return reduction wrapper for implementation-agnostic reductions."""
    wrapped = getattr(np, name)

    def wrapper(self):
        return wrapped(self.vector)

    wrapper.__name__ = name
    wrapper.__doc__ = doc
    return wrapper


class TensorSetUfuncs(object):

    """Ufuncs for `GeneralizedTensor` objects.

    Internal object, should not be created except in `GeneralizedTensor`.
    """

    def __init__(self, vector):
        """Create ufunc wrapper for vector."""
        self.vector = vector


# Add ufunc methods to ufunc class
for name, n_in, n_out, doc in UFUNCS:
    method = wrap_ufunc_base(name, n_in, n_out, doc)
    setattr(TensorSetUfuncs, name, method)

# Add reduction methods to ufunc class
for name, doc in REDUCTIONS:
    method = wrap_reduction_base(name, doc)
    setattr(TensorSetUfuncs, name, method)


# Optimized implementation of ufuncs since we can use the out parameter
# as well as the data parameter to avoid one call to asarray() when using a
# Numpy-based data class.
def wrap_ufunc_numpy(name, n_in, n_out, doc):
    """Return ufunc wrapper for Numpy-based ufunc classes."""
    # Get method from numpy
    wrapped = getattr(np, name)
    if n_in == 1:
        if n_out == 0:
            def wrapper(self):
                return wrapped(self.vector)

        elif n_out == 1:
            def wrapper(self, out=None):
                if out is None:
                    out = self.vector.space.element()
                wrapped(self.vector, out.data)
                return out

        elif n_out == 2:
            def wrapper(self, out1=None, out2=None):
                if out1 is None:
                    out1 = self.vector.space.element()
                if out2 is None:
                    out2 = self.vector.space.element()

                y1, y2 = wrapped(self.vector, out1.data, out2.data)
                return out1, out2

        else:
            raise NotImplementedError

    elif n_in == 2:
        if n_out == 1:
            def wrapper(self, x2, out=None):
                if out is None:
                    out = self.vector.space.element()

                wrapped(self.vector, x2, out.data)
                return out

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    wrapper.__name__ = name
    wrapper.__doc__ = doc
    return wrapper


class NumpyTensorSetUfuncs(TensorSetUfuncs):

    """Ufuncs for `NumpyGeneralizedTensor` objects.

    Internal object, should not be created except in
    `NumpyGeneralizedTensor`.
    """


# Add ufunc methods to ufunc class
for name, n_in, n_out, doc in UFUNCS:
    method = wrap_ufunc_numpy(name, n_in, n_out, doc)
    setattr(NumpyTensorSetUfuncs, name, method)


# For DiscreteLP, basically the ufunc mechanism can be propagated from its
# `tensor` attribute, which is NumpyTensor or CudaNtuple. Sometimes,
# reshaping is required.
def wrap_ufunc_discretelp(name, n_in, n_out, doc):
    """Return ufunc wrapper for `DiscreteLpUfuncs`."""
    if n_in == 1:
        if n_out == 0:
            def wrapper(self):
                method = getattr(self.vector.tensor.ufuncs, name)
                return self.vector.space.element(method())

        elif n_out == 1:
            def wrapper(self, out=None):
                method = getattr(self.vector.tensor.ufuncs, name)
                if out is None:
                    return self.vector.space.element(method())
                else:
                    method(out=out.tensor)
                    return out

        elif n_out == 2:
            def wrapper(self, out1=None, out2=None):
                method = getattr(self.vector.tensor.ufuncs, name)
                if out1 is None:
                    out1 = self.vector.space.element()
                if out2 is None:
                    out2 = self.vector.space.element()

                y1, y2 = method(out1.tensor, out2.tensor)
                return out1, out2

        else:
            raise NotImplementedError

    elif n_in == 2:
        if n_out == 1:
            def wrapper(self, x2, out=None):
                if x2 in self.vector.space:
                    x2 = x2.tensor

                try:
                    # Try to reshape to linear data
                    x2 = x2.reshape(self.vector.size,
                                    order=self.vector.space.order)
                except AttributeError:
                    pass

                method = getattr(self.vector.tensor.ufuncs, name)
                if out is None:
                    return self.vector.space.element(method(x2))
                else:
                    method(x2, out.tensor)
                    return out

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    wrapper.__name__ = name
    wrapper.__doc__ = doc
    return wrapper


def wrap_reduction_discretelp(name, doc):
    """Return reduction wrapper for `DiscreteLpUfuncs`."""

    def wrapper(self):
        method = getattr(self.vector.tensor.ufuncs, name)
        return method()

    wrapper.__name__ = name
    wrapper.__doc__ = doc
    return wrapper


class DiscreteLpUfuncs(TensorSetUfuncs):

    """Ufuncs for `DiscreteLpElement` objects.

    Internal object, should not be created except in `DiscreteLpElement`.
    """


# Add ufunc methods to ufunc class
for name, n_in, n_out, doc in UFUNCS:
    method = wrap_ufunc_discretelp(name, n_in, n_out, doc)
    setattr(DiscreteLpUfuncs, name, method)

for name, doc in REDUCTIONS:
    method = wrap_reduction_discretelp(name, doc)
    setattr(DiscreteLpUfuncs, name, method)


# Ufuncs for product space elements
def wrap_ufunc_productspace(name, n_in, n_out, doc):
    """Return ufunc wrapper for `ProductSpaceUfuncs`."""

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


def wrap_reduction_productspace(name, doc):
    """Return reduction wrapper for `ProductSpaceUfuncs`."""
    def wrapper(self):
        results = [getattr(x.ufuncs, name)() for x in self.vector]
        return getattr(np, name)(results)

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


# Add reduction methods to ufunc class
for name, doc in REDUCTIONS:
    method = wrap_reduction_productspace(name, doc)
    setattr(ProductSpaceUfuncs, name, method)
