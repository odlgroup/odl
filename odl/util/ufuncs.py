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

"""UFuncs for ODL vectors.

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


__all__ = ('NtuplesBaseUFuncs', 'NumpyNtuplesUFuncs',
           'DiscreteLpUFuncs', 'ProductSpaceUFuncs')


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
    """Add ufunc methods to `NtuplesBaseUFuncs`."""
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
    """Add ufunc methods to `NtuplesBaseUFuncs`."""
    wrapped = getattr(np, name)

    def wrapper(self):
        return wrapped(self.vector)

    wrapper.__name__ = name
    wrapper.__doc__ = doc
    return wrapper


class NtuplesBaseUFuncs(object):

    """UFuncs for `NtuplesBaseVector` objects.

    Internal object, should not be created except in `NtuplesBaseVector`.
    """

    def __init__(self, vector):
        """Create ufunc wrapper for vector."""
        self.vector = vector


# Add ufunc methods to UFunc class
for name, n_in, n_out, doc in UFUNCS:
    method = wrap_ufunc_base(name, n_in, n_out, doc)
    setattr(NtuplesBaseUFuncs, name, method)

# Add reduction methods to UFunc class
for name, doc in REDUCTIONS:
    method = wrap_reduction_base(name, doc)
    setattr(NtuplesBaseUFuncs, name, method)


# Optimized implementation of ufuncs since we can use the out parameter
# as well as the data parameter to avoid one call to asarray() when using an
# NumpyNtuplesVector
def wrap_ufunc_ntuples(name, n_in, n_out, doc):
    """Add ufunc methods to `NumpyNtuplesUFuncs`."""

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


class NumpyNtuplesUFuncs(NtuplesBaseUFuncs):

    """UFuncs for `NumpyNtuplesVector` objects.

    Internal object, should not be created except in `NumpyNtuplesVector`.
    """


# Add ufunc methods to UFunc class
for name, n_in, n_out, doc in UFUNCS:
    method = wrap_ufunc_ntuples(name, n_in, n_out, doc)
    setattr(NumpyNtuplesUFuncs, name, method)


# Optimized implementation of ufuncs since we can use the out parameter
# as well as the data parameter to avoid one call to asarray() when using a
# NumpyNtuplesVector
def wrap_ufunc_discretelp(name, n_in, n_out, doc):
    """Add ufunc methods to `DiscreteLpUFuncs`."""

    if n_in == 1:
        if n_out == 0:
            def wrapper(self):
                method = getattr(self.vector.ntuple.ufunc, name)
                return self.vector.space.element(method())

        elif n_out == 1:
            def wrapper(self, out=None):
                method = getattr(self.vector.ntuple.ufunc, name)
                if out is None:
                    return self.vector.space.element(method())
                else:
                    method(out=out.ntuple)
                    return out

        elif n_out == 2:
            def wrapper(self, out1=None, out2=None):
                method = getattr(self.vector.ntuple.ufunc, name)
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

                method = getattr(self.vector.ntuple.ufunc, name)
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


def wrap_reduction_discretelp(name, doc):
    def wrapper(self):
        method = getattr(self.vector.ntuple.ufunc, name)
        return method()

    wrapper.__name__ = name
    wrapper.__doc__ = doc
    return wrapper


class DiscreteLpUFuncs(NtuplesBaseUFuncs):

    """UFuncs for `DiscreteLpElement` objects.

    Internal object, should not be created except in `DiscreteLpElement`.
    """


# Add ufunc methods to UFunc class
for name, n_in, n_out, doc in UFUNCS:
    method = wrap_ufunc_discretelp(name, n_in, n_out, doc)
    setattr(DiscreteLpUFuncs, name, method)

for name, doc in REDUCTIONS:
    method = wrap_reduction_discretelp(name, doc)
    setattr(DiscreteLpUFuncs, name, method)


# Ufuncs for product space elements
def wrap_ufunc_productspace(name, n_in, n_out, doc):
    """Add ufunc methods to `ProductSpaceElement`."""

    if n_in == 1:
        if n_out == 0:
            def wrapper(self):
                result = [getattr(x.ufunc, name)() for x in self.vector]
                return self.vector.space.element(result)

        elif n_out == 1:
            def wrapper(self, out=None):
                if out is None:
                    result = [getattr(x.ufunc, name)() for x in self.vector]
                    return self.vector.space.element(result)
                else:
                    for x, out_x in zip(self.vector, out):
                        getattr(x.ufunc, name)(out=out_x)
                    return out

        elif n_out == 2:
            def wrapper(self, out1=None, out2=None):
                if out1 is None:
                    out1 = self.vector.space.element()
                if out2 is None:
                    out2 = self.vector.space.element()
                for x, out1_x, out2_x in zip(self.vector, out1, out2):
                    getattr(x.ufunc, name)(out1=out1_x, out2=out2_x)
                return out1, out2

        else:
            raise NotImplementedError

    elif n_in == 2:
        if n_out == 1:
            def wrapper(self, x2, out=None):
                if x2 in self.vector.space:
                    if out is None:
                        result = [getattr(x.ufunc, name)(x2p)
                                  for x, x2p in zip(self.vector, x2)]
                        return self.vector.space.element(result)
                    else:
                        for x, x2p, outp in zip(self.vector, x2, out):
                            getattr(x.ufunc, name)(x2p, out=outp)
                        return out
                else:
                    if out is None:
                        result = [getattr(x.ufunc, name)(x2)
                                  for x in self.vector]
                        return self.vector.space.element(result)
                    else:
                        for x, outp in zip(self.vector, out):
                            getattr(x.ufunc, name)(x2, out=outp)
                        return out

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    wrapper.__name__ = name
    wrapper.__doc__ = doc
    return wrapper


def wrap_reduction_productspace(name, doc):
    """Add reduction methods to `ProductSpaceElement`."""
    def wrapper(self):
        results = [getattr(x.ufunc, name)() for x in self.vector]
        return getattr(np, name)(results)

    wrapper.__name__ = name
    wrapper.__doc__ = doc
    return wrapper


class ProductSpaceUFuncs(object):

    """UFuncs for `ProductSpaceElement` objects.

    Internal object, should not be created except in `ProductSpaceElement`.
    """
    def __init__(self, vector):
        """Create ufunc wrapper for vector."""
        self.vector = vector


# Add ufunc methods to UFunc class
for name, n_in, n_out, doc in UFUNCS:
    method = wrap_ufunc_productspace(name, n_in, n_out, doc)
    setattr(ProductSpaceUFuncs, name, method)


# Add reduction methods to UFunc class
for name, doc in REDUCTIONS:
    method = wrap_reduction_productspace(name, doc)
    setattr(ProductSpaceUFuncs, name, method)
