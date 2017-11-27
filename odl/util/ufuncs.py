# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Universal functions (ufuncs) for ODL-wrapped arrays.

These functions are internal and should only be used as methods on
`Tensor`-like classes.

See `numpy.ufuncs
<http://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_
for more information.

Notes
-----
The default implementation of these methods uses the ``__array_ufunc__``
dispatch machinery `introduced in Numpy 1.13
<https://github.com/charris/numpy/blob/master/doc/source/reference/\
arrays.classes.rst#special-attributes-and-methods>`_.
"""

from __future__ import print_function, division, absolute_import
from builtins import object
import numpy as np
import re


__all__ = ('TensorSpaceUfuncs', 'ProductSpaceUfuncs')


_npy_maj, _npy_min = [int(n) for n in np.__version__.split('.')[:2]]

# Supported by Numpy 1.9 and higher
UFUNC_NAMES = [
    'absolute', 'add', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan',
    'arctan2', 'arctanh', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'ceil',
    'conj', 'conjugate', 'copysign', 'cos', 'cosh', 'deg2rad', 'degrees',
    'divide', 'equal', 'exp', 'exp2', 'expm1', 'fabs', 'floor', 'floor_divide',
    'fmax', 'fmin', 'fmod', 'frexp', 'greater', 'greater_equal', 'hypot',
    'invert', 'isfinite', 'isinf', 'isnan', 'ldexp', 'left_shift', 'less',
    'less_equal', 'log', 'log10', 'log1p', 'log2', 'logaddexp', 'logaddexp2',
    'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'maximum',
    'minimum', 'mod', 'modf', 'multiply', 'negative', 'nextafter',
    'not_equal', 'power', 'rad2deg', 'radians', 'reciprocal', 'remainder',
    'right_shift', 'rint', 'sign', 'signbit', 'sin', 'sinh', 'sqrt',
    'square', 'spacing', 'subtract', 'tan', 'tanh', 'true_divide', 'trunc']

if (_npy_maj, _npy_min) >= (1, 10):
    UFUNC_NAMES.extend(['abs', 'cbrt', 'bitwise_not'])

if (_npy_maj, _npy_min) >= (1, 12):
    UFUNC_NAMES.extend(['float_power'])

if (_npy_maj, _npy_min) >= (1, 13):
    UFUNC_NAMES.extend(['divmod', 'heaviside', 'positive'])

# Add some standardized information
UFUNCS = []
for name in UFUNC_NAMES:
    ufunc = getattr(np, name)
    n_in, n_out = ufunc.nin, ufunc.nout
    descr = ufunc.__doc__.splitlines()[2]
    # Numpy occasionally uses single ticks for doc, we only use them for links
    descr = re.sub('`+', '``', descr)
    doc = descr + """

See Also
--------
numpy.{}
""".format(name)
    UFUNCS.append((name, n_in, n_out, doc))

# TODO: add the following reductions (to the CUDA implementation):
# ['var', 'trace', 'tensordot', 'std', 'ptp', 'mean', 'diff', 'cumsum',
#  'cumprod', 'average']


# --- Wrappers for `Tensor` --- #


def wrap_ufunc_base(name, n_in, n_out, doc):
    """Return ufunc wrapper for implementation-agnostic ufunc classes."""
    ufunc = getattr(np, name)
    if n_in == 1:
        if n_out == 1:
            def wrapper(self, out=None, **kwargs):
                if out is None or isinstance(out, (type(self.elem),
                                                   type(self.elem.data))):
                    out = (out,)

                return self.elem.__array_ufunc__(
                    ufunc, '__call__', self.elem, out=out, **kwargs)

        elif n_out == 2:
            def wrapper(self, out=None, **kwargs):
                if out is None:
                    out = (None, None)

                return self.elem.__array_ufunc__(
                    ufunc, '__call__', self.elem, out=out, **kwargs)

        else:
            raise NotImplementedError

    elif n_in == 2:
        if n_out == 1:
            def wrapper(self, x2, out=None, **kwargs):
                return self.elem.__array_ufunc__(
                    ufunc, '__call__', self.elem, x2, out=(out,), **kwargs)

        elif n_out == 2:
            def wrapper(self, x2, out=None, **kwargs):
                if out is None:
                    out = (None, None)

                return self.elem.__array_ufunc__(
                    ufunc, '__call__', self.elem, x2, out=out, **kwargs)

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    wrapper.__name__ = wrapper.__qualname__ = name
    wrapper.__doc__ = doc
    return wrapper


class TensorSpaceUfuncs(object):

    """Ufuncs for `Tensor` objects.

    Internal object, should not be created except in `Tensor`.
    """

    def __init__(self, elem):
        """Create ufunc wrapper for elem."""
        self.elem = elem

    # Reductions for backwards compatibility

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        """Return the sum of ``self``.

        See Also
        --------
        numpy.sum
        prod
        """
        return self.elem.__array_ufunc__(
            np.add, 'reduce', self.elem,
            axis=axis, dtype=dtype, out=(out,), keepdims=keepdims)

    def prod(self, axis=None, dtype=None, out=None, keepdims=False):
        """Return the product of ``self``.

        See Also
        --------
        numpy.prod
        sum
        """
        return self.elem.__array_ufunc__(
            np.multiply, 'reduce', self.elem,
            axis=axis, dtype=dtype, out=(out,), keepdims=keepdims)

    def min(self, axis=None, dtype=None, out=None, keepdims=False):
        """Return the minimum of ``self``.

        See Also
        --------
        numpy.amin
        max
        """
        return self.elem.__array_ufunc__(
            np.minimum, 'reduce', self.elem,
            axis=axis, dtype=dtype, out=(out,), keepdims=keepdims)

    def max(self, axis=None, dtype=None, out=None, keepdims=False):
        """Return the maximum of ``self``.

        See Also
        --------
        numpy.amax
        min
        """
        return self.elem.__array_ufunc__(
            np.maximum, 'reduce', self.elem,
            axis=axis, dtype=dtype, out=(out,), keepdims=keepdims)


# Add ufunc methods to ufunc class
for name, n_in, n_out, doc in UFUNCS:
    method = wrap_ufunc_base(name, n_in, n_out, doc)
    setattr(TensorSpaceUfuncs, name, method)


# --- Wrappers for `ProductSpaceElement` --- #


def wrap_ufunc_productspace(name, n_in, n_out, doc):
    """Return ufunc wrapper for `ProductSpaceUfuncs`."""
    if n_in == 1:
        if n_out == 1:
            def wrapper(self, out=None, **kwargs):
                from odl.space.pspace import ProductSpace
                if out is None:
                    out = [None] * len(self.elem.space)

                res = []
                for xi, out_i in zip(self.elem, out):
                    r = getattr(xi.ufuncs, name)(out=out_i, **kwargs)
                    res.append(r)
                out_space = ProductSpace(*[r.space for r in res])
                return out_space.element(res)

        elif n_out == 2:
            def wrapper(self, out1=None, out2=None, **kwargs):
                from odl.space.pspace import ProductSpace
                if out1 is None:
                    out1 = [None] * len(self.elem.space)
                if out2 is None:
                    out2 = [None] * len(self.elem.space)

                res1, res2 = [], []
                for xi, out1_i, out2_i in zip(self.elem, out1, out2):
                    r1, r2 = getattr(xi.ufuncs, name)(out1=out1_i,
                                                      out2=out2_i,
                                                      **kwargs)
                    res1.append(r1)
                    res2.append(r2)
                out_space_1 = ProductSpace(*[r.space for r in res1])
                out_space_2 = ProductSpace(*[r.space for r in res2])
                return out_space_1.element(res1), out_space_2.element(res2)

        else:
            raise NotImplementedError

    elif n_in == 2:
        if n_out == 1:
            def wrapper(self, x2, out=None, **kwargs):
                from odl.space.pspace import ProductSpace
                if out is None:
                    out = [None] * len(self.elem.space)

                res = []
                for x1_i, x2_i, out_i in zip(self.elem, x2, out):
                    r = getattr(x1_i.ufuncs, name)(x2_i, out=out_i, **kwargs)
                    res.append(r)
                out_space = ProductSpace(*[r.space for r in res])
                return out_space.element(res)

        elif n_out == 2:
            def wrapper(self, x2, out1=None, out2=None, **kwargs):
                from odl.space.pspace import ProductSpace
                if out1 is None:
                    out1 = [None] * len(self.elem.space)
                if out2 is None:
                    out2 = [None] * len(self.elem.space)

                res1, res2 = [], []
                for x1_i, x2_i, out1_i, out2_i in zip(self.elem, x2, out1,
                                                      out2):
                    r1, r2 = getattr(x1_i.ufuncs, name)(x2_i, out1=out1_i,
                                                        out2=out2_i,
                                                        **kwargs)
                    res1.append(r1)
                    res2.append(r2)
                out_space_1 = ProductSpace(*[r.space for r in res1])
                out_space_2 = ProductSpace(*[r.space for r in res2])
                return out_space_1.element(res1), out_space_2.element(res2)

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    wrapper.__name__ = wrapper.__qualname__ = name
    wrapper.__doc__ = doc
    return wrapper


class ProductSpaceUfuncs(object):

    """Ufuncs for `ProductSpaceElement` objects.

    Internal object, should not be created except in `ProductSpaceElement`.
    """
    def __init__(self, elem):
        """Create ufunc wrapper for ``elem``."""
        self.elem = elem

    def sum(self):
        """Return the sum of ``self``.

        See Also
        --------
        numpy.sum
        prod
        """
        results = [x.ufuncs.sum() for x in self.elem]
        return np.sum(results)

    def prod(self):
        """Return the product of ``self``.

        See Also
        --------
        numpy.prod
        sum
        """
        results = [x.ufuncs.prod() for x in self.elem]
        return np.prod(results)

    def min(self):
        """Return the minimum of ``self``.

        See Also
        --------
        numpy.amin
        max
        """
        results = [x.ufuncs.min() for x in self.elem]
        return np.min(results)

    def max(self):
        """Return the maximum of ``self``.

        See Also
        --------
        numpy.amax
        min
        """
        results = [x.ufuncs.max() for x in self.elem]
        return np.max(results)


# Add ufunc methods to ufunc class
for name, n_in, n_out, doc in UFUNCS:
    method = wrap_ufunc_productspace(name, n_in, n_out, doc)
    setattr(ProductSpaceUfuncs, name, method)
