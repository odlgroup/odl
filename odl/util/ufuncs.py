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
            def wrapper(self, out1=None, out2=None, **kwargs):
                return self.elem.__array_ufunc__(
                    ufunc, '__call__', self.elem, out=(out1, out2), **kwargs)

        else:
            raise NotImplementedError

    elif n_in == 2:
        if n_out == 1:
            def wrapper(self, x2, out=None, **kwargs):
                if out is None or isinstance(out, (type(self.elem),
                                                   type(self.elem.data))):
                    out = (out,)

                return self.elem.__array_ufunc__(
                    ufunc, '__call__', self.elem, x2, out=out, **kwargs)

        elif n_out == 2:
            def wrapper(self, x2, out1=None, out2=None, **kwargs):
                return self.elem.__array_ufunc__(
                    ufunc, '__call__', self.elem, x2, out=(out1, out2),
                    **kwargs)

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

    def cumsum(self, axis=None, dtype=None, out=None):
        """Return the cumulative sum of ``self``.

        See Also
        --------
        numpy.cumsum
        cumprod
        """
        return self.elem.__array_ufunc__(
            np.add, 'accumulate', self.elem,
            axis=axis, dtype=dtype, out=(out,))

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

    def cumprod(self, axis=None, dtype=None, out=None):
        """Return the cumulative product of ``self``.

        See Also
        --------
        numpy.cumprod
        cumsum
        """
        return self.elem.__array_ufunc__(
            np.multiply, 'accumulate', self.elem,
            axis=axis, dtype=dtype, out=(out,))

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


def _add_leading_dims(x2, pspace):
    """Add leading dimensions to an input of a binary product space ufunc.

    Parameters
    ----------
    x2
        Input to a binary ufunc.
    pspace : `ProductSpace`
        Spac on which the ufunc is evaluated.

    Returns
    -------
    x2_extra_dims
        Variant of ``x2`` with extra leading dimensions appropriate for
        ``pspace``.

    Examples
    --------
    Product space elements, arrays and nested sequences can be rewrapped
    such that they have leading dimensions of size 1 for broadcasting:

    >>> pspace = odl.rn(2) ** (3, 4)
    >>> reshaped = _add_leading_dims([0, 0], pspace)
    >>> np.shape([0, 0])
    (2,)
    >>> reshaped
    [[[0, 0]]]
    >>> np.shape(reshaped)
    (1, 1, 2)
    >>> reshaped = _add_leading_dims(np.zeros(2), pspace)
    >>> reshaped
    array([[[ 0.,  0.]]])
    >>> reshaped.shape
    (1, 1, 2)
    >>> reshaped = _add_leading_dims(odl.rn(2).zero(), pspace)
    >>> reshaped  # returning array for simplicity
    array([[[ 0.,  0.]]])
    >>> reshaped.shape
    (1, 1, 2)
    >>> reshaped = _add_leading_dims((odl.rn(2) ** 4).zero(), pspace)
    >>> reshaped  # Broadcasting along outer dimension of size 3
    ProductSpace(ProductSpace(rn(2), 4), 1).element([
        [
            [ 0.,  0.],
            [ 0.,  0.],
            [ 0.,  0.],
            [ 0.,  0.]
        ]
    ])
    """
    from odl.space.pspace import ProductSpace, ProductSpaceElement
    from odl.space.base_tensors import Tensor

    # Workaround for `shape` not using the base space shape of a
    # power space
    # TODO: remove when fixed, see
    # https://github.com/odlgroup/odl/pull/1152
    num_levels_pspace = 0
    tmp_pspace = pspace
    while True:
        if isinstance(tmp_pspace, ProductSpace):
            tmp_pspace = tmp_pspace[0]
            num_levels_pspace += 1
        else:
            base_ndim_pspace = len(getattr(tmp_pspace, 'shape', ()))
            break

    pspace_ndim = num_levels_pspace + base_ndim_pspace

    if isinstance(x2, ProductSpaceElement):
        # Workaround for `shape` not using the base space shape of a
        # power space element
        # TODO: remove when fixed, see
        # https://github.com/odlgroup/odl/pull/1152
        num_levels_x2 = 0
        tmp_x2 = x2
        while True:
            if isinstance(tmp_x2, ProductSpaceElement):
                tmp_x2 = tmp_x2[0]
                num_levels_x2 += 1
            else:
                base_ndim_x2 = len(getattr(tmp_x2, 'shape', ()))
                break

        x2_ndim = num_levels_x2 + base_ndim_x2

        # TODO: replace by `pspace.ndim` and `x2.ndim` when fixed, see
        # https://github.com/odlgroup/odl/pull/1152
        for _ in range(pspace_ndim - x2_ndim):
            x2 = (x2.space ** 1).element([x2])

        return x2

    if isinstance(x2, Tensor):
        # Work with array directly
        x2 = x2.data

    if hasattr(x2, 'shape'):
        # Some type of array
        x2_ndim = len(x2.shape)

        slc = (None,) * (pspace_ndim - x2_ndim) + (slice(None),) * x2_ndim
        x2 = x2[slc]

        # Downstream code will raise in case of bad shape
        return x2

    else:
        # Array-like
        x2_ndim = np.ndim(x2)

        for _ in range(pspace_ndim - x2_ndim):
            x2 = [x2]

        # Downstream code will raise in case of bad shape
        return x2


def wrap_ufunc_productspace(name, n_in, n_out, doc):
    """Return ufunc wrapper for `ProductSpaceUfuncs`."""
    if n_in == 1:
        if n_out == 1:
            def wrapper(self, out=None, **kwargs):
                from odl.space.pspace import ProductSpace, ProductSpaceElement
                if out is None:
                    out_seq = [None] * len(self.elem.space)
                else:
                    assert isinstance(out, ProductSpaceElement)
                    out_seq = out

                res = []
                for xi, out_i in zip(self.elem, out_seq):
                    ufunc = getattr(xi.ufuncs, name)
                    r = ufunc(out=out_i, **kwargs)
                    res.append(r)

                if out is None:
                    out_space = ProductSpace(*[r.space for r in res])
                    out = out_space.element(res)

                return out

        elif n_out == 2:
            def wrapper(self, out1=None, out2=None, **kwargs):
                from odl.space.pspace import ProductSpace, ProductSpaceElement
                if out1 is None:
                    out1_seq = [None] * len(self.elem.space)
                else:
                    assert isinstance(out1, ProductSpaceElement)
                    out1_seq = out1
                if out2 is None:
                    out2_seq = [None] * len(self.elem.space)
                else:
                    assert isinstance(out2, ProductSpaceElement)
                    out2_seq = out2

                res1, res2 = [], []
                for xi, out1_i, out2_i in zip(self.elem, out1_seq, out2_seq):
                    ufunc = getattr(xi.ufuncs, name)
                    r1, r2 = ufunc(out1=out1_i, out2=out2_i, **kwargs)
                    res1.append(r1)
                    res2.append(r2)

                if out1 is None:
                    out_space_1 = ProductSpace(*[r.space for r in res1])
                    out1 = out_space_1.element(res1)
                if out2 is None:
                    out_space_2 = ProductSpace(*[r.space for r in res2])
                    out2 = out_space_2.element(res2)

                return out1, out2

        else:
            raise NotImplementedError

    elif n_in == 2:
        if n_out == 1:
            def wrapper(self, x2, out=None, **kwargs):
                from odl.space.pspace import ProductSpace, ProductSpaceElement
                if out is None:
                    out_seq = [None] * len(self.elem.space)
                else:
                    assert isinstance(out, ProductSpaceElement)
                    out_seq = out

                # Implement broadcasting
                x2 = _add_leading_dims(x2, self.elem.space)
                if len(x2) == 1 and len(self.elem.space) != 1:
                    x2 = [x2[0]] * len(self.elem.space)

                res = []
                for x1_i, x2_i, out_i in zip(self.elem, x2, out_seq):
                    ufunc = getattr(x1_i.ufuncs, name)
                    r = ufunc(x2_i, out=out_i, **kwargs)
                    res.append(r)

                if out is None:
                    out_space = ProductSpace(*[r.space for r in res])
                    out = out_space.element(res)

                return out

        elif n_out == 2:
            def wrapper(self, x2, out1=None, out2=None, **kwargs):
                from odl.space.pspace import ProductSpace, ProductSpaceElement
                if out1 is None:
                    out1_seq = [None] * len(self.elem.space)
                else:
                    assert isinstance(out1, ProductSpaceElement)
                    out1_seq = out1
                if out2 is None:
                    out2_seq = [None] * len(self.elem.space)
                else:
                    out2_seq = out2

                # Implement broadcasting
                x2 = _add_leading_dims(x2, self.elem.space)
                if len(x2) == 1 and len(self.elem.space) != 1:
                    x2 = [x2[0]] * len(self.elem.space)

                res1, res2 = [], []
                for x1_i, x2_i, out1_i, out2_i in zip(self.elem, x2,
                                                      out1_seq, out2_seq):
                    ufunc = getattr(x1_i.ufuncs, name)
                    r1, r2 = ufunc(x2_i, out1=out1_i, out2=out2_i, **kwargs)
                    res1.append(r1)
                    res2.append(r2)

                if out1 is None:
                    out_space_1 = ProductSpace(*[r.space for r in res1])
                    out1 = out_space_1.element(res1)
                if out2 is None:
                    out_space_2 = ProductSpace(*[r.space for r in res2])
                    out2 = out_space_2.element(res2)

                return out1, out2

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
