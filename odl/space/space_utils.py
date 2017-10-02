# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utility functions for space implementations."""

from __future__ import print_function, division, absolute_import
import numpy as np

from odl.set import RealNumbers, ComplexNumbers
from odl.space.entry_points import fn_impl

__all__ = ('vector', 'fn', 'cn', 'rn')


def vector(array, dtype=None, impl='numpy'):
    """Create an n-tuples type vector from an array.

    Parameters
    ----------
    array : `array-like`
        Array from which to create the vector. Scalars become
        one-dimensional vectors.
    dtype : optional
        Set the data type of the vector manually with this option.
        By default, the space type is inferred from the input data.
    impl : string, optional
        The backend to use. See `odl.space.entry_points.fn_impl_names`
        for available options.

    Returns
    -------
    vec : `FnBaseVector`
        Vector created from the input array. Its concrete type depends
        on the provided arguments.

    Notes
    -----
    This is a convenience function and not intended for use in
    speed-critical algorithms.

    Examples
    --------
    >>> vector([1, 2, 3])  # No automatic cast to float
    fn(3, 'int').element([1, 2, 3])
    >>> vector([1, 2, 3], dtype=float)
    rn(3).element([ 1.,  2.,  3.])
    >>> vector([1 + 1j, 2, 3 - 2j])
    cn(3).element([ 1.+1.j,  2.+0.j,  3.-2.j])

    Non-scalar types are also supported:

    >>> vector([True, False])
    fn(2, 'bool').element([ True, False])

    Scalars become a one-element vector:

    >>> vector(0.0)
    rn(1).element([ 0.])
    """
    # Sanitize input
    arr = np.array(array, copy=False, ndmin=1)

    # Validate input
    if arr.ndim > 1:
        raise ValueError('array has {} dimensions, expected 1'
                         ''.format(arr.ndim))

    # Set dtype
    if dtype is not None:
        space_dtype = dtype
    else:
        space_dtype = arr.dtype

    return fn(len(arr), dtype=space_dtype, impl=impl).element(arr)


def fn(size, dtype=None, impl='numpy', **kwargs):
    """Return the space ``F^n`` for arbitrary field ``F``.

    Parameters
    ----------
    size : positive int
        The number of dimensions of the space
    dtype : `object`, optional
        The data type of the storage array. Can be provided in any
        way the `numpy.dtype` function understands, most notably
        as built-in type, as one of NumPy's internal datatype
        objects or as string.

        Default: default of the implementation given by calling
        ``default_dtype()`` on the `FnBase` implementation.
    impl : string, optional
        The backend to use. See `odl.space.entry_points.fn_impl_names` for
        available options.
    kwargs :
        Extra keyword arguments to pass to the implmentation.

    Returns
    -------
    fn : `FnBase`

    See Also
    --------
    rn, cn : Constructors for real and complex spaces
    """
    fn_cls = fn_impl(impl)

    if dtype is None:
        dtype = fn_impl.default_dtype()

    return fn_cls(size, dtype=dtype, **kwargs)


def cn(size, dtype=None, impl='numpy', **kwargs):
    """Return the complex vector space ``C^n``.

    Parameters
    ----------
    size : positive int
        The number of dimensions of the space
    dtype : `object`, optional
        The data type of the storage array. Can be provided in any
        way the `numpy.dtype` function understands, most notably
        as built-in type, as one of NumPy's internal datatype
        objects or as string.

        Only complex floating-point data types are allowed.

        Default: default of the implementation given by calling
        ``default_dtype(ComplexNumbers())`` on the `FnBase` implementation.
    impl : string, optional
        The backend to use. See `odl.space.entry_points.fn_impl_names` for
        available options.
    kwargs :
        Extra keyword arguments to pass to the implmentation.

    Returns
    -------
    cn : `FnBase`

    See Also
    --------
    fn : n-tuples over a field with arbitrary scalar data type.
    """
    cn_cls = fn_impl(impl)

    if dtype is None:
        dtype = cn_cls.default_dtype(ComplexNumbers())

    cn = cn_cls(size, dtype, **kwargs)

    if not cn.is_complex:
        raise ValueError('data type {!r} not a complex floating-point type.'
                         ''.format(dtype))
    return cn


def rn(size, dtype=None, impl='numpy', **kwargs):
    """Return the real vector space ``R^n``.

    Parameters
    ----------
    size : positive int
        The number of dimensions of the space
    dtype : `object`, optional
        The data type of the storage array. Can be provided in any
        way the `numpy.dtype` function understands, most notably
        as built-in type, as one of NumPy's internal datatype
        objects or as string.

        Only real floating-point data types are allowed.
        Default: default of the implementation given by calling
        ``default_dtype(RealNumbers())`` on the `FnBase` implementation.
    impl : string, optional
        The backend to use. See `odl.space.entry_points.fn_impl_names` for
        available options.
    kwargs :
        Extra keyword arguments to pass to the implmentation.

    Returns
    -------
    rn : `FnBase`

    See Also
    --------
    fn : n-tuples over a field with arbitrary scalar data type.
    """
    rn_impl = fn_impl(impl)

    if dtype is None:
        dtype = rn_impl.default_dtype(RealNumbers())

    rn = rn_impl(size, dtype, **kwargs)

    if not rn.is_real:
        raise ValueError('data type {!r} not a real floating-point type.'
                         ''.format(dtype))
    return rn


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
