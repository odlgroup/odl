﻿# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utility functions for space implementations."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

import numpy as np

from odl.set import RealNumbers, ComplexNumbers
from odl.space.entry_points import tensor_space_impl
from odl.util import is_scalar_dtype


__all__ = ('vector', 'tensor_space', 'cn', 'rn')


def vector(array, dtype=None, order='C', impl='numpy'):
    """Create a vector from an array-like object.

    Parameters
    ----------
    array : `array-like`
        Array from which to create the vector. Scalars become
        one-dimensional vectors.
    dtype : optional
        Set the data type of the vector manually with this option.
        By default, the space type is inferred from the input data.
    order : {'C', 'F'}, optional
        Axis ordering of the data storage.
    impl : str, optional
        Impmlementation back-end for the space. See
        `odl.space.entry_points.tensor_space_impl_names` for available
        options.

    Returns
    -------
    vector : `GeneralizedTensor`
        Vector created from the input array. Its concrete type depends
        on the provided arguments.

    Notes
    -----
    This is a convenience function and not intended for use in
    speed-critical algorithms.

    Examples
    --------
    >>> odl.vector([[1, 2, 3],
    ...             [4, 5, 6]])  # No automatic cast to float
    tensor_space((2, 3), 'int').element(
    [[1, 2, 3],
     [4, 5, 6]]
    )
    >>> odl.vector([[1, 2, 3],
    ...             [4, 5, 6]], dtype=float)
    rn((2, 3)).element(
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0]]
    )
    >>> odl.vector([[1, 2 - 1j, 3],
    ...             [4, 5, 6 + 2j]])
    cn((2, 3)).element(
    [[(1+0j), (2-1j), (3+0j)],
     [(4+0j), (5+0j), (6+2j)]]
    )

    Non-scalar types are also supported:

    >>> odl.vector([[True, True, False],
    ...             [False, False, True]])
    tensor_set((2, 3), 'bool').element(
    [[True, True, False],
     [False, False, True]]
    )
    """
    # Sanitize input
    arr = np.array(array, copy=False, order=order, ndmin=1)

    if arr.dtype is object:
        raise ValueError('invalid input data resulting in `dtype==object`')

    # Set dtype
    if dtype is not None:
        space_dtype = dtype
    else:
        space_dtype = arr.dtype

    # Select implementation
    if space_dtype is None or is_scalar_dtype(space_dtype):
        space_constr = tensor_space
    else:
        space_constr = tensor_set

    return space_constr(arr.shape, dtype=space_dtype, order=order,
                        impl=impl).element(arr)


def tensor_space(shape, dtype=None, order='C', impl='numpy', **kwargs):
    """Return a tensor space with arbitrary scalar data type.

    Parameters
    ----------
    shape : positive int or sequence of positive ints
        Number of entries per axis for each element.
    dtype : optional
        Data type of each element. Can be provided in any way the
        `numpy.dtype` function understands, e.g. as built-in type or
        as a string.
        For ``None``, the `TensorSpace.default_dtype` of the
        created space is used.
    order : {'C', 'F'}, optional
        Axis ordering of the data storage.
    impl : string, optional
        Impmlementation back-end for the space. See
        `odl.space.entry_points.tensor_space_impl_names` for available
        options.
    kwargs :
        Extra keyword arguments passed to the space constructor.

    Returns
    -------
    tspace : `TensorSpace`

    Examples
    --------
    Space of 3-tuples with ``int64`` entries (although not strictly a
    vector space):

    >>> odl.tensor_space(3, dtype='int64')
    tensor_space(3, 'int')

    2x3 tensors with ``int64`` entries (although not strictly a
    vector space):

    >>> odl.tensor_space((2, 3), dtype='int64')
    tensor_space((2, 3), 'int')

    The default data type depends on the implementation. For
    ``impl='numpy'``, it is ``'float64'``:

    >>> ts = odl.tensor_space((2, 3))
    >>> ts
    rn((2, 3))
    >>> ts.dtype
    dtype('float64')

    One-dimensional spaces have special constructors:

    >>> odl.tensor_space((3,), dtype='int64')
    tensor_space(3, 'int')

    See also
    --------
    tensor_set : Set of tensors with arbitrary data type.
    """
    tspace_cls = tensor_space_impl(impl)

    if dtype is None:
        dtype = tspace_cls.default_dtype()

    return tspace_cls(shape, dtype, order, **kwargs)


def cn(shape, dtype=None, order='C', impl='numpy', **kwargs):
    """Return a space of complex tensors.

    Parameters
    ----------
    shape : positive int or sequence of positive ints
        Number of entries per axis of an element in the created space.
    dtype : optional
        Data type of each element. Can be provided in any way the
        `numpy.dtype` function understands, e.g. as built-in type or
        as a string. Only complex floating-point data types are allowed.
        For ``None``, the `TensorSpace.default_dtype` of the
        created space is used in the form
        ``default_dtype(ComplexNumbers())``.
    order : {'C', 'F'}, optional
        Axis ordering of the data storage.
    impl : str, optional
        Impmlementation back-end for the space. See
        `odl.space.entry_points.tensor_space_impl_names` for available
        options.
    kwargs :
        Extra keyword arguments passed to the space constructor.

    Returns
    -------
    cn : `TensorSpace`

    Examples
    --------
    Space of complex 3-tuples with ``complex64`` entries:

    >>> odl.cn(3, dtype='complex64')
    cn(3, 'complex64')

    Complex 2x3 tensors with ``complex64`` entries:

    >>> odl.cn((2, 3), dtype='complex64')
    cn((2, 3), 'complex64')

    The default data type depends on the implementation. For
    ``impl='numpy'``, it is ``'complex128'``:

    >>> space = odl.cn((2, 3))
    >>> space
    cn((2, 3))
    >>> space.dtype
    dtype('complex128')

    See also
    --------
    tensor_space : Space of tensors with arbitrary scalar data type.
    rn : Real tensor spaces.
    """
    cn_cls = tensor_space_impl(impl)

    if dtype is None:
        dtype = cn_cls.default_dtype(ComplexNumbers())

    cn = cn_cls(shape, dtype, order, **kwargs)
    if not cn.is_complex:
        raise TypeError('data type {!r} not a complex floating-point type.'
                        ''.format(dtype))
    return cn


def rn(shape, dtype=None, order='C', impl='numpy', **kwargs):
    """Return a space of real tensors.

    Parameters
    ----------
    shape : positive int or sequence of positive ints
        Number of entries per axis of an element in the created space.
    dtype : optional
        Data type of each element. Can be provided in any way the
        `numpy.dtype` function understands, e.g. as built-in type or
        as a string. Only real floating-point data types are allowed.
        For ``None``, the `TensorSpace.default_dtype` of the
        created space is used in the form
        ``default_dtype(RealNumbers())``.
    order : {'C', 'F'}, optional
        Axis ordering of the data storage.
    impl : string, optional
        Impmlementation back-end for the space. See
        `odl.space.entry_points.tensor_space_impl_names` for available
        options.
    kwargs :
        Extra keyword arguments passed to the space constructor.

    Returns
    -------
    real_space : `TensorSpace`

    Examples
    --------
    Space of real 3-tuples with ``float32`` entries:

    >>> odl.rn(3, dtype='float32')
    rn(3, 'float32')

    Real 2x3 tensors with ``float32`` entries:

    >>> odl.rn((2, 3), dtype='float32')
    rn((2, 3), 'float32')

    The default data type depends on the implementation. For
    ``impl='numpy'``, it is ``'float64'``:

    >>> ts = odl.rn((2, 3))
    >>> ts
    rn((2, 3))
    >>> ts.dtype
    dtype('float64')

    One-dimensional spaces have special constructors:

    >>> odl.rn((3,))
    rn(3)

    See also
    --------
    tensor_space : Space of tensors with arbitrary scalar data type.
    cn : Complex tensor space.
    """
    rn_cls = tensor_space_impl(impl)

    if dtype is None:
        dtype = rn_cls.default_dtype(RealNumbers())

    rn = rn_cls(shape, dtype, order, **kwargs)
    if not rn.is_real:
        raise TypeError('data type {!r} not a real floating-point type.'
                        ''.format(dtype))
    return rn


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
