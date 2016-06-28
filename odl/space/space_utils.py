# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utility functions for space implementations."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np

from odl.set import RealNumbers, ComplexNumbers
from odl.space.entry_points import TENSOR_SET_IMPLS, TENSOR_SPACE_IMPLS
from odl.util import is_scalar_dtype


__all__ = ('vector', 'tensor_set', 'tensor_space', 'cn', 'rn')


def vector(array, dtype=None, impl='numpy'):
    """Create an n-tuples type vector from an array-like object.

    Parameters
    ----------
    array : `array-like`
        Array from which to create the vector. Scalars become
        one-dimensional vectors.
    dtype : optional
        Set the data type of the vector manually with this option.
        By default, the space type is inferred from the input data.
    impl : string
        The backend to use. See `odl.space.entry_points.TENSOR_SET_IMPLS` and
        `odl.space.entry_points.TENSOR_SPACE_IMPLS` for available options.

    Returns
    -------
    vec : `GeneralizedTensor`
        Vector created from the input array. Its concrete type depends
        on the provided arguments.

    Notes
    -----
    This is a convenience function and not intended for use in
    speed-critical algorithms.

    Examples
    --------
    >>> vector([1, 2, 3])  # No automatic cast to float
    tensor_space(3, 'int').element([1, 2, 3])
    >>> vector([1, 2, 3], dtype=float)
    rn(3).element([1.0, 2.0, 3.0])
    >>> vector([1 + 1j, 2, 3 - 2j])
    cn(3).element([(1+1j), (2+0j), (3-2j)])

    Non-scalar types are also supported:

    >>> vector([True, False])
    tensor_set(2, 'bool').element([True, False])

    Scalars become a one-element vector:

    >>> vector(0.0)
    rn(1).element([0.0])
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

    # Select implementation
    if space_dtype is None or is_scalar_dtype(space_dtype):
        space_constr = tensor_space
    else:
        space_constr = tensor_set

    return space_constr(len(arr), dtype=space_dtype, impl=impl).element(arr)


def tensor_set(shape, dtype, order='C', impl='numpy', **kwargs):
    """Return a tensor set with arbitrary data type.

    Parameters
    ----------
    shape : sequence of positive int
        Number of entries per axis for each element.
    dtype :
        Data type of each element. Can be provided in any way the
        `numpy.dtype` function understands, e.g. as built-in type or
        as a string.
    order : {'C', 'F'}, optional
        Axis ordering of the data storage.
    impl : str, optional
        The backend to use. See `TENSOR_SET_IMPLS` for available
        options.
    kwargs :
        Extra keyword arguments passed to the set constructor.

    Returns
    -------
    tset : `TensorSet`

    See also
    --------
    tensor_space : space of tensors with arbitrary scalar data type.
    """
    return TENSOR_SET_IMPLS[impl](shape, dtype, order, **kwargs)


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
    impl : str, optional
        The backend to use. See `TENSOR_SPACE_IMPLS` for available
        options.
    kwargs :
        Extra keyword arguments passed to the space constructor.

    Returns
    -------
    tspace : `TensorSpace`

    See also
    --------
    tensor_set : set of tensors with arbitrary data type.
    """
    tspace_constr = TENSOR_SPACE_IMPLS[impl]

    if dtype is None:
        dtype = tspace_constr.default_dtype()

    return tspace_constr(shape, dtype, **kwargs)


def cn(shape, dtype=None, order='C', impl='numpy', **kwargs):
    """Return a space of complex tensors.

    Parameters
    ----------
    shape : positive int or sequence of positive ints
        Number of entries per axis for each element.
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
        The backend to use. See `TENSOR_SPACE_IMPLS` for available
        options.
    kwargs :
        Extra keyword arguments passed to the space constructor.

    Returns
    -------
    complex_tspace : `TensorSpace`

    See also
    --------
    tensor_space : space of tensors with arbitrary scalar data type.
    """
    cn_constr = TENSOR_SPACE_IMPLS[impl]

    if dtype is None:
        dtype = cn_constr.default_dtype(ComplexNumbers())

    cn = cn_constr(shape, dtype, **kwargs)
    if not cn.is_complex_space:
        raise ValueError('data type {!r} not a complex floating-point type'
                         ''.format(dtype))
    return cn


def rn(shape, dtype=None, order='C', impl='numpy', **kwargs):
    """Return a space of real tensors.

    Parameters
    ----------
    shape : positive int or sequence of positive ints
        Number of entries per axis for each element.
    dtype : optional
        Data type of each element. Can be provided in any way the
        `numpy.dtype` function understands, e.g. as built-in type or
        as a string. Only real floating-point data types are allowed.
        For ``None``, the `TensorSpace.default_dtype` of the
        created space is used in the form
        ``default_dtype(RealNumbers())``.
    order : {'C', 'F'}, optional
        Axis ordering of the data storage.
    impl : str, optional
        The backend to use. See `TENSOR_SPACE_IMPLS` for available
        options.
    kwargs :
        Extra keyword arguments passed to the space constructor.

    Returns
    -------
    real_tspace : `TensorSpace`

    See also
    --------
    tensor_space : space of tensors with arbitrary scalar data type.
    """
    rn_constr = TENSOR_SPACE_IMPLS[impl]

    if dtype is None:
        dtype = rn_constr.default_dtype(RealNumbers())

    rn = rn_constr(shape, dtype, **kwargs)
    if not rn.is_real_space:
        raise ValueError('data type {!r} not a real floating-point type'
                         ''.format(dtype))
    return rn


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
