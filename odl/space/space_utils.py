# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utility functions for space implementations."""

from __future__ import print_function, division, absolute_import

from odl.set import RealNumbers, ComplexNumbers
from odl.space.entry_points import tensor_space_impl


__all__ = ('tensor_space', 'cn', 'rn')


def tensor_space(shape, dtype=None, impl='numpy', **kwargs):
    """Return a tensor space with arbitrary scalar data type.

    Parameters
    ----------
    shape : positive int or sequence of positive ints
        Number of entries per axis for elements in this space. A
        single integer results in a space with 1 axis.
    dtype : optional
        Data type of each element. Can be provided in any way the
        `numpy.dtype` function understands, e.g. as built-in type or
        as a string.
        For ``None``, the `TensorSpace.default_dtype` of the
        created space is used.
    impl : str, optional
        Impmlementation back-end for the space. See
        `odl.space.entry_points.tensor_space_impl_names` for available
        options.
    kwargs :
        Extra keyword arguments passed to the space constructor.

    Returns
    -------
    space : `TensorSpace`

    Examples
    --------
    Space of 3-tuples with ``uint64`` entries (although not strictly a
    vector space):

    >>> odl.tensor_space(3, dtype='uint64')
    tensor_space(3, dtype='uint64')

    2x3 tensors with same data type:

    >>> odl.tensor_space((2, 3), dtype='uint64')
    tensor_space((2, 3), dtype='uint64')

    The default data type depends on the implementation. For
    ``impl='numpy'``, it is ``'float64'``:

    >>> ts = odl.tensor_space((2, 3))
    >>> ts
    rn((2, 3))
    >>> ts.dtype
    dtype('float64')

    See Also
    --------
    rn, cn : Constructors for real and complex spaces
    """
    tspace_cls = tensor_space_impl(impl)

    if dtype is None:
        dtype = tspace_cls.default_dtype()

    # Use args by keyword since the constructor may take other arguments
    # by position
    return tspace_cls(shape=shape, dtype=dtype, **kwargs)


def cn(shape, dtype=None, impl='numpy', **kwargs):
    """Return a space of complex tensors.

    Parameters
    ----------
    shape : positive int or sequence of positive ints
        Number of entries per axis for elements in this space. A
        single integer results in a space with 1 axis.
    dtype : optional
        Data type of each element. Can be provided in any way the
        `numpy.dtype` function understands, e.g. as built-in type or
        as a string. Only complex floating-point data types are allowed.
        For ``None``, the `TensorSpace.default_dtype` of the
        created space is used in the form
        ``default_dtype(ComplexNumbers())``.
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
    cn(3, dtype='complex64')

    Complex 2x3 tensors with ``complex64`` entries:

    >>> odl.cn((2, 3), dtype='complex64')
    cn((2, 3), dtype='complex64')

    The default data type depends on the implementation. For
    ``impl='numpy'``, it is ``'complex128'``:

    >>> space = odl.cn((2, 3))
    >>> space
    cn((2, 3))
    >>> space.dtype
    dtype('complex128')

    See Also
    --------
    tensor_space : Space of tensors with arbitrary scalar data type.
    rn : Real tensor space.
    """
    cn_cls = tensor_space_impl(impl)

    if dtype is None:
        dtype = cn_cls.default_dtype(ComplexNumbers())

    # Use args by keyword since the constructor may take other arguments
    # by position
    cn = cn_cls(shape=shape, dtype=dtype, **kwargs)
    if not cn.is_complex:
        raise ValueError('data type {!r} not a complex floating-point type.'
                         ''.format(dtype))
    return cn


def rn(shape, dtype=None, impl='numpy', **kwargs):
    """Return a space of real tensors.

    Parameters
    ----------
    shape : positive int or sequence of positive ints
        Number of entries per axis for elements in this space. A
        single integer results in a space with 1 axis.
    dtype : optional
        Data type of each element. Can be provided in any way the
        `numpy.dtype` function understands, e.g. as built-in type or
        as a string. Only real floating-point data types are allowed.
        For ``None``, the `TensorSpace.default_dtype` of the
        created space is used in the form
        ``default_dtype(RealNumbers())``.
    impl : str, optional
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
    rn(3, dtype='float32')

    Real 2x3 tensors with ``float32`` entries:

    >>> odl.rn((2, 3), dtype='float32')
    rn((2, 3), dtype='float32')

    The default data type depends on the implementation. For
    ``impl='numpy'``, it is ``'float64'``:

    >>> ts = odl.rn((2, 3))
    >>> ts
    rn((2, 3))
    >>> ts.dtype
    dtype('float64')

    See Also
    --------
    tensor_space : Space of tensors with arbitrary scalar data type.
    cn : Complex tensor space.
    """
    rn_cls = tensor_space_impl(impl)

    if dtype is None:
        dtype = rn_cls.default_dtype(RealNumbers())

    # Use args by keyword since the constructor may take other arguments
    # by position
    rn = rn_cls(shape=shape, dtype=dtype, **kwargs)
    if not rn.is_real:
        raise ValueError('data type {!r} not a real floating-point type.'
                         ''.format(dtype))
    return rn


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
