# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utility functions for space implementations."""

from __future__ import print_function, division, absolute_import
import numpy as np

from odl.util.npy_compat import AVOID_UNNECESSARY_COPY
from odl.core.array_API_support import lookup_array_backend

from odl.space.base_tensors import default_dtype

from odl.util.dtype_utils import is_available_dtype, is_complex_dtype, is_floating_dtype
from odl.space.entry_points import tensor_space_impl, tensor_space_impl_names

__all__ = ('vector', 'tensor_space', 'cn', 'rn')


def vector(array, dtype=None, impl='numpy', device = 'cpu'):
    """Create a vector from an array-like object.

    Parameters
    ----------
    array : `array-like`
        Array from which to create the vector. Scalars become
        one-dimensional vectors.
    dtype : optional
        Set the data type of the vector manually with this option.
        By default, the space type is inferred from the input data.
    impl : str, optional
        Impmlementation back-end for the space. See
        `odl.space.entry_points.tensor_space_impl_names` for available
        options.

    Returns
    -------
    vector : `Tensor`
        Vector created from the input array. Its concrete type depends
        on the provided arguments.

    Notes
    -----
    This is a convenience function and not intended for use in
    speed-critical algorithms.

    Examples
    --------
    Create one-dimensional vectors:

    >>> odl.vector([1, 2, 3])  # No automatic cast to float
    tensor_space(3, 'int64').element([1, 2, 3])
    >>> odl.vector([1, 2, 3], dtype=float)
    rn(3).element([ 1.,  2.,  3.])
    >>> odl.vector([1, 2 - 1j, 3])
    cn(3).element([ 1.+0.j,  2.-1.j,  3.+0.j])

    Non-scalar types are also supported:

    >>> odl.vector([True, True, False])
    tensor_space(3, 'bool').element([ True,  True, False])

    The function also supports multi-dimensional input:

    >>> odl.vector([[1, 2, 3],
    ...             [4, 5, 6]])
    tensor_space((2, 3), 'int64').element(
        [[1, 2, 3],
         [4, 5, 6]]
    )
    """
    backend = lookup_array_backend(impl)
    # Sanitize input
    # I don't understand was a ndim prepended to the array dimension
    arr = backend.array_constructor(array, copy=AVOID_UNNECESSARY_COPY, device = device)

    if arr.dtype is object:
        raise ValueError('invalid input data resulting in `dtype==object`')

    # Set dtype
    if dtype is not None:
        space_dtype = dtype
    else:
        space_dtype = arr.dtype

    space = tensor_space(arr.shape, dtype=space_dtype, impl=impl, device=device)
    return space.element(arr)


def tensor_space(shape, dtype='float64', impl='numpy', device = 'cpu', **kwargs):
    """Return a tensor space with arbitrary scalar data type.

    Parameters
    ----------
    shape : positive int or sequence of positive ints
        Number of entries per axis for elements in this space. A
        single integer results in a space with 1 axis.
    dtype (str) : optional
        Data type of each element. Defaults to float64
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
    tensor_space(3, 'uint64')

    2x3 tensors with same data type:

    >>> odl.tensor_space((2, 3), dtype='uint64')
    tensor_space((2, 3), 'uint64')

    The default data type is ``'float64'``. How that is represented as a dtype-object
    depends on the backend.

    >>> ts = odl.tensor_space((2, 3))
    >>> ts
    rn((2, 3))
    >>> ts.dtype
    dtype('float64')

    See Also
    --------
    rn, cn : Constructors for real and complex spaces
    """
    # Check the dtype argument
    is_available_dtype(dtype)
    # Check the impl argument
    assert (
        impl in tensor_space_impl_names()
    ), f"The only supported impls are {tensor_space_impl_names()}, but {impl} was provided"

    # Use args by keyword since the constructor may take other arguments
    # by position
    return tensor_space_impl(impl)(shape=shape, dtype=dtype, device=device, **kwargs)


def cn(shape, dtype='complex128', impl='numpy', device='cpu', **kwargs):
    """Return a space of complex tensors.

    Parameters
    ----------
    shape : positive int or sequence of positive ints
        Number of entries per axis for elements in this space. A
        single integer results in a space with 1 axis.
    dtype (str) : optional
        Data type of each element. Must be provided as a string or Python complex type.
        Defaults to complex128
    impl (str) : str, optional
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

    See Also
    --------
    tensor_space : Space of tensors with arbitrary scalar data type.
    rn : Real tensor space.
    """
    is_complex_dtype(dtype)
    return tensor_space(shape, dtype=dtype, impl=impl, device=device, **kwargs)


def rn(shape, dtype=None, impl='numpy', device ='cpu', **kwargs):
    """Return a space of real tensors.

    Parameters
    ----------
    shape : positive int or sequence of positive ints
        Number of entries per axis for elements in this space. A
        single integer results in a space with 1 axis.
    dtype (str) : optional
        Data type of each element. See REAL_DTYPES in 
        `odl.util.utility.py` for available options. Defaults to float64
    impl (str) : str, optional
        Impmlementation back-end for the space. See the constant
        TENSOR_SPACE_IMPLS for available backends
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

    The default data type is float64

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
    if dtype is None:
        dtype = default_dtype(lookup_array_backend(str(impl).lower()))
    is_floating_dtype(dtype)
    return tensor_space(shape, dtype=dtype, impl=impl, device=device, **kwargs)



if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
