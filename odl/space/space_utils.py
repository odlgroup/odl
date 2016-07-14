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

"""Utility functions for space implementations."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np

from odl.set import RealNumbers, ComplexNumbers
from odl.util.utility import (
    is_real_floating_dtype, is_complex_floating_dtype, is_scalar_dtype,
    dtype_repr)
from odl.space.entry_points import (
    NTUPLES_IMPLS, FN_IMPLS, TENSOR_SET_IMPLS, TENSOR_SPACE_IMPLS)


__all__ = ('vector', 'tensor', 'ntuples',
           'fn', 'cn', 'rn',
           'tensor_set', 'tensor_space', 'ctensors', 'rtensors')


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
        The backend to use. See `odl.space.entry_points.NTUPLES_IMPLS` and
        `odl.space.entry_points.FN_IMPLS` for available options.

    Returns
    -------
    vec : `NtuplesBaseVector`
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
    rn(3).element([1.0, 2.0, 3.0])
    >>> vector([1 + 1j, 2, 3 - 2j])
    cn(3).element([(1+1j), (2+0j), (3-2j)])

    Non-scalar types are also supported:

    >>> vector([True, False])
    ntuples(2, 'bool').element([True, False])

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
        space_constructor = fn
    else:
        space_constructor = ntuples

    return space_constructor(
        len(arr), dtype=space_dtype, impl=impl).element(arr)


def ntuples(size, dtype, impl='numpy', **kwargs):
    """Return a set of n-tuples of arbitrary data type.

    Parameters
    ----------
    size : positive int
        The number of dimensions of the space
    dtype :
        Data type of each element. Can be provided in any
        way the `numpy.dtype` function understands, e.g. as built-in type
        or as a string.
    impl : str, optional
        The backend to use. See `odl.space.entry_points.NTUPLES_IMPLS` for
        available options.
    kwargs : optional
        Extra keyword arguments passed to the set constructor.

    Returns
    -------
    ntuple : `NtuplesBase`

    See Also
    --------
    fn : n-tuples over a field with arbitrary scalar data type.
    """
    return NTUPLES_IMPLS[impl](size, dtype, **kwargs)


def fn(size, dtype=None, impl='numpy', **kwargs):
    """Return a space of n-tuples of arbitrary scalar data type.

    Parameters
    ----------
    size : positive int
        The number of dimensions of the space
    dtype : optional
        Data type of each element. Can be provided in any
        way the `numpy.dtype` function understands, e.g. as built-in type
        or as a string.
        For ``None``, the `FnBase.default_dtype` of the created space
        is used.
    impl : str, optional
        The backend to use. See `odl.space.entry_points.FN_IMPLS` for
        available options.
    kwargs : optional
        Extra keyword arguments passed to the space constructor.

    Returns
    -------
    fn : `FnBase`

    See Also
    --------
    ntuples : n-tuples with arbitrary data type.
    """
    fn_type = FN_IMPLS[impl]

    if dtype is None:
        dtype = fn_type.default_dtype()

    return fn_type(size, dtype, **kwargs)


def cn(size, dtype=None, impl='numpy', **kwargs):
    """Return the complex vector space ``C^n``.

    Parameters
    ----------
    size : positive int
        Number of entries in a space element.
    dtype : optional
        Data type of each element. Can be provided in any
        way the `numpy.dtype` function understands, e.g. as built-in type
        or as a string. Only complex floating-point data types are
        allowed.
        For ``None``, the `FnBase.default_dtype` of the created space
        is used in the form ``default_dtype(ComplexNumbers())``.
    impl : str, optional
        The backend to use. See `odl.space.entry_points.FN_IMPLS` for
        available options.
    kwargs :
        Extra keyword arguments passed to the space constructor.

    Returns
    -------
    cn : `FnBase`

    See Also
    --------
    fn : n-tuples over a field with arbitrary scalar data type.
    """
    cn_type = FN_IMPLS[impl]

    if dtype is None:
        dtype = cn_type.default_dtype(ComplexNumbers())

    cn = cn_type(size, dtype, **kwargs)

    if not cn.is_cn:
        raise ValueError('data type {} not a complex floating-point type'
                         ''.format(dtype_repr(dtype)))
    return cn


def rn(size, dtype=None, impl='numpy', **kwargs):
    """Return the real vector space ``R^n``.

    Parameters
    ----------
    size : positive int
        Number of entries in a space element.
    dtype : optional
        Data type of each element. Can be provided in any
        way the `numpy.dtype` function understands, e.g. as built-in type
        or as a string. Only real floating-point data types are
        allowed.
        For ``None``, the `FnBase.default_dtype` of the created space
        is used in the form ``default_dtype(RealNumbers())``.
    impl : str, optional
        The backend to use. See `odl.space.entry_points.FN_IMPLS` for
        available options.
    kwargs :
        Extra keyword arguments passed to the space constructor.

    Returns
    -------
    rn : `FnBase`

    See Also
    --------
    fn : n-tuples over a field with arbitrary scalar data type.
    """
    rn_type = FN_IMPLS[impl]

    if dtype is None:
        dtype = rn_type.default_dtype(RealNumbers())

    rn = rn_type(size, dtype, **kwargs)

    if not rn.is_rn:
        raise ValueError('data type {} not a real floating-point type'
                         ''.format(dtype_repr(dtype)))
    return rn


def tensor(array, dtype=None, order='C', impl='numpy'):
    """Create a tensor from an array-like object.

    Parameters
    ----------
    array : `array-like`
        Array from which to create the vector. Scalars become
        one-dimensional tensors.
    dtype : `object`, optional
        Set the data type of the tensor manually with this option.
        By default, the space type is inferred from the input data.
    order : {'C', 'F'}, optional
        Axis ordering of the data storage.
    impl : `str`
        Implementation backend for the vector. See `TENSOR_SET_IMPLS`
        and `TENSOR_SPACE_IMPLS` for more information.

    Returns
    -------
    tensor : `GeneralTensorBase`
        Tensor created from the input array. Its concrete type depends
        on the provided arguments.

    Notes
    -----
    This is a convenience function and not intended for use in
    speed-critical algorithms.

    Examples
    --------
    >>> odl.tensor([[1, 2, 3],
    ...             [4, 5, 6]])  # No automatic cast to float
    tensor_space((2, 3), 'int').element(
    [[1, 2, 3],
     [4, 5, 6]]
    )
    >>> odl.tensor([[1, 2, 3],
    ...             [4, 5, 6]], dtype=float)
    rtensors((2, 3)).element(
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0]]
    )
    >>> odl.tensor([[1, 2 - 1j, 3],
    ...             [4, 5, 6 + 2j]])
    ctensors((2, 3)).element(
    [[(1+0j), (2-1j), (3+0j)],
     [(4+0j), (5+0j), (6+2j)]]
    )

    Non-scalar types are also supported:

    >>> odl.tensor([[True, True, False],
    ...             [False, False, True]])
    NumpyTensorSet((2, 3), 'bool').element(
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
        space_constructor = tensor_space
    else:
        space_constructor = tensor_set

    return space_constructor(
        arr.shape, dtype=space_dtype, order=order, impl=impl).element(arr)


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
    tset : `TensorSetBase`

    See also
    --------
    tensor_space : space of tensors with arbitrary scalar data type.
    """
    return TENSOR_SET_IMPLS[impl](shape, dtype, order, **kwargs)


def tensor_space(shape, dtype=None, order='C', impl='numpy', **kwargs):
    """Return a tensor space with arbitrary scalar data type.

    Parameters
    ----------
    shape : sequence of positive int
        Number of entries per axis for each element.
    dtype : optional
        Data type of each element. Can be provided in any way the
        `numpy.dtype` function understands, e.g. as built-in type or
        as a string.
        For ``None``, the `TensorSpaceBase.default_dtype` of the
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
    tspace : `TensorSpaceBase`

    See also
    --------
    tensor_set : set of tensors with arbitrary data type.
    """
    tspace_type = TENSOR_SPACE_IMPLS[impl]

    if dtype is None:
        dtype = tspace_type.default_dtype()

    return tspace_type(shape, dtype, order, **kwargs)


def ctensors(shape, dtype=None, order='C', impl='numpy', **kwargs):
    """Return a space of complex tensors.

    Parameters
    ----------
    shape : sequence of positive int
        Number of entries per axis for each element.
    dtype : optional
        Data type of each element. Can be provided in any way the
        `numpy.dtype` function understands, e.g. as built-in type or
        as a string. Only complex floating-point data types are allowed.
        For ``None``, the `TensorSpaceBase.default_dtype` of the
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
    complex_tspace : `TensorSpaceBase`

    See also
    --------
    tensor_space : space of tensors with arbitrary scalar data type.
    """
    complex_tspace_type = TENSOR_SPACE_IMPLS[impl]

    if dtype is None:
        dtype = complex_tspace_type.default_dtype(ComplexNumbers())

    complex_tspace = complex_tspace_type(shape, dtype, order, **kwargs)

    if not complex_tspace.is_complex_space:
        raise ValueError('data type {!r} not a complex floating-point type'
                         ''.format(dtype))

    return complex_tspace


def rtensors(shape, dtype=None, order='C', impl='numpy', **kwargs):
    """Return a space of real tensors.

    Parameters
    ----------
    shape : sequence of positive int
        Number of entries per axis for each element.
    dtype : optional
        Data type of each element. Can be provided in any way the
        `numpy.dtype` function understands, e.g. as built-in type or
        as a string. Only real floating-point data types are allowed.
        For ``None``, the `TensorSpaceBase.default_dtype` of the
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
    real_tspace : `TensorSpaceBase`

    See also
    --------
    tensor_space : space of tensors with arbitrary scalar data type.
    """
    real_tspace_type = TENSOR_SPACE_IMPLS[impl]

    if dtype is None:
        dtype = real_tspace_type.default_dtype(RealNumbers())

    real_tspace = real_tspace_type(shape, dtype, order, **kwargs)

    if not real_tspace.is_real_space:
        raise ValueError('data type {!r} not a real floating-point type'
                         ''.format(dtype))

    return real_tspace


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
