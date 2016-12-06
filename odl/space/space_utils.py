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
from odl.util.utility import is_scalar_dtype
from odl.space.entry_points import TENSOR_SET_IMPLS, TENSOR_SPACE_IMPLS


__all__ = ('tensor', 'tensor_set', 'tensor_space', 'ctensors', 'rtensors',
           'vector', 'ntuples', 'fn', 'cn', 'rn')


def tensor(array, dtype=None, order='C', impl='numpy'):
    """Create a tensor from an array-like object.

    Parameters
    ----------
    array : `array-like`
        Array from which to create the vector. Scalars become
        one-dimensional tensors.
    dtype : optional
        Set the data type of the tensor manually with this option.
        By default, the space type is inferred from the input data.
    order : {'C', 'F'}, optional
        Axis ordering of the data storage.
    impl : `str`
        Implementation backend for the vector. See `TENSOR_SET_IMPLS`
        and `TENSOR_SPACE_IMPLS` for more information.

    Returns
    -------
    tensor : `BaseGeneralizedTensor`
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
    tset : `BaseTensorSet`

    Examples
    --------
    Set of 2x3 tensors with unsigned integer entries:

    >>> odl.tensor_set((2, 3), dtype='uint64')
    tensor_set((2, 3), 'uint64')

    One-dimensional spaces have special constructors:

    >>> odl.tensor_set((3,), dtype='uint64')
    ntuples(3, 'uint64')

    See also
    --------
    tensor_space : Space of tensors with arbitrary scalar data type.
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
        For ``None``, the `BaseTensorSpace.default_dtype` of the
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
    tspace : `BaseTensorSpace`

    Examples
    --------
    Space of 2x3 tensors with ``int64`` entries (although not strictly a
    vector space):

    >>> odl.tensor_space((2, 3), dtype='int64')
    tensor_space((2, 3), 'int')

    The default data type depends on the implementation. For
    ``impl='numpy'``, it is ``'float64'``:

    >>> ts = odl.tensor_space((2, 3))
    >>> ts
    rtensors((2, 3))
    >>> ts.dtype
    dtype('float64')

    One-dimensional spaces have special constructors:

    >>> odl.tensor_space((3,), dtype='int64')
    fn(3, 'int')

    See also
    --------
    tensor_set : Set of tensors with arbitrary data type.
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
        For ``None``, the `BaseTensorSpace.default_dtype` of the
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
    complex_tspace : `BaseTensorSpace`

    Examples
    --------
    Space of complex 2x3 tensors with ``complex64`` entries:

    >>> odl.ctensors((2, 3), dtype='complex64')
    ctensors((2, 3), 'complex64')

    The default data type depends on the implementation. For
    ``impl='numpy'``, it is ``'complex128'``:

    >>> ts = odl.ctensors((2, 3))
    >>> ts
    ctensors((2, 3))
    >>> ts.dtype
    dtype('complex128')

    One-dimensional spaces have special constructors:

    >>> odl.ctensors((3,))
    cn(3)

    See also
    --------
    tensor_space : Space of tensors with arbitrary scalar data type.
    rtensors : Real tensor space.
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
        For ``None``, the `BaseTensorSpace.default_dtype` of the
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
    real_tspace : `BaseTensorSpace`

    Examples
    --------
    Space of real 2x3 tensors with ``float32`` entries:

    >>> odl.rtensors((2, 3), dtype='float32')
    rtensors((2, 3), 'float32')

    The default data type depends on the implementation. For
    ``impl='numpy'``, it is ``'float64'``:

    >>> ts = odl.rtensors((2, 3))
    >>> ts
    rtensors((2, 3))
    >>> ts.dtype
    dtype('float64')

    One-dimensional spaces have special constructors:

    >>> odl.rtensors((3,))
    rn(3)

    See also
    --------
    tensor_space : Space of tensors with arbitrary scalar data type.
    ctensors : Complex tensor space.
    """
    real_tspace_type = TENSOR_SPACE_IMPLS[impl]

    if dtype is None:
        dtype = real_tspace_type.default_dtype(RealNumbers())

    real_tspace = real_tspace_type(shape, dtype, order, **kwargs)

    if not real_tspace.is_real_space:
        raise ValueError('data type {!r} not a real floating-point type'
                         ''.format(dtype))

    return real_tspace


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
    vec : `BaseGeneralizedTensor`
        Vector created from the input array. Its concrete type depends
        on the provided arguments.

    Notes
    -----
    This is a convenience function and not intended for use in
    speed-critical algorithms.

    Examples
    --------
    >>> odl.vector([1, 2, 3])  # No automatic cast to float
    fn(3, 'int').element(
    [1, 2, 3]
    )
    >>> odl.vector([1, 2, 3], dtype=float)
    rn(3).element(
    [1.0, 2.0, 3.0]
    )
    >>> odl.vector([1 + 1j, 2, 3 - 2j])
    cn(3).element(
    [(1+1j), (2+0j), (3-2j)]
    )

    Non-scalar types are also supported:

    >>> odl.vector([True, False])
    ntuples(2, 'bool').element(
    [True, False]
    )

    Scalars become a one-element vector:

    >>> odl.vector(0.0)
    rn(1).element(
    [0.0]
    )
    """
    tens = tensor(array, dtype, order='C', impl=impl)
    if tens.ndim > 1:
        raise ValueError('array has {} dimensions, expected 1'
                         ''.format(tens.ndim))
    return tens


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
    ntuple : `BaseTensorSet`

    Examples
    --------
    Set of 3-tuples with integer entries:

    >>> odl.ntuples(3, dtype='uint64')
    ntuples(3, 'uint64')

    See Also
    --------
    fn : n-tuples over a field with arbitrary scalar data type.
    tensor_set : Generalization for multiple dimensions.
    """
    return tensor_set(shape=(size,), dtype=dtype, order='C', impl=impl,
                      **kwargs)


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
        For ``None``, the `BaseTensorSpace.default_dtype` of the created space
        is used.
    impl : str, optional
        The backend to use. See `odl.space.entry_points.FN_IMPLS` for
        available options.
    kwargs : optional
        Extra keyword arguments passed to the space constructor.

    Returns
    -------
    fn : `BaseTensorSpace`

    Examples
    --------
    Space of 3-tuples with ``int64`` entries (although not strictly a
    vector space):

    >>> odl.fn(3, dtype='int64')
    fn(3, 'int')

    The default data type depends on the implementation. For
    ``impl='numpy'``, it is ``'float64'``:

    >>> spc = odl.fn(3)
    >>> spc
    rn(3)
    >>> spc.dtype
    dtype('float64')

    See Also
    --------
    ntuples : n-tuples with arbitrary data type.
    tensor_space : Generalization for multiple dimensions.
    """
    return tensor_space(shape=(size,), dtype=dtype, order='C', impl=impl,
                        **kwargs)


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
        For ``None``, the `BaseTensorSpace.default_dtype` of the created space
        is used in the form ``default_dtype(ComplexNumbers())``.
    impl : str, optional
        The backend to use. See `odl.space.entry_points.FN_IMPLS` for
        available options.
    kwargs :
        Extra keyword arguments passed to the space constructor.

    Returns
    -------
    cn : `BaseTensorSpace`

    Examples
    --------
    Space of complex 3-tuples with ``complex64`` entries:

    >>> odl.cn(3, dtype='complex64')
    cn(3, 'complex64')

    The default data type depends on the implementation. For
    ``impl='numpy'``, it is ``'complex128'``:

    >>> spc = odl.cn(3)
    >>> spc
    cn(3)
    >>> spc.dtype
    dtype('complex128')

    See Also
    --------
    fn : n-tuples over a field with arbitrary scalar data type.
    ctensors : Generalization for multiple dimensions.
    """
    return ctensors(shape=(size,), dtype=dtype, order='C', impl=impl,
                    **kwargs)


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
        For ``None``, the `BaseTensorSpace.default_dtype` of the created space
        is used in the form ``default_dtype(RealNumbers())``.
    impl : str, optional
        The backend to use. See `odl.space.entry_points.FN_IMPLS` for
        available options.
    kwargs :
        Extra keyword arguments passed to the space constructor.

    Returns
    -------
    rn : `BaseTensorSpace`

    Examples
    --------
    Space of real 3-tuples with ``float32`` entries:

    >>> odl.rn(3, dtype='float32')
    rn(3, 'float32')

    The default data type depends on the implementation. For
    ``impl='numpy'``, it is ``'float64'``:

    >>> spc = odl.rn(3)
    >>> spc
    rn(3)
    >>> spc.dtype
    dtype('float64')

    See Also
    --------
    fn : n-tuples over a field with arbitrary scalar data type.
    rtensors : Generalization for multiple dimensions.
    """
    return rtensors(shape=(size,), dtype=dtype, order='C', impl=impl,
                    **kwargs)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
