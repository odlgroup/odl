﻿# Copyright 2014-2016 The ODL development group
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

__all__ = ('vector', 'ntuples', 'fn', 'cn', 'rn')

import numpy as np

from odl.set import RealNumbers, ComplexNumbers
from odl.util.utility import (
    is_real_floating_dtype, is_complex_floating_dtype, is_scalar_dtype)
from odl.space.entry_points import NTUPLES_IMPLS, FN_IMPLS


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
        raise ValueError('data type {!r} not a complex floating-point type'
                         ''.format(dtype))
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
        raise ValueError('data type {!r} not a real floating-point type'
                         ''.format(dtype))
    return rn


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
