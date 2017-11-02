﻿# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""CPU implementations of ``n``-dimensional Cartesian spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future.utils import native
import builtins

import ctypes
from functools import partial
from numbers import Integral
import numpy as np
import scipy.linalg as linalg
from scipy.sparse.base import isspmatrix
import sys

from odl.set import RealNumbers, ComplexNumbers
from odl.space.base_ntuples import FnBase, FnBaseVector
from odl.space.weighting import (
    Weighting, MatrixWeighting, ArrayWeighting,
    ConstWeighting, NoWeighting,
    CustomInner, CustomNorm, CustomDist)
from odl.util import dtype_repr, is_real_dtype
from odl.util.ufuncs import NumpyFnUfuncs


__all__ = ('NumpyFn', 'NumpyFnVector',
           'npy_weighted_dist', 'npy_weighted_norm', 'npy_weighted_inner')


_BLAS_DTYPES = (np.dtype('float32'), np.dtype('float64'),
                np.dtype('complex64'), np.dtype('complex128'))

# Define thresholds for when different implementations should be used
THRESHOLD_SMALL = 100
THRESHOLD_MEDIUM = 50000


class NumpyFn(FnBase):

    """Set of n-tuples of arbitrary type.

    This space implements n-tuples of elements from a `Field`, which is
    usually the real or complex numbers.

    Its elements are represented as instances of the `NumpyFnVector` class.
    """

    impl = 'numpy'

    def __init__(self, size, dtype='float64', **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        size : positive int
            The number of dimensions of the space
        dtype :
            The data type of the storage array. Can be provided in any
            way the `numpy.dtype` function understands, most notably
            as built-in type, as `numpy.dtype` or as string.

            Only scalar data types are allowed.

        weighting : optional
            Use weighted inner product, norm, and dist. The following
            types are supported as ``weighting``:

            `FnWeighting`:
            Use this weighting as-is. Compatibility with this
            space's elements is not checked during init.

            float: Weighting by a constant

            array-like: Weighting by a matrix (2-dim. array) or a vector
            (1-dim. array, corresponds to a diagonal matrix). A matrix
            can also be given as a sparse matrix
            ( `scipy.sparse.spmatrix`).

            Default: no weighting

            This option cannot be combined with ``dist``,
            ``norm`` or ``inner``.

        exponent : positive float, optional
            Exponent of the norm. For values other than 2.0, no
            inner product is defined.
            If ``weight`` is a sparse matrix, only 1.0, 2.0 and
            ``inf`` are allowed.

            This option is ignored if ``dist``, ``norm`` or
            ``inner`` is given.

            Default: 2.0

        Other Parameters
        ----------------

        dist : callable, optional
            The distance function defining a metric on the space.
            It must accept two `NumpyFnVector` arguments and
            fulfill the following mathematical conditions for any
            three vectors ``x, y, z``:

            - ``dist(x, y) >= 0``
            - ``dist(x, y) = 0``  if and only if  ``x = y``
            - ``dist(x, y) = dist(y, x)``
            - ``dist(x, y) <= dist(x, z) + dist(z, y)``

            By default, ``dist(x, y)`` is calculated as ``norm(x - y)``.

            This option cannot be combined with ``weighting``,
            ``norm`` or ``inner``.

        norm : callable, optional
            The norm implementation. It must accept an
            `NumpyFnVector` argument, return a float and satisfy the
            following conditions for all vectors ``x, y`` and scalars
            ``s``:

            - ``||x|| >= 0``
            - ``||x|| = 0``  if and only if  ``x = 0``
            - ``||s * x|| = |s| * ||x||``
            - ``||x + y|| <= ||x|| + ||y||``

            By default, ``norm(x)`` is calculated as ``inner(x, x)``.

            This option cannot be combined with ``weighting``,
            ``dist`` or ``inner``.

        inner : callable, optional
            The inner product implementation. It must accept two
            `NumpyFnVector` arguments, return a element from
            the field of the space (real or complex number) and
            satisfy the following conditions for all vectors
            ``x, y, z`` and scalars ``s``:

            - ``<x, y> = conj(<y, x>)``
            - ``<s*x + y, z> = s * <x, z> + <y, z>``
            - ``<x, x> = 0``  if and only if  ``x = 0``

            This option cannot be combined with ``weighting``,
            ``dist`` or ``norm``.

        kwargs :
            Further keyword arguments are passed to the weighting
            classes.

        See Also
        --------
        NumpyFnMatrixWeighting
        NumpyFnArrayWeighting
        NumpyFnConstWeighting

        Examples
        --------
        >>> space = NumpyFn(3, 'float')
        >>> space
        rn(3)
        >>> space = NumpyFn(3, 'float', weighting=[1, 2, 3])
        >>> space
        rn(3, weighting=[1, 2, 3])
        """
        # TODO: fix dead link `scipy.sparse.spmatrix`
        if sys.version_info.major < 3 and dtype is builtins.int:
            raise TypeError('cannot use `builtins.int` as `dtype` since '
                            'Numpy does not recognize it as int')

        if np.dtype(dtype).char not in self.available_dtypes():
            raise ValueError('`dtype` {!r} not supported'.format(dtype))
        super(NumpyFn, self).__init__(size, dtype)

        dist = kwargs.pop('dist', None)
        norm = kwargs.pop('norm', None)
        inner = kwargs.pop('inner', None)
        weighting = kwargs.pop('weighting', None)
        exponent = kwargs.pop('exponent', 2.0)

        # Check validity of option combination (3 or 4 out of 4 must be None)
        if sum(x is None for x in (dist, norm, inner, weighting)) < 3:
            raise ValueError('invalid combination of options `weighting`, '
                             '`dist`, `norm` and `inner`')

        if any(x is not None for x in (dist, norm, inner)) and exponent != 2.0:
            raise ValueError('`exponent` cannot be used together with '
                             '`dist`, `norm` and `inner`')

        # Set the weighting
        if weighting is not None:
            if isinstance(weighting, Weighting):
                self.__weighting = weighting
            elif np.isscalar(weighting):
                self.__weighting = NumpyFnConstWeighting(
                    weighting, exponent, **kwargs)
            elif weighting is None:
                # Need to wait until dist, norm and inner are handled
                pass
            elif isspmatrix(weighting):
                self.__weighting = NumpyFnMatrixWeighting(
                    weighting, exponent, **kwargs)
            else:  # last possibility: make a matrix
                arr = np.asarray(weighting)
                if arr.dtype == object:
                    raise ValueError('invalid weighting argument {}'
                                     ''.format(weighting))
                if arr.ndim == 1:
                    self.__weighting = NumpyFnArrayWeighting(
                        arr, exponent, **kwargs)
                elif arr.ndim == 2:
                    self.__weighting = NumpyFnMatrixWeighting(
                        arr, exponent, **kwargs)
                else:
                    raise ValueError('array-like input {} is not 1- or '
                                     '2-dimensional'.format(weighting))

        elif dist is not None:
            self.__weighting = NumpyFnCustomDist(dist)
        elif norm is not None:
            self.__weighting = NumpyFnCustomNorm(norm)
        elif inner is not None:
            self.__weighting = NumpyFnCustomInner(inner)
        else:  # all None -> no weighing
            self.__weighting = NumpyFnNoWeighting(exponent)

    @property
    def exponent(self):
        """Exponent of the norm and distance."""
        return self.weighting.exponent

    @property
    def weighting(self):
        """This space's weighting scheme."""
        return self.__weighting

    @property
    def is_weighted(self):
        """``True`` if the weighting is not `NumpyFnNoWeighting`."""
        return not isinstance(self.weighting, NumpyFnNoWeighting)

    def element(self, inp=None, data_ptr=None):
        """Create a new element.

        Parameters
        ----------
        inp : `array-like`, optional
            Input to initialize the new element.

            If ``inp`` is ``None``, an empty element is created with no
            guarantee of its state (memory allocation only).

            If ``inp`` is a `numpy.ndarray` of shape ``(size,)``
            and the same data type as this space, the array is wrapped,
            not copied.
            Other `array-like` objects are copied.

        Returns
        -------
        element : `NumpyFnVector`
            The new element created (from ``inp``).

        Notes
        -----
        This method preserves "array views" of correct size and type,
        see the examples below.

        Examples
        --------
        >>> bool3 = odl.fn(3, dtype=bool)
        >>> x = bool3.element([True, True, False])
        >>> x
        fn(3, 'bool').element([ True,  True, False])
        >>> x.space
        fn(3, 'bool')

        Construction from data pointer:

        >>> int3 = odl.fn(3, dtype='int')
        >>> x = int3.element([1, 2, 3])
        >>> y = int3.element(data_ptr=x.data_ptr)
        >>> print(y)
        [1, 2, 3]
        >>> y[0] = 5
        >>> print(x)
        [5, 2, 3]
        """
        if inp is None:
            if data_ptr is None:
                arr = np.empty(self.size, dtype=self.dtype)
                return self.element_type(self, arr)
            else:
                ctype_array_def = ctypes.c_byte * (self.size *
                                                   self.dtype.itemsize)
                as_ctype_array = ctype_array_def.from_address(data_ptr)
                as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
                arr = as_numpy_array.view(dtype=self.dtype)
                return self.element_type(self, arr)
        else:
            if data_ptr is None:
                if inp in self:
                    return inp
                else:
                    arr = np.array(inp, copy=False, dtype=self.dtype, ndmin=1)
                    if arr.shape != (self.size,):
                        raise ValueError('expected input shape {}, got {}'
                                         ''.format((self.size,), arr.shape))

                    return self.element_type(self, arr)
            else:
                raise ValueError('cannot provide both `inp` and `data_ptr`')

    def zero(self):
        """Create a vector of zeros.

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> x = r3.zero()
        >>> x
        rn(3).element([ 0.,  0.,  0.])
        """
        return self.element(np.zeros(self.size, dtype=self.dtype))

    def one(self):
        """Create a vector of ones.

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> x = r3.one()
        >>> x
        rn(3).element([ 1.,  1.,  1.])
        """
        return self.element(np.ones(self.size, dtype=self.dtype))

    def _lincomb(self, a, x1, b, x2, out):
        """Linear combination of ``x1`` and ``x2``.

        Calculate ``out = a*x1 + b*x2`` using optimized BLAS
        routines if possible.

        Parameters
        ----------
        a, b : `FnBase.field` elements
            Scalars to multiply ``x1`` and ``x2`` with
        x1, x2 : `NumpyFnVector`
            Summands in the linear combination
        out : `NumpyFnVector`
            Vector to which the result is written

        Returns
        -------
        None

        Examples
        --------
        >>> c3 = NumpyFn(3, dtype=complex)
        >>> x = c3.element([1+1j, 2-1j, 3])
        >>> y = c3.element([4+0j, 5, 6+0.5j])
        >>> out = c3.element()
        >>> c3.lincomb(2j, x, 3-1j, y, out)  # out is returned
        cn(3).element([ 10.0-2.j ,  17.0-1.j ,  18.5+1.5j])
        >>> out
        cn(3).element([ 10.0-2.j ,  17.0-1.j ,  18.5+1.5j])
        """
        _lincomb_impl(a, x1, b, x2, out, self.dtype)

    def _dist(self, x1, x2):
        """Calculate the distance between two vectors.

        Parameters
        ----------
        x1, x2 : `NumpyFnVector`
            Vectors whose mutual distance is calculated

        Returns
        -------
        dist : float
            Distance between the vectors

        Examples
        --------
        The default case is the euclidean distance

        >>> c2_2 = NumpyFn(2, dtype=complex)
        >>> x = c2_2.element([3+1j, 4])
        >>> y = c2_2.element([1j, 4-4j])
        >>> c2_2.dist(x, y)
        5.0

        If the user has given another dist function, that one is used instead.
        For example, the 2-norm can be given explicitly:

        >>> from numpy.linalg import norm
        >>> c2_2 = NumpyFn(2, dtype=complex,
        ...                dist=lambda x, y: norm(x - y, ord=2))
        >>> x = c2_2.element([3+1j, 4])
        >>> y = c2_2.element([1j, 4-4j])
        >>> c2_2.dist(x, y)
        5.0

        Likewise, the 1-norm can be given

        >>> c2_1 = NumpyFn(2, dtype=complex,
        ...                dist=lambda x, y: norm(x - y, ord=1))
        >>> x = c2_1.element([3+1j, 4])
        >>> y = c2_1.element([1j, 4-4j])
        >>> c2_1.dist(x, y)
        7.0
        """
        return self.weighting.dist(x1, x2)

    def _norm(self, x):
        """Calculate the norm of a vector.

        Parameters
        ----------
        x : `NumpyFnVector`
            The vector whose norm is calculated

        Returns
        -------
        norm : float
            Norm of the vector

        Examples
        --------
        >>> from numpy.linalg import norm

        2-norm

        >>> c2_2 =  NumpyFn(2, dtype=complex, norm=norm)
        >>> x = c2_2.element([3+1j, 1-5j])
        >>> c2_2.norm(x)
        6.0

        1-norm

        >>> from functools import partial
        >>> c2_1 = NumpyFn(2, dtype=complex, norm=partial(norm, ord=1))
        >>> x = c2_1.element([3-4j, 12+5j])
        >>> c2_1.norm(x)
        18.0
        """
        return self.weighting.norm(x)

    def _inner(self, x1, x2):
        """Raw inner product of two vectors.

        Parameters
        ----------
        x1, x2 : `NumpyFnVector`
            The vectors whose inner product is calculated

        Returns
        -------
        inner : `field` element
            Inner product of the vectors

        Examples
        --------
        >>> c3 = NumpyFn(2, dtype=complex, inner=lambda x, y: np.vdot(y, x))
        >>> x = c3.element([5+1j, -2j])
        >>> y = c3.element([1, 1+1j])
        >>> c3.inner(x, y) == (5+1j)*1 + (-2j)*(1-1j)
        True

        Define a space with custom inner product:

        >>> weights = np.array([1., 2.])
        >>> def weighted_inner(x, y):
        ...     return np.vdot(weights * y.data, x.data)

        >>> c3w = NumpyFn(2, dtype=complex, inner=weighted_inner)
        >>> x = c3w.element(x)  # elements must be cast (no copy)
        >>> y = c3w.element(y)
        >>> c3w.inner(x, y) == 1*(5+1j)*1 + 2*(-2j)*(1-1j)
        True
        """
        return self.weighting.inner(x1, x2)

    def _multiply(self, x1, x2, out):
        """Entry-wise product of two vectors, assigned to out.

        Parameters
        ----------
        x1, x2 : `NumpyFnVector`
            Factors in the product
        out : `NumpyFnVector`
            Vector to which the result is written

        Returns
        -------
        None

        Examples
        --------
        >>> c3 = NumpyFn(3, dtype=complex)
        >>> x = c3.element([5+1j, 3, 2-2j])
        >>> y = c3.element([1, 2+1j, 3-1j])
        >>> out = c3.element()
        >>> c3.multiply(x, y, out)  # out is returned
        cn(3).element([ 5.+1.j,  6.+3.j,  4.-8.j])
        >>> out
        cn(3).element([ 5.+1.j,  6.+3.j,  4.-8.j])
        """
        np.multiply(x1.data, x2.data, out=out.data)

    def _divide(self, x1, x2, out):
        """Entry-wise division of two vectors, assigned to out.

        Parameters
        ----------
        x1, x2 : `NumpyFnVector`
            Dividend and divisor in the quotient
        out : `NumpyFnVector`
            Vector to which the result is written

        Returns
        -------
        None

        Examples
        --------
        >>> r3 = NumpyFn(3)
        >>> x = r3.element([3, 5, 6])
        >>> y = r3.element([1, 2, 2])
        >>> out = r3.element()
        >>> r3.divide(x, y, out)  # out is returned
        rn(3).element([ 3. ,  2.5,  3. ])
        >>> out
        rn(3).element([ 3. ,  2.5,  3. ])
        """
        np.divide(x1.data, x2.data, out=out.data)

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if ``other`` is an instance of this space's type
            with the same `NtuplesBase.size` and `NtuplesBase.dtype`,
            and identical distance function, ``False`` otherwise.

        Examples
        --------
        >>> from numpy.linalg import norm
        >>> def dist(x, y, ord):
        ...     return norm(x - y, ord)

        >>> from functools import partial
        >>> dist2 = partial(dist, ord=2)
        >>> c3 = NumpyFn(3, dtype=complex, dist=dist2)
        >>> c3_same = NumpyFn(3, dtype=complex, dist=dist2)
        >>> c3  == c3_same
        True

        Different ``dist`` functions result in different spaces - the
        same applies for ``norm`` and ``inner``:

        >>> dist1 = partial(dist, ord=1)
        >>> r3_1 = NumpyFn(3, dist=dist1)
        >>> r3_2 = NumpyFn(3, dist=dist2)
        >>> r3_1 == r3_2
        False

        Be careful with Lambdas - they result in non-identical function
        objects:

        >>> r3_lambda1 = NumpyFn(3, dist=lambda x, y: norm(x-y, ord=1))
        >>> r3_lambda2 = NumpyFn(3, dist=lambda x, y: norm(x-y, ord=1))
        >>> r3_lambda1 == r3_lambda2
        False

        An `NumpyFn` space with the same data type is considered equal:

        >>> c3 = NumpyFn(3, dtype=complex)
        >>> f3_cdouble = NumpyFn(3, dtype='complex128')
        >>> c3 == f3_cdouble
        True
        """
        if other is self:
            return True

        return (super(NumpyFn, self).__eq__(other) and
                self.weighting == other.weighting)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((super(NumpyFn, self).__hash__(), self.weighting))

    @staticmethod
    def available_dtypes():
        """Available data types.

        Notes
        -----
        This is all dtypes available to numpy. See ``numpy.sctypes``
        for more information.

        The available dtypes may depend on the specific system used.
        """
        all_dtypes = []
        for lst in np.sctypes.values():
            all_dtypes.extend(set(lst))
        dtypes = [np.dtype(dtype) for dtype in all_dtypes]
        dtypes.remove(np.dtype('void'))
        return tuple(dtypes)

    @staticmethod
    def default_dtype(field=None):
        """Return the default of `NumpyFn` data type for a given field.

        Parameters
        ----------
        field : `Field`, optional
            Set of numbers to be represented by a data type.
            Currently supported : `RealNumbers`, `ComplexNumbers`.
            Default: `RealNumbers`

        Returns
        -------
        dtype : `numpy.dtype`
            Numpy data type specifier. The returned defaults are:

            ``RealNumbers()`` : ``np.dtype('float64')``

            ``ComplexNumbers()`` : ``np.dtype('complex128')``
        """
        if field is None or field == RealNumbers():
            return np.dtype('float64')
        elif field == ComplexNumbers():
            return np.dtype('complex128')
        else:
            raise ValueError('no default data type defined for field {}.'
                             ''.format(field))

    def __repr__(self):
        """Return ``repr(self)``."""
        if self.is_real:
            constructor_name = 'rn'
        elif self.is_complex:
            constructor_name = 'cn'
        else:
            constructor_name = 'fn'

        inner_str = '{}'.format(self.size)
        if self.dtype != self.default_dtype(self.field):
            inner_str += ', {}'.format(dtype_repr(self.dtype))

        weight_str = self.weighting.repr_part
        if weight_str:
            inner_str += ', ' + weight_str
        return '{}({})'.format(constructor_name, inner_str)

    @property
    def element_type(self):
        """`NumpyFnVector`"""
        return NumpyFnVector


class NumpyFnVector(FnBaseVector):

    """Representation of an `NumpyFn` element."""

    def __init__(self, space, data):
        """Initialize a new instance."""
        super(NumpyFnVector, self).__init__(space)
        self.__data = data

    @property
    def data(self):
        """Raw Numpy array representing the data."""
        return self.__data

    def asarray(self, start=None, stop=None, step=None, out=None):
        """Extract the data of this array as a numpy array.

        Parameters
        ----------
        start : int, optional
            Start position. ``None`` means the first element.
        stop : int, optional
            One element past the last element to be extracted.
            ``None`` means the last element.
        step : int, optional
            Step length. ``None`` is equivalent to 1.
        out : `numpy.ndarray`, optional
            Array to which the result should be written.
            Has to be contiguous and of the correct data type.

        Returns
        -------
        asarray : `numpy.ndarray`
            Numpy array of the same type as the space.

        Examples
        --------
        >>> import ctypes
        >>> vec = odl.fn(3, 'float').element([1, 2, 3])
        >>> vec.asarray()
        array([ 1.,  2.,  3.])
        >>> vec.asarray(start=1, stop=3)
        array([ 2.,  3.])

        Using the out parameter

        >>> out = np.empty((3,), dtype='float')
        >>> result = vec.asarray(out=out)
        >>> out
        array([ 1.,  2.,  3.])
        >>> result is out
        True
        """
        if out is None:
            return self.data[start:stop:step]
        else:
            out[:] = self.data[start:stop:step]
            return out

    @property
    def data_ptr(self):
        """A raw pointer to the data container.

        Examples
        --------
        >>> import ctypes
        >>> vec = odl.fn(3, 'int32').element([1, 2, 3])
        >>> arr_type = ctypes.c_int32 * 3
        >>> buffer = arr_type.from_address(vec.data_ptr)
        >>> arr = np.frombuffer(buffer, dtype='int32')
        >>> print(arr)
        [1 2 3]

        In-place modification via pointer:

        >>> arr[0] = 5
        >>> print(vec)
        [5, 2, 3]
        """
        return self.data.ctypes.data

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if all entries of other are equal to this
            vector's entries, ``False`` otherwise.

        Notes
        -----
        Space membership is not checked, hence vectors from
        different spaces can be equal.

        Examples
        --------
        >>> vec1 = odl.fn(3, int).element([1, 2, 3])
        >>> vec2 = odl.fn(3, int).element([-1, 2, 0])
        >>> vec1 == vec2
        False
        >>> vec2 = odl.fn(3, int).element([1, 2, 3])
        >>> vec1 == vec2
        True

        Space membership matters:

        >>> vec2 = odl.fn(3, float).element([1, 2, 3])
        >>> vec1 == vec2 or vec2 == vec1
        False
        """
        if other is self:
            return True
        elif other not in self.space:
            return False
        else:
            return np.array_equal(self.data, other.data)

    def copy(self):
        """Create an identical (deep) copy of this vector.

        Parameters
        ----------
        None

        Returns
        -------
        copy : `NumpyFnVector`
            The deep copy

        Examples
        --------
        >>> vec1 = odl.fn(3, 'int').element([1, 2, 3])
        >>> vec2 = vec1.copy()
        >>> vec2
        fn(3, 'int').element([1, 2, 3])
        >>> vec1 == vec2
        True
        >>> vec1 is vec2
        False
        """
        return self.space.element(self.data.copy())

    def __getitem__(self, indices):
        """Access values of this vector.

        Parameters
        ----------
        indices : int or `slice`
            The position(s) that should be accessed

        Returns
        -------
        values : scalar or `NumpyFnVector`
            The value(s) at the index (indices)

        Examples
        --------
        >>> bool3 = odl.fn(4, dtype=bool)
        >>> x = bool3.element([True, False, True, True])
        >>> x[0]
        True
        >>> x[1:3]
        fn(2, 'bool').element([False,  True])
        """
        if isinstance(indices, Integral):
            return self.data[indices]  # single index
        else:
            arr = self.data[indices]
            return type(self.space)(len(arr), dtype=self.dtype).element(arr)

    def __setitem__(self, indices, values):
        """Set values of this vector.

        Parameters
        ----------
        indices : int or `slice`
            The position(s) that should be set
        values : scalar or `array-like`
            The value(s) that are to be assigned.

            If ``indices`` is an integer, ``value`` must be scalar.

            If ``indices`` is a slice, ``value`` must be
            broadcastable to the size of the slice (same size,
            shape ``(1,)`` or single value).

        Returns
        -------
        None

        Examples
        --------
        >>> int_3 = odl.fn(3, 'int')
        >>> x = int_3.element([1, 2, 3])
        >>> x[0] = 5
        >>> x
        fn(3, 'int').element([5, 2, 3])

        Assignment from array-like structures or another
        vector:

        >>> y = odl.fn(2, 'short').element([-1, 2])
        >>> x[:2] = y
        >>> x
        fn(3, 'int').element([-1, 2, 3])
        >>> x[1:3] = [7, 8]
        >>> x
        fn(3, 'int').element([-1, 7, 8])
        >>> x[:] = np.array([0, 0, 0])
        >>> x
        fn(3, 'int').element([0, 0, 0])

        Broadcasting is also supported:

        >>> x[1:3] = -2.
        >>> x
        fn(3, 'int').element([ 0, -2, -2])

        Array views are preserved:

        >>> y = x[::2]  # view into x
        >>> y[:] = -9
        >>> print(y)
        [-9, -9]
        >>> print(x)
        [-9, -2, -9]

        Be aware of unsafe casts and over-/underflows, there
        will be warnings at maximum.

        >>> x = odl.fn(2, 'int8').element([0, 0])
        >>> maxval = 255  # maximum signed 8-bit unsigned int
        >>> x[0] = maxval + 1
        >>> x
        fn(2, 'int8').element([0, 0])
        """
        if isinstance(values, NumpyFnVector):
            self.data[indices] = values.data
        else:
            self.data[indices] = values

    @property
    def ufuncs(self):
        """`NumpyFnUfuncs`, access to numpy style ufuncs.

        Examples
        --------
        >>> r2 = NumpyFn(2)
        >>> x = r2.element([1, -2])
        >>> x.ufuncs.absolute()
        rn(2).element([ 1.,  2.])

        These functions can also be used with broadcasting

        >>> x.ufuncs.add(3)
        rn(2).element([ 4.,  1.])

        and non-space elements

        >>> x.ufuncs.subtract([3, 3])
        rn(2).element([-2., -5.])

        There is also support for various reductions (sum, prod, min, max)

        >>> x.ufuncs.sum()
        -1.0

        They also support an out parameter

        >>> y = r2.element([3, 4])
        >>> out = r2.element()
        >>> result = x.ufuncs.add(y, out=out)
        >>> result
        rn(2).element([ 4.,  2.])
        >>> result is out
        True

        Notes
        -----
        These are optimized for use with ``NumpyFnVector`` objects and
        incur no overhead.
        """
        return NumpyFnUfuncs(self)

    @property
    def real(self):
        """Real part of this vector.

        Returns
        -------
        real : `NumpyFnVector` view with real dtype
            The real part this vector as a vector in `rn`

        Examples
        --------
        >>> x = odl.cn(3).element([5+1j, 3, 2-2j])
        >>> x.real
        rn(3).element([ 5.,  3.,  2.])

        The `real` vector is really a view, so changes affect
        the original array:

        >>> x.real *= 2
        >>> x
        cn(3).element([ 10.+1.j,   6.+0.j,   4.-2.j])
        """
        return self.space.real_space.element(self.data.real)

    @real.setter
    def real(self, newreal):
        """Setter for the real part.

        This method is invoked by ``vec.real = other``.

        Parameters
        ----------
        newreal : `array-like` or scalar
            The new real part for this vector.

        Examples
        --------
        >>> x = odl.cn(3).element([5+1j, 3, 2-2j])
        >>> a = odl.rn(3).element([0, 0, 0])
        >>> x.real = a
        >>> x
        cn(3).element([ 0.+1.j,  0.+0.j,  0.-2.j])

        Other array-like types and broadcasting:

        >>> x.real = 1.0
        >>> x
        cn(3).element([ 1.+1.j,  1.+0.j,  1.-2.j])
        >>> x.real = [0, 2, -1]
        >>> x
        cn(3).element([ 0.+1.j,  2.+0.j, -1.-2.j])
        """
        self.real.data[:] = newreal

    @property
    def imag(self):
        """Imaginary part of this vector.

        Returns
        -------
        imag : `NumpyFnVector`
            The imaginary part this vector as a vector in `rn`

        Examples
        --------
        >>> x = odl.cn(3).element([5+1j, 3, 2-2j])
        >>> x.imag
        rn(3).element([ 1.,  0., -2.])

        The `imag` vector is really a view, so changes affect
        the original array:

        >>> x.imag *= 2
        >>> x
        cn(3).element([ 5.+2.j,  3.+0.j,  2.-4.j])
        """
        return self.space.real_space.element(self.data.imag)

    @imag.setter
    def imag(self, newimag):
        """Setter for the imaginary part.

        This method is invoked by ``vec.imag = other``.

        Parameters
        ----------
        newimag : `array-like` or scalar
            The new imaginary part for this vector.

        Examples
        --------
        >>> x = odl.cn(3).element([5+1j, 3, 2-2j])
        >>> a = odl.rn(3).element([0, 0, 0])
        >>> x.imag = a
        >>> x
        cn(3).element([ 5.+0.j,  3.+0.j,  2.+0.j])

        Other array-like types and broadcasting:

        >>> x.imag = 1.0
        >>> x
        cn(3).element([ 5.+1.j,  3.+1.j,  2.+1.j])
        >>> x.imag = [0, 2, -1]
        >>> x
        cn(3).element([ 5.+0.j,  3.+2.j,  2.-1.j])
        """
        self.imag.data[:] = newimag

    def conj(self, out=None):
        """Complex conjugate of this vector.

        Parameters
        ----------
        out : `NumpyFnVector`, optional
            Vector to which the complex conjugate is written.
            Must be an element of this vector's space.

        Returns
        -------
        out : `NumpyFnVector`
            The complex conjugate vector. If ``out`` was provided,
            the returned object is a reference to it.

        Examples
        --------
        Default usage:

        >>> x = odl.cn(3).element([5+1j, 3, 2-2j])
        >>> x.conj()
        cn(3).element([ 5.-1.j,  3.-0.j,  2.+2.j])

        The out parameter allows you to avoid a copy:

        >>> z = odl.cn(3).element()
        >>> z_out = x.conj(out=z)
        >>> z
        cn(3).element([ 5.-1.j,  3.-0.j,  2.+2.j])
        >>> z_out is z
        True

        It can also be used for in-place conjugation:

        >>> x_out = x.conj(out=x)
        >>> x
        cn(3).element([ 5.-1.j,  3.-0.j,  2.+2.j])
        >>> x_out is x
        True
        """
        if out is None:
            return self.space.element(self.data.conj())
        else:
            self.data.conj(out.data)
            return out

    def __ipow__(self, other):
        """Return ``self **= other``."""
        try:
            if other == int(other):
                return super(NumpyFnVector, self).__ipow__(other)
        except TypeError:
            pass

        np.power(self.data, other, out=self.data)
        return self


def _weighting(weights, exponent):
    """Return a weighting whose type is inferred from the arguments."""
    if np.isscalar(weights):
        weighting = NumpyFnConstWeighting(weights, exponent=exponent)
    elif isspmatrix(weights):
        weighting = NumpyFnMatrixWeighting(weights, exponent=exponent)
    else:
        weights, weights_in = np.asarray(weights), weights
        if weights.dtype == object:
            raise ValueError('bad weights {}'.format(weights_in))
        if weights.ndim == 1:
            weighting = NumpyFnArrayWeighting(weights, exponent=exponent)
        elif weights.ndim == 2:
            weighting = NumpyFnMatrixWeighting(weights, exponent=exponent)
        else:
            raise ValueError('array-like `weights` must have 1 or 2 '
                             'dimensions, but {} has {} dimensions'
                             ''.format(weights, weights.ndim))
    return weighting


def npy_weighted_inner(weights):
    """Weighted inner product on `NumpyFn` spaces as free function.

    Parameters
    ----------
    weights : scalar or `array-like`
        Weights of the inner product. A scalar is interpreted as a
        constant weight, a 1-dim. array as a weighting array and a
        2-dimensional array as a weighting matrix.

    Returns
    -------
    inner : callable
        Inner product function with given weight. Constant weightings
        are applicable to spaces of any size, for arrays the sizes
        of the weighting and the space must match.

    See Also
    --------
    NumpyFnConstWeighting
    NumpyFnArrayWeighting
    NumpyFnMatrixWeighting
    """
    return _weighting(weights, exponent=2.0).inner


def npy_weighted_norm(weights, exponent=2.0):
    """Weighted norm on `NumpyFn` spaces as free function.

    Parameters
    ----------
    weights : scalar or `array-like`
        Weights of the norm. A scalar is interpreted as a
        constant weight, a 1-dim. array as a weighting array and a
        2-dimensional array as a weighting matrix.
    exponent : positive float, optional
        Exponent of the norm. If ``weight`` is a sparse matrix, only
        1.0, 2.0 and ``inf`` are allowed.

    Returns
    -------
    norm : callable
        Norm function with given weight. Constant weightings
        are applicable to spaces of any size, for arrays the sizes
        of the weighting and the space must match.

    See Also
    --------
    NumpyFnConstWeighting
    NumpyFnArrayWeighting
    NumpyFnMatrixWeighting
    """
    return _weighting(weights, exponent=exponent).norm


def npy_weighted_dist(weights, exponent=2.0, use_inner=False):
    """Weighted distance on `NumpyFn` spaces as free function.

    Parameters
    ----------
    weights : scalar or `array-like`
        Weights of the distance. A scalar is interpreted as a
        constant weight, a 1-dim. array as a weighting array and a
        2-dimensional array as a weighting matrix.
    exponent : positive float, optional
        Exponent of the norm. If ``weight`` is a sparse matrix, only
        1.0, 2.0 and ``inf`` are allowed.

    Returns
    -------
    dist : callable
        Distance function with given weight. Constant weightings
        are applicable to spaces of any size, for arrays the sizes
        of the weighting and the space must match.

    See Also
    --------
    NumpyFnConstWeighting
    NumpyFnArrayWeighting
    NumpyFnMatrixWeighting
    """
    return _weighting(weights, exponent=exponent).dist


def _norm_default(x):
    """Default Euclidean norm implementation."""
    if _blas_is_applicable(x):
        nrm2 = linalg.blas.get_blas_funcs('nrm2', dtype=x.dtype)
        norm = partial(nrm2, n=native(x.size))
    else:
        norm = np.linalg.norm
    return norm(x.data)


def _pnorm_default(x, p):
    """Default p-norm implementation."""
    return np.linalg.norm(x.data, ord=p)


def _pnorm_diagweight(x, p, w):
    """Diagonally weighted p-norm implementation."""
    # This is faster than first applying the weights and then summing with
    # BLAS dot or nrm2
    xp = np.abs(x.data)
    if np.isfinite(p):
        xp = np.power(xp, p, out=xp)
        xp *= w  # w is a plain NumPy array
        return np.sum(xp) ** (1 / p)
    else:
        xp *= w
        return np.max(xp)


def _inner_default(x1, x2):
    """Default Euclidean inner product implementation."""
    size = x1.size

    # x2 as first argument because we want linearity in x1

    if size > THRESHOLD_MEDIUM and _blas_is_applicable(x1, x2):
        dotc = linalg.blas.get_blas_funcs('dotc', dtype=x1.dtype)
        dot = dotc(x2.data, x1.data, n=native(size))
    elif is_real_dtype(x1.dtype):
        dot = np.dot(x2.data, x1.data)  # still much faster than vdot
    else:
        dot = np.vdot(x2.data, x1.data)  # slowest alternative

    return dot


class NumpyFnMatrixWeighting(MatrixWeighting):

    """Matrix weighting for `NumpyFn`.

    For exponent 2.0, a new weighted inner product with matrix ``W``
    is defined as::

        <a, b>_W := <W * a, b> = b^H * W * a

    with ``b^H`` standing for transposed complex conjugate.

    For other exponents, only norm and dist are defined. In the case of
    exponent ``inf``, the weighted norm is::

        ||a||_{W, inf} := ||W * a||_inf

    otherwise it is::

        ||a||_{W, p} := ||W^{1/p} * a||_p

    Note that this definition does **not** fulfill the limit property
    in ``p``, i.e.::

        ||x||_{W, p} --/-> ||x||_{W, inf}  for p --> inf

    unless ``W`` is the identity matrix.

    The matrix must be Hermitian and posivive definite, otherwise it
    does not define an inner product or norm, respectively. This is not
    checked during initialization.
    """

    def __init__(self, matrix, exponent=2.0, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        matrix :  `scipy.sparse.spmatrix` or `array-like`, 2-dim.
            Square weighting matrix of the inner product
        exponent : positive float, optional
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
            If ``matrix`` is a sparse matrix, only 1.0, 2.0 and ``inf``
            are allowed.
        precomp_mat_pow : bool, optional
            If ``True``, precompute the matrix power ``W ** (1/p)``
            during initialization. This has no effect if ``exponent``
            is 1.0, 2.0 or ``inf``.

            Default: False

        cache_mat_pow : bool, optional
            If ``True``, cache the matrix power ``W ** (1/p)``. This can
            happen either during initialization or in the first call to
            ``norm`` or ``dist``, resp. This has no effect if
            ``exponent`` is 1.0, 2.0 or ``inf``.

            Default: True

        cache_mat_decomp : bool, optional
            If ``True``, cache the eigenbasis decomposition of the
            matrix. This can happen either during initialization or in
            the first call to ``norm`` or ``dist``, resp. This has no
            effect if ``exponent`` is 1.0, 2.0 or ``inf``.

            Default: False

        Notes
        -----
        The matrix power ``W ** (1/p)`` is computed with by eigenbasis
        decomposition::

            eigval, eigvec = scipy.linalg.eigh(matrix)
            mat_pow = (eigval ** p * eigvec).dot(eigvec.conj().T)

        Depending on the matrix size, this can be rather expensive.
        """
        # TODO: fix dead link `scipy.sparse.spmatrix`
        super(NumpyFnMatrixWeighting, self).__init__(
            matrix, impl='numpy', exponent=exponent, **kwargs)

    def inner(self, x1, x2):
        """Calculate the matrix-weighted inner product of two vectors.

        Parameters
        ----------
        x1, x2 : `NumpyFnVector`
            Vectors whose inner product is calculated

        Returns
        -------
        inner : float or complex
            The inner product of the vectors
        """
        if self.exponent != 2.0:
            raise NotImplementedError('no inner product defined for '
                                      'exponent != 2 (got {})'
                                      ''.format(self.exponent))
        else:
            inner = _inner_default(x1.space.element(self.matrix.dot(x1)), x2)
            if is_real_dtype(x1.dtype):
                return float(inner)
            else:
                return complex(inner)

    def norm(self, x):
        """Calculate the matrix-weighted norm of a vector.

        Parameters
        ----------
        x : `NumpyFnVector`
            Vector whose norm is calculated

        Returns
        -------
        norm : float
            The norm of the vector
        """
        if self.exponent == 2.0:
            norm_squared = self.inner(x, x).real  # TODO: optimize?
            return np.sqrt(norm_squared)

        if self._mat_pow is None:
            # This case can only be reached if p != 1,2,inf
            if self.matrix_issparse:
                raise NotImplementedError('sparse matrix powers not '
                                          'suppoerted')

            if self._eigval is None or self._eigvec is None:
                # No cached decomposition, computing new one
                eigval, eigvec = linalg.eigh(self.matrix)
                if self._cache_mat_decomp:
                    self._eigval, self._eigvec = eigval, eigvec
                    eigval_pow = eigval ** (1.0 / self.exponent)
                else:
                    # Not storing eigenvalues, so we can destroy them
                    eigval_pow = eigval
                    eigval_pow **= 1.0 / self.exponent
            else:
                # Using cached, cannot destroy
                eigval, eigvec = self._eigval, self._eigvec
                eigval_pow = eigval ** (1.0 / self.exponent)

            mat_pow = (eigval_pow * eigvec).dot(eigvec.conj().T)
            if self._cache_mat_pow:
                self._mat_pow = mat_pow
        else:
            mat_pow = self._mat_pow

        return float(_pnorm_default(x.space.element(mat_pow.dot(x)),
                                    self.exponent))


class NumpyFnArrayWeighting(ArrayWeighting):

    """Weighting of `Fn` by an array.

    This class defines a point-wise weighting, i.e., a weighting with
    a different value for each index.
    See ``Notes`` for mathematical details.
    """

    def __init__(self, array, exponent=2.0):
        """Initialize a new instance.

        Parameters
        ----------
        array : `array-like`, one-dim.
            Weighting array of the inner product, norm and distance.
        exponent : positive float, optional
            Exponent of the norm. For values other than 2.0, no inner
            product is defined.

        Notes
        -----
        - For exponent 2.0, a new weighted inner product with array
          :math:`w` is defined as

          .. math::
              \\langle a, b\\rangle_w :=
              \\langle w \odot a, b\\rangle =
              b^{\mathrm{H}} (w \odot a),

          where :math:`b^{\mathrm{H}}` stands for transposed complex
          conjugate and :math:`w \odot a` for entry-wise multiplication.

        - For other exponents, only norm and dist are defined. In the
          case of exponent :math:`\\infty`, the weighted norm is

          .. math::
              \| a\|_{w, \\infty} :=
              \| w \odot a\|_{\\infty},

          otherwise it is (using point-wise exponentiation)

          .. math::

              \| a\|_{w, p} :=
              \| w^{1/p} \odot a\|_{\\infty}.

        - Note that this definition does **not** fulfill the limit
          property in :math:`p`, i.e.

          .. math::
              \| a\|_{w, p} \\not\\to
              \| a\|_{w, \\infty} \quad (p \\to \\infty)

          unless :math:`w = (1, \dots, 1)`. The reason for this choice
          is that the alternative with the limit property consists in
          ignoring the weights altogether.

        - The array may only have positive entries, otherwise it does
          not define an inner product or norm, respectively. This is not
          checked during initialization.
        """
        super(NumpyFnArrayWeighting, self).__init__(
            array, impl='numpy', exponent=exponent)

    def inner(self, x1, x2):
        """Return the weighted inner product of two vectors.

        Parameters
        ----------
        x1, x2 : `NumpyFnVector`
            Vectors whose inner product is calculated

        Returns
        -------
        inner : float or complex
            The inner product of the two provided vectors.
        """
        if self.exponent != 2.0:
            raise NotImplementedError('no inner product defined for '
                                      'exponent != 2 (got {})'
                                      ''.format(self.exponent))
        else:
            inner = _inner_default(x1 * self.array, x2)
            if is_real_dtype(x1.dtype):
                return float(inner)
            else:
                return complex(inner)

    def norm(self, x):
        """Calculate the array-weighted norm of a vector.

        Parameters
        ----------
        x : `NumpyFnVector`
            Vector whose norm is calculated

        Returns
        -------
        norm : float
            The norm of the provided vector
        """
        if self.exponent == 2.0:
            norm_squared = self.inner(x, x).real  # TODO: optimize?!
            if norm_squared < 0:
                norm_squared = 0.0  # Compensate for numerical error
            return np.sqrt(norm_squared)
        else:
            return float(_pnorm_diagweight(x, self.exponent, self.array))


class NumpyFnConstWeighting(ConstWeighting):

    """Weighting of `NumpyFn` by a constant.

    This class defines a weighting with the same constant for each index.
    See ``Notes`` for mathematical details.
    """

    def __init__(self, constant, exponent=2.0):
        """Initialize a new instance.

        Parameters
        ----------
        constant : positive float
            Weighting constant of the inner product.
        exponent : positive float, optional
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.

        Notes
        -----
        - For exponent 2.0, a new weighted inner product with constant
          :math:`c` is defined as

          .. math::
              \\langle a, b\\rangle_c = c\, \\langle a, b\\rangle
              = c\, b^{\mathrm{H}} a,

          with :math:`b^{\mathrm{H}}` standing for transposed complex
          conjugate.

          For other exponents, only norm and dist are defined. In the case of
          exponent :math:`\infty`, the weighted norm is defined as

          .. math::
              \|a\|_{c, \infty} := c\, \|a\|_\infty,

          otherwise it is

          .. math::
              \|a\|_{c, p} := c^{1/p}\, \|a\|_p.

        - Note that this definition does **not** fulfill the limit property
          in ``p``, i.e.,

          .. math::
              \|a\|_{c,p} \\not\\to \|a\|_{c,\infty}
              \quad\\text{for } p \\to \infty

          unless :math:`c = 1`. The reason for this choice is that the
          alternative fulfilling the limit property consists in ignoring
          the weight altogether.

        - The constant must be positive, otherwise it does not define an
          inner product or norm, respectively.
        """
        super(NumpyFnConstWeighting, self).__init__(
            constant, impl='numpy', exponent=exponent)

    def inner(self, x1, x2):
        """Calculate the constant-weighted inner product of two vectors.

        Parameters
        ----------
        x1, x2 : `NumpyFnVector`
            Vectors whose inner product is calculated

        Returns
        -------
        inner : float or complex
            The inner product of the two provided vectors
        """
        if self.exponent != 2.0:
            raise NotImplementedError('no inner product defined for '
                                      'exponent != 2 (got {})'
                                      ''.format(self.exponent))
        else:
            inner = self.const * _inner_default(x1, x2)
            return x1.space.field.element(inner)

    def norm(self, x):
        """Calculate the constant-weighted norm of a vector.

        Parameters
        ----------
        x1 : `NumpyFnVector`
            Vector whose norm is calculated

        Returns
        -------
        norm : float
            The norm of the vector
        """
        if self.exponent == 2.0:
            return np.sqrt(self.const) * float(_norm_default(x))
        elif self.exponent == float('inf'):
            return self.const * float(_pnorm_default(x, self.exponent))
        else:
            return (self.const ** (1 / self.exponent) *
                    float(_pnorm_default(x, self.exponent)))

    def dist(self, x1, x2):
        """Calculate the constant-weighted distance between two vectors.

        Parameters
        ----------
        x1, x2 : `NumpyFnVector`
            Vectors whose mutual distance is calculated

        Returns
        -------
        dist : float
            The distance between the vectors
        """
        if self.exponent == 2.0:
            return np.sqrt(self.const) * _norm_default(x1 - x2)
        elif self.exponent == float('inf'):
            return self.const * float(_pnorm_default(x1 - x2, self.exponent))
        else:
            return (self.const ** (1 / self.exponent) *
                    float(_pnorm_default(x1 - x2, self.exponent)))


class NumpyFnNoWeighting(NoWeighting, NumpyFnConstWeighting):

    """Weighting of `NumpyFn` with constant 1.

    For exponent 2.0, the unweighted inner product is defined as::

        <a, b> := b^H a

    with ``b^H`` standing for transposed complex conjugate.

    For other exponents, only norm and dist are defined.
    """

    # Implement singleton pattern for efficiency in the default case
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern if ``exp==2.0``."""
        if len(args) == 0:
            exponent = kwargs.pop('exponent', 2.0)
        else:
            exponent = args[0]
            args = args[1:]

        if exponent == 2.0:
            if cls._instance is None:
                cls._instance = super(NoWeighting, cls).__new__(
                    cls, *args, **kwargs)
            return cls._instance
        else:
            return super(NoWeighting, cls).__new__(cls, *args, **kwargs)

    def __init__(self, exponent=2.0):
        """Initialize a new instance.

        Parameters
        ----------
        exponent : positive float, optional
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
        """
        super(NumpyFnNoWeighting, self).__init__(
            impl='numpy', exponent=exponent)


class NumpyFnCustomInner(CustomInner):

    """Class for handling a user-specified inner product in `NumpyFn`."""

    def __init__(self, inner):
        """Initialize a new instance.

        Parameters
        ----------
        inner : callable
            The inner product implementation. It must accept two
            `NumpyFnVector` arguments, return an element from their space's
            field (real or complex number) and satisfy the following
            conditions for all vectors ``x, y, z`` and scalars ``s``:

            - ``<x, y> = conj(<y, x>)``
            - ``<s*x + y, z> = s * <x, z> + <y, z>``
            - ``<x, x> = 0``  if and only if  ``x = 0``
        """
        super(NumpyFnCustomInner, self).__init__(inner, impl='numpy')


class NumpyFnCustomNorm(CustomNorm):

    """Class for handling a user-specified norm in `NumpyFn`.

    Note that this removes ``inner``.
    """

    def __init__(self, norm):
        """Initialize a new instance.

        Parameters
        ----------
        norm : callable
            The norm implementation. It must accept an `NumpyFnVector`
            argument, return a float and satisfy the following
            conditions for all vectors ``x, y`` and scalars ``s``:

            - ``||x|| >= 0``
            - ``||x|| = 0``  if and only if  ``x = 0``
            - ``||s * x|| = |s| * ||x||``
            - ``||x + y|| <= ||x|| + ||y||``
        """
        super(NumpyFnCustomNorm, self).__init__(norm, impl='numpy')


class NumpyFnCustomDist(CustomDist):

    """Class for handling a user-specified distance in `NumpyFn`.

    Note that this removes ``inner`` and ``norm``.
    """

    def __init__(self, dist):
        """Initialize a new instance.

        Parameters
        ----------
        dist : callable
            The distance function defining a metric on `NumpyFn`. It must
            accept two `NumpyFnVector` arguments, return a float and
            fulfill the following mathematical conditions for any three
            vectors ``x, y, z``:

            - ``dist(x, y) >= 0``
            - ``dist(x, y) = 0``  if and only if  ``x = y``
            - ``dist(x, y) = dist(y, x)``
            - ``dist(x, y) <= dist(x, z) + dist(z, y)``
        """
        super(NumpyFnCustomDist, self).__init__(dist, impl='numpy')


# --- Auxiliary functions --- #


def _blas_is_applicable(*args):
    """Whether BLAS routines can be applied or not.

    BLAS routines are available for single and double precision
    float or complex data only. If the arrays are non-contiguous,
    BLAS methods are usually slower, and array-writing routines do
    not work at all. Hence, only contiguous arrays are allowed.
    Furthermore, BLAS uses 32-bit integers internally for indexing,
    which makes it unusable for arrays lager than ``2 ** 31 - 1``.

    Parameters
    ----------
    x1,...,xN : `NtuplesBaseVector`
        The vectors to be tested for BLAS conformity
    """
    return (all(x.dtype == args[0].dtype and
                x.dtype in _BLAS_DTYPES and
                x.data.flags.contiguous and
                x.size <= np.iinfo('int32').max
                for x in args))


def _lincomb_impl(a, x1, b, x2, out, dtype):
    """Raw linear combination depending on data type."""
    # Convert to native since BLAS needs it
    size = native(x1.size)

    # Shortcut for small problems
    if size <= THRESHOLD_SMALL:  # small array optimization
        out.data[:] = a * x1.data + b * x2.data
        return

    # If data is very big, use BLAS if possible
    if size > THRESHOLD_MEDIUM and _blas_is_applicable(x1, x2, out):
        axpy, scal, copy = linalg.blas.get_blas_funcs(
            ['axpy', 'scal', 'copy'], arrays=(x1.data, x2.data, out.data))
    else:
        # Use fallbacks otherwise
        def fallback_axpy(x1, x2, n, a):
            """Fallback axpy implementation avoiding copy."""
            if a != 0:
                x2 /= a
                x2 += x1
                x2 *= a
            return x2

        def fallback_scal(a, x, n):
            """Fallback scal implementation."""
            x *= a
            return x

        def fallback_copy(x1, x2, n):
            """Fallback copy implementation."""
            x2[...] = x1[...]
            return x2

        axpy, scal, copy = (fallback_axpy, fallback_scal, fallback_copy)

    if x1 is x2 and b != 0:
        # x1 is aligned with x2 -> out = (a+b)*x1
        _lincomb_impl(a + b, x1, 0, x1, out, dtype)
    elif out is x1 and out is x2:
        # All the vectors are aligned -> out = (a+b)*out
        scal(a + b, out.data, size)
    elif out is x1:
        # out is aligned with x1 -> out = a*out + b*x2
        if a != 1:
            scal(a, out.data, size)
        if b != 0:
            axpy(x2.data, out.data, size, b)
    elif out is x2:
        # out is aligned with x2 -> out = a*x1 + b*out
        if b != 1:
            scal(b, out.data, size)
        if a != 0:
            axpy(x1.data, out.data, size, a)
    else:
        # We have exhausted all alignment options, so x1 != x2 != out
        # We now optimize for various values of a and b
        if b == 0:
            if a == 0:  # Zero assignment -> out = 0
                out.data[:] = 0
            else:  # Scaled copy -> out = a*x1
                copy(x1.data, out.data, size)
                if a != 1:
                    scal(a, out.data, size)
        else:
            if a == 0:  # Scaled copy -> out = b*x2
                copy(x2.data, out.data, size)
                if b != 1:
                    scal(b, out.data, size)

            elif a == 1:  # No scaling in x1 -> out = x1 + b*x2
                copy(x1.data, out.data, size)
                axpy(x2.data, out.data, size, b)
            else:  # Generic case -> out = a*x1 + b*x2
                copy(x2.data, out.data, size)
                if b != 1:
                    scal(b, out.data, size)
                axpy(x1.data, out.data, size, a)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
