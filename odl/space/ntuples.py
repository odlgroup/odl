# Copyright 2014, 2015 The ODL development group
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

"""CPU implementations of `n`-dimensional Cartesian spaces.

This is a default implementation of :math:`A^n` for an arbitrary set
:math:`A` as well as the real and complex spaces :math:`R^n` and
:math:`C^n`. The latter two each come in a basic version with vector
multiplication only and as metric, normed, Hilbert and Euclidean space
variants. The data is represented by NumPy arrays.

List of classes
---------------

+-------------+--------------+----------------------------------------+
|Class name   |Direct        |Description                             |
|             |Ancestors     |                                        |
+=============+==============+========================================+
|`Ntuples`    |`Set`         |Basic class of `n`-tuples where each    |
|             |              |entry is of the same type               |
+-------------+--------------+----------------------------------------+
|`Fn`         |`EuclideanCn` |`HilbertRn` with the standard inner     |
|             |              |(dot) product                           |
+-------------+--------------+----------------------------------------+
|`Cn`         |`Ntuples`,    |`n`-tuples of complex numbers with      |
|             |`Algebra`     |vector-vector multiplication            |
+-------------+--------------+----------------------------------------+
|`Rn`         |`Cn`          |`n`-tuples of real numbers with         |
|             |              |vector-vector multiplication            |
+-------------+--------------+----------------------------------------+
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import int, super
from future.utils import with_metaclass

# External module imports
from abc import ABCMeta, abstractmethod
from math import sqrt
import numpy as np
import scipy as sp
import ctypes
from scipy.linalg.blas import get_blas_funcs
from numbers import Integral
import platform

# ODL imports
from odl.operator.operator import LinearOperator
from odl.set.sets import Set, RealNumbers, ComplexNumbers
from odl.set.space import LinearSpace
from odl.util.utility import array1d_repr, array1d_str, dtype_repr
from odl.util.utility import is_real_dtype, is_complex_dtype


__all__ = ('NtuplesBase', 'FnBase', 'Ntuples', 'Fn', 'Cn', 'Rn',
           'MatVecOperator')


_TYPE_MAP_C2R = {np.dtype('float32'): np.dtype('float32'),
                 np.dtype('float64'): np.dtype('float64'),
                 np.dtype('complex64'): np.dtype('float32'),
                 np.dtype('complex128'): np.dtype('float64')}

_TYPE_MAP_R2C = {np.dtype('float32'): np.dtype('complex64'),
                 np.dtype('float64'): np.dtype('complex128')}

if platform.system() == 'Linux':
    _TYPE_MAP_C2R.update({np.dtype('float128'): np.dtype('float128'),
                          np.dtype('complex256'): np.dtype('float128')})
    _TYPE_MAP_R2C.update({np.dtype('float128'): np.dtype('complex256')})


_BLAS_DTYPES = (np.dtype('float32'), np.dtype('float64'),
                np.dtype('complex64'), np.dtype('complex128'))


class NtuplesBase(with_metaclass(ABCMeta, Set)):

    """Base class for sets of n-tuples independent of implementation."""

    def __init__(self, size, dtype):
        """Initialize a new instance.

        Parameters
        ----------
        size : int
            The number of entries per tuple
        dtype : object
            The data type for each tuple entry. Can be provided in any
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.
        """
        if not isinstance(size, Integral) or size < 0:
            raise TypeError('size {} is not a non-negative integer.'
                            ''.format(size))
        self._size = int(size)
        self._dtype = np.dtype(dtype)

    @property
    def dtype(self):
        """The data type of each entry."""
        return self._dtype

    @property
    def size(self):
        """The number of entries per tuple."""
        return self._size

    def __contains__(self, other):
        """`s.__contains__(other) <==> other in s`.

        Returns
        -------
        contains : bool
            `True` if `other` is an `NtuplesBase.Vector` instance and
            `other.space` is equal to this space, `False` otherwise.

        Examples
        --------
        >>> long_3 = Ntuples(3, dtype='int64')
        >>> long_3.element() in long_3
        True
        >>> long_3.element() in Ntuples(3, dtype='int32')
        False
        >>> long_3.element() in Ntuples(3, dtype='float64')
        False
        """
        return isinstance(other, NtuplesBase.Vector) and other.space == self

    def __eq__(self, other):
        """`s.__eq__(other) <==> s == other`.

        Returns
        -------
        equals : bool
            `True` if `other` is an instance of this space's type
            with the same `size` and `dtype`, otherwise `False`.

        Examples
        --------
        >>> int_3 = Ntuples(3, dtype=int)
        >>> int_3 == int_3
        True

        Equality is not identity:

        >>> int_3a, int_3b = Ntuples(3, int), Ntuples(3, int)
        >>> int_3a == int_3b
        True
        >>> int_3a is int_3b
        False

        >>> int_3, int_4 = Ntuples(3, int), Ntuples(4, int)
        >>> int_3 == int_4
        False
        >>> int_3, str_3 = Ntuples(3, 'int'), Ntuples(3, 'S2')
        >>> int_3 == str_3
        False
        """
        if other is self:
            return True

        return (isinstance(other, type(self)) and
                isinstance(self, type(other)) and
                self.size == other.size and
                self.dtype == other.dtype)

    def __repr__(self):
        """s.__repr__() <==> repr(s)."""
        return '{}({}, {})'.format(self.__class__.__name__, self.size,
                                   dtype_repr(self.dtype))

    def __str__(self):
        """s.__str__() <==> str(s)."""
        return '{}({}, {})'.format(self.__class__.__name__, self.size,
                                   dtype_repr(self.dtype))

    class Vector(with_metaclass(ABCMeta, object)):

        """Abstract class for representation of n-tuples.

        Defines abstract attributes and concrete ones which are
        independent of data representation.
        """

        def __init__(self, space, *args, **kwargs):
            """Initialize a new instance."""
            self._space = space

        @property
        def space(self):
            """Space to which this vector."""
            return self._space

        @property
        def ndim(self):
            """Number of dimensions, always 1."""
            return 1

        @property
        def dtype(self):
            """Length of this vector, equal to space size."""
            return self.space.dtype

        @property
        def size(self):
            """Length of this vector, equal to space size."""
            return self.space.size

        @property
        def shape(self):
            """Shape of this vector, equals `(size,)`."""
            return (self.space,)

        @abstractmethod
        def copy(self):
            """Create an identical (deep) copy of this vector."""

        @abstractmethod
        def asarray(self, start=None, stop=None, step=None):
            """Extract the data of this array as a numpy array.

            Parameters
            ----------
            start : int, optional (default: `None`)
                Start position. None means the first element.
            start : int, optional (default: `None`)
                One element past the last element to be extracted.
                None means the last element.
            start : int, optional (default: `None`)
                Step length. None means 1.

            Returns
            -------
            asarray : `ndarray`
                Numpy array of the same type as the space.
            """

        def __len__(self):
            """v.__len__() <==> len(v).

            Return the number of space dimensions.
            """
            return self.space.size

        @abstractmethod
        def __eq__(self, other):
            """`vec.__eq__(other) <==> vec == other`.

            Returns
            -------
            equals : bool
                `True` if all entries of `other` are equal to this
                vector's entries, `False` otherwise.
            """

        @abstractmethod
        def __getitem__(self, indices):
            """Access values of this vector.

            Parameters
            ----------
            indices : int or slice
                The position(s) that should be accessed

            Returns
            -------
            values : `space.dtype` or `space.Vector`
                The value(s) at the index (indices)
            """

        @abstractmethod
        def __setitem__(self, indices, values):
            """Set values of this vector.

            Parameters
            ----------
            indices : int or slice
                The position(s) that should be set
            values : {scalar, array-like, `Ntuples.Vector`}
                The value(s) that are to be assigned.

                If `index` is an integer, `value` must be single value.

                If `index` is a slice, `value` must be broadcastable
                to the size of the slice (same size, shape (1,)
                or single value).
            """

        def __ne__(self, other):
            """`vec.__ne__(other) <==> vec != other`."""
            return not self.__eq__(other)

        def __str__(self):
            """`vec.__str__() <==> str(vec)`."""
            return array1d_str(self)

        def __repr__(self):
            """`vec.__repr__() <==> repr(vec)`."""
            return '{!r}.element({})'.format(self.space,
                                             array1d_repr(self))


class Ntuples(NtuplesBase):

    """The set of `n`-tuples of arbitrary type.

    See also
    --------
    See the module documentation for attributes, methods etc.
    """

    def element(self, inp=None, data_ptr=None):
        """Create a new element.

        Parameters
        ----------
        inp : array-like or scalar, optional
            Input to initialize the new element.

            If `inp` is `None`, an empty element is created with no
            guarantee of its state (memory allocation only).

            If `inp` is a `numpy.ndarray` of shape `(size,)` and the
            same data type as this space, the array is wrapped, not
            copied.
            Other array-like objects are copied (with broadcasting
            if necessary).

            If a single value is given, it is copied to all entries.

        Returns
        -------
        element : `Ntuples.Vector`
            The new element created (from `inp`).

        Note
        ----
        This method preserves "array views" of correct size and type,
        see the examples below.

        Examples
        --------
        >>> strings3 = Ntuples(3, dtype='U1')  # 1-char strings
        >>> x = strings3.element(['w', 'b', 'w'])
        >>> print(x)
        [w, b, w]
        >>> x.space
        Ntuples(3, '<U1')

        Construction from data pointer:

        >>> int3 = Ntuples(3, dtype='int')
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
                return self.Vector(self, arr)
            else:
                ctype_array_def = ctypes.c_byte * (self.size *
                                                   self.dtype.itemsize)
                as_ctype_array = ctype_array_def.from_address(data_ptr)
                as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
                arr = as_numpy_array.view(dtype=self.dtype)
                return self.Vector(self, arr)
        else:
            if data_ptr is None:
                if isinstance(inp, Ntuples.Vector):
                    inp = inp.data.astype(self.dtype, copy=True)
                else:
                    inp = np.atleast_1d(inp).astype(self.dtype, copy=False)

                if inp.shape == (1,):
                    arr = np.empty(self.size, dtype=self.dtype)
                    arr[:] = inp
                elif inp.shape == (self.size,):
                    arr = inp
                else:
                    raise ValueError('input shape {} not broadcastable to '
                                     'shape ({},).'.format(inp.shape,
                                                           self.size))

                return self.Vector(self, arr)
            else:
                raise ValueError('Cannot provide both `inp` and `data_ptr`')

    class Vector(NtuplesBase.Vector):

        """Representation of an `Ntuples` element.

        See also
        --------
        See the module documentation for attributes, methods etc.
        """

        def __init__(self, space, data):
            """Initialize a new instance."""
            if not isinstance(space, Ntuples):
                raise TypeError('{!r} not an `Ntuples` instance.'
                                ''.format(space))

            if not isinstance(data, np.ndarray):
                raise TypeError('data {!r} not a `numpy.ndarray` instance.'
                                ''.format(data))

            if data.dtype != space.dtype:
                raise TypeError('data {!r} not of dtype `{!r}`.'
                                ''.format(data, space.dtype))

            self._data = data

            super().__init__(space)

        @property
        def data(self):
            """The raw numpy array representing the data."""
            return self._data

        def asarray(self, start=None, stop=None, step=None, out=None):
            """Extract the data of this array as a numpy array.

            Parameters
            ----------
            start : int, optional (default: `None`)
                Start position. None means the first element.
            start : int, optional (default: `None`)
                One element past the last element to be extracted.
                None means the last element.
            start : int, optional (default: `None`)
                Step length. None means 1.
            out : `ndarray`, optional (default: `None`)
                Array in which the result should be written in-place.
                Has to be contiguous and of the correct dtype.

            Returns
            -------
            asarray : `ndarray`
                Numpy array of the same type as the space.

            Examples
            --------
            >>> import ctypes
            >>> vec = Ntuples(3, 'float').element([1, 2, 3])
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
            >>> vec = Ntuples(3, 'int32').element([1, 2, 3])
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
            return self._data.ctypes.data

        def __eq__(self, other):
            """`vec.__eq__(other) <==> vec == other`.

            Returns
            -------
            equals :  bool
                `True` if all entries of `other` are equal to this
                vector's entries, `False` otherwise.

            Note
            ----
            Space membership is not checked, hence vectors from
            different spaces can be equal.

            Examples
            --------
            >>> vec1 = Ntuples(3, int).element([1, 2, 3])
            >>> vec2 = Ntuples(3, int).element([-1, 2, 0])
            >>> vec1 == vec2
            False
            >>> vec2 = Ntuples(3, int).element([1, 2, 3])
            >>> vec1 == vec2
            True

            Space membership matters:

            >>> vec2 = Ntuples(3, float).element([1, 2, 3])
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

            Returns
            -------
            copy : `Ntuples.Vector`
                The deep copy

            Examples
            --------
            >>> vec1 = Ntuples(3, 'int').element([1, 2, 3])
            >>> vec2 = vec1.copy()
            >>> vec2
            Ntuples(3, 'int').element([1, 2, 3])
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
            indices : int or slice
                The position(s) that should be accessed

            Returns
            -------
            values : `space.dtype` or `space.Vector`
                The value(s) at the index (indices)


            Examples
            --------
            >>> str_3 = Ntuples(3, dtype='U6')  # 6-char unicode
            >>> x = str_3.element(['a', 'Hello!', '0'])
            >>> print(x[0])
            a
            >>> print(x[1:3])
            [Hello!, 0]
            >>> x[1:3].space
            Ntuples(2, '<U6')
            """
            try:
                return self.data[int(indices)]  # single index
            except TypeError:
                arr = self.data[indices]
                return type(self.space)(len(arr),
                                        dtype=self.space.dtype).element(arr)

        def __setitem__(self, indices, values):
            """Set values of this vector.

            Parameters
            ----------
            indices : int or slice
                The position(s) that should be set
            values : {scalar, array-like, `Ntuples.Vector`}
                The value(s) that are to be assigned.

                If `indices` is an integer, `value` must be single value.

                If `indices` is a slice, `value` must be
                broadcastable to the size of the slice (same size,
                shape (1,) or single value).

            Returns
            -------
            None

            Examples
            --------
            >>> int_3 = Ntuples(3, 'int')
            >>> x = int_3.element([1, 2, 3])
            >>> x[0] = 5
            >>> x
            Ntuples(3, 'int').element([5, 2, 3])

            Assignment from array-like structures or another
            vector:

            >>> y = Ntuples(2, 'short').element([-1, 2])
            >>> x[:2] = y
            >>> x
            Ntuples(3, 'int').element([-1, 2, 3])
            >>> x[1:3] = [7, 8]
            >>> x
            Ntuples(3, 'int').element([-1, 7, 8])
            >>> x[:] = np.array([0, 0, 0])
            >>> x
            Ntuples(3, 'int').element([0, 0, 0])

            Broadcasting is also supported:

            >>> x[1:3] = -2.
            >>> x
            Ntuples(3, 'int').element([0, -2, -2])

            Array views are preserved:

            >>> y = x[::2]  # view into x
            >>> y[:] = -9
            >>> print(y)
            [-9, -9]
            >>> print(x)
            [-9, -2, -9]

            Be aware of unsafe casts and over-/underflows, there
            will be warnings at maximum.

            >>> x = Ntuples(2, 'int8').element([0, 0])
            >>> maxval = 255  # maximum signed 8-bit unsigned int
            >>> x[0] = maxval + 1
            >>> x
            Ntuples(2, 'int8').element([0, 0])
            """
            if isinstance(values, Ntuples.Vector):
                return self.data.__setitem__(indices, values.data)
            else:
                return self.data.__setitem__(indices, values)


class FnBase(with_metaclass(ABCMeta, NtuplesBase, LinearSpace)):

    """Base class for :math:`F^n` independent of implementation."""

    def __init__(self, size, dtype):
        """Initialize a new instance.

        Parameters
        ----------
        size : int
            The number of dimensions of the space
        dtype : object
            The data type of the storage array. Can be provided in any
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.
            Only scalar data types (numbers) are allowed.
        """
        super().__init__(size, dtype)
        if not np.issubsctype(self._dtype, np.number):
            raise TypeError('{} not a scalar data type.'.format(dtype))

        if is_real_dtype(self._dtype):
            self._field = RealNumbers()
        else:
            self._field = ComplexNumbers()

    @abstractmethod
    def zero(self):
        """Create a vector of zeros."""

    @property
    def field(self):
        """The field of this space."""
        return self._field

    @abstractmethod
    def _multiply(self, z, x1, x2):
        """The entry-wise product of two vectors, assigned to `z`."""

    class Vector(with_metaclass(ABCMeta, NtuplesBase.Vector,
                                LinearSpace.Vector)):

        """Abstract class for representation of :math:`F^n` vectors.

        Defines abstract attributes and concrete ones which are
        independent of data representation.
        """


def _blas_is_applicable(*args):
    """Whether BLAS routines can be applied or not.

    BLAS routines are available for single and double precision
    float or complex data only. If the arrays are non-contiguous,
    BLAS methods are usually slower, and array-writing routines do
    not work at all. Hence, only contiguous arrays are allowed.
    """
    if len(args) == 0:
        return False

    return (all(x.dtype == args[0].dtype for x in args) and
            all(x.dtype in _BLAS_DTYPES for x in args) and
            all(x.data.flags.contiguous for x in args))


def _lincomb(z, a, x1, b, x2, dtype):
    """Raw linear combination depending on data type."""
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

    if _blas_is_applicable(x1, x2, z):
        # pylint: disable=unbalanced-tuple-unpacking
        axpy, scal, copy = get_blas_funcs(
            ['axpy', 'scal', 'copy'], arrays=(x1.data, x2.data))
    else:
        axpy, scal, copy = (fallback_axpy, fallback_scal, fallback_copy)

    if x1 is x2 and b != 0:
        # x1 is aligned with x2 -> z = (a+b)*x1
        _lincomb(z, a+b, x1, 0, x1, dtype)
    elif z is x1 and z is x2:
        # All the vectors are aligned -> z = (a+b)*z
        scal(a+b, z.data, len(z))
    elif z is x1:
        # z is aligned with x1 -> z = a*z + b*x2
        if a != 1:
            scal(a, z.data, len(z))
        if b != 0:
            axpy(x2.data, z.data, len(z), b)
    elif z is x2:
        # z is aligned with x2 -> z = a*x1 + b*z
        if b != 1:
            scal(b, z.data, len(z))
        if a != 0:
            axpy(x1.data, z.data, len(z), a)
    else:
        # We have exhausted all alignment options, so x1 != x2 != z
        # We now optimize for various values of a and b
        if b == 0:
            if a == 0:  # Zero assignment -> z = 0
                z.data[:] = 0
            else:  # Scaled copy -> z = a*x1
                copy(x1.data, z.data, len(z))
                if a != 1:
                    scal(a, z.data, len(z))
        else:
            if a == 0:  # Scaled copy -> z = b*x2
                copy(x2.data, z.data, len(z))
                if b != 1:
                    scal(b, z.data, len(z))

            elif a == 1:  # No scaling in x1 -> z = x1 + b*x2
                copy(x1.data, z.data, len(z))
                axpy(x2.data, z.data, len(z), b)
            else:  # Generic case -> z = a*x1 + b*x2
                copy(x2.data, z.data, len(z))
                if b != 1:
                    scal(b, z.data, len(z))
                axpy(x1.data, z.data, len(z), a)


class Fn(FnBase, Ntuples):

    """The vector space :math:`F^n` with vector multiplication.

    This space implements n-tuples of elements from a field :math:`F`,
    which can be the real or the complex numbers.

    Its elements are represented as instances of the inner `Fn.Vector`
    class.

    See also
    --------
    See the module documentation for attributes, methods etc.
    """

    def __init__(self, size, dtype, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        size : positive int
            The number of dimensions of the space
        dtype : object
            The data type of the storage array. Can be provided in any
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.

            Only scalar data types are allowed.

        kwargs : {'weight', 'dist', 'norm', 'inner', 'dist_using_inner'}
            'weight' : matrix, float or `None`
                Use weighted inner product, norm, and dist.

                `None` (default) : Use the standard unweighted functions

                float : Use functions weighted by a constant

                matrix : Use functions weighted by a matrix. The matrix
                can be dense (`numpy.matrix`) or sparse
                (`scipy.sparse.spmatrix`).

                This option cannot be combined with `dist`, `norm` or
                `inner`.

            'dist' : callable, optional
                The distance function defining a metric on :math:`F^n`.
                It must accept two `Fn.Vector` arguments and fulfill the
                following conditions for any vectors `x`, `y` and `z`:

                - `dist(x, y) == dist(y, x)`
                - `dist(x, y) >= 0`
                - `dist(x, y) == 0` (approx.) if and only if `x == y`
                  (approx.)
                - `dist(x, y) <= dist(x, z) + dist(z, y)`

                By default, `dist(x, y)` is calculated as `norm(x - y)`.
                This creates an intermediate array `x-y`, which can be
                avoided by choosing `dist_using_inner=True`.

                This option cannot be combined with `weight`, `norm`
                or `inner`.

            'norm' : callable, optional (Default: `sqrt(inner(x,y))`)
                The norm implementation. It must accept an `Fn.Vector`
                argument, return a `RealNumber` and satisfy the
                following properties:

                - `norm(x) >= 0`
                - `norm(x) == 0` (approx.) only if `x == 0` (approx.)
                - `norm(s * x) == abs(s) * norm(x)` for `s` scalar
                - `norm(x + y) <= norm(x) + norm(y)`

                This option cannot be combined with `weight`, `dist`
                or `inner`.

            'inner' : callable, optional
                The inner product implementation. It must accept two
                `Fn.Vector` arguments, return a complex number and
                satisfy the following conditions for all vectors `x`,
                `y` and `z` and scalars `s`:

                 - `inner(x, y) == conjugate(inner(y, x))`
                 - `inner(s * x, y) == s * inner(x, y)`
                 - `inner(x + z, y) == inner(x, y) + inner(z, y)`
                 - `inner(x, x) == 0` (approx.) only if `x == 0`
                   (approx.)

                This option cannot be combined with `weight`, `dist`
                or `norm`.

            'dist_using_inner' : bool, optional  (Default: `False`)
                Calculate `dist(x, y)` as

                `sqrt(norm(x)**2 + norm(y)**2 - 2 * inner(x, y).real)`

                This avoids the creation of new arrays and is thus
                faster for large arrays. On the downside, it will not
                evaluate to exactly zero for equal (but not identical)
                `x` and `y`.
        """
        super().__init__(size, dtype)

        dist = kwargs.pop('dist', None)
        norm = kwargs.pop('norm', None)
        inner = kwargs.pop('inner', None)
        weight = kwargs.pop('weight', None)
        dist_using_inner = bool(kwargs.pop('dist_using_inner', False))

        # Check validity of option combination (3 or 4 out of 4 must be None)
        if (dist, norm, inner, weight).count(None) < 3:
            raise ValueError('invalid combination of options `weight`, '
                             '`dist`, `norm` and `inner`.')
        if weight is not None:
            if np.isscalar(weight):
                self._space_funcs = FnConstWeighting(
                    weight, dist_using_inner=dist_using_inner)
            elif isinstance(weight, (np.matrix, sp.sparse.spmatrix)):
                self._space_funcs = FnMatrixWeighting(
                    weight, dist_using_inner=dist_using_inner)
            elif weight is None:
                pass
            else:
                raise ValueError('invalid weight argument {!r}.'
                                 ''.format(weight))
        elif dist is not None:
            self._space_funcs = FnCustomDist(dist)
        elif norm is not None:
            self._space_funcs = FnCustomNorm(norm)
        elif inner is not None:
            self._space_funcs = FnCustomInnerProduct(inner)
        else:  # all None -> no weighing
            self._space_funcs = FnNoWeighting()

    def _lincomb(self, z, a, x1, b, x2):
        """Linear combination of `x` and `y`.

        Calculate `z = a * x1 + b * x2` using optimized BLAS routines if
        possible.

        Parameters
        ----------
        z : `Fn.Vector`
            The Vector that the result is written to.
        a, b : `field` element
            Scalar to multiply `x` and `y` with.
        x1, x2 : `Fn.Vector`
            The summands

        Returns
        -------
        None

        Examples
        --------
        >>> c3 = Cn(3)
        >>> x = c3.element([1+1j, 2-1j, 3])
        >>> y = c3.element([4+0j, 5, 6+0.5j])
        >>> z = c3.element()
        >>> c3.lincomb(z, 2j, x, 3-1j, y)
        >>> z
        Cn(3).element([(10-2j), (17-1j), (18.5+1.5j)])
        """
        _lincomb(z, a, x1, b, x2, self.dtype)

    def _dist(self, x1, x2):
        """Calculate the distance between two vectors.

        Parameters
        ----------
        x1, x2 : `Fn.Vector`
            The vectors whose mutual distance is calculated

        Returns
        -------
        dist : float
            Distance between the vectors

        Examples
        --------
        >>> from numpy.linalg import norm
        >>> c2_2 = Cn(2, dist=lambda x, y: norm(x - y, ord=2))
        >>> x = c2_2.element([3+1j, 4])
        >>> y = c2_2.element([1j, 4-4j])
        >>> c2_2.dist(x, y)
        5.0

        >>> c2_2 = Cn(2, dist=lambda x, y: norm(x - y, ord=1))
        >>> x = c2_2.element([3+1j, 4])
        >>> y = c2_2.element([1j, 4-4j])
        >>> c2_2.dist(x, y)
        7.0
        """
        return self._space_funcs.dist(x1, x2)

    def _norm(self, x):
        """Calculate the norm of a vector.

        Parameters
        ----------
        x : `Fn.Vector`
            The vector whose norm is calculated

        Returns
        -------
        norm : float
            Norm of the vector

        Examples
        --------
        >>> import numpy as np
        >>> c2_2 = Cn(2, norm=np.linalg.norm)  # 2-norm
        >>> x = c2_2.element([3+1j, 1-5j])
        >>> c2_2.norm(x)
        6.0

        >>> from functools import partial
        >>> c2_1 = Cn(2, norm=partial(np.linalg.norm, ord=1))
        >>> x = c2_1.element([3-4j, 12+5j])
        >>> c2_1.norm(x)
        18.0
        """
        return self._space_funcs.norm(x)

    def _inner(self, x1, x2):
        """Raw inner product of two vectors.

        Parameters
        ----------

        x1, x2 : `Cn.Vector`
            The vectors whose inner product is calculated

        Returns
        -------
        inner : `complex`
            Inner product of `x1` and `x2`.

        Examples
        --------
        >>> import numpy as np
        >>> c3 = Cn(2, inner=lambda x, y: np.vdot(y, x))
        >>> x = c3.element([5+1j, -2j])
        >>> y = c3.element([1, 1+1j])
        >>> c3.inner(x, y) == (5+1j)*1 + (-2j)*(1-1j)
        True
        >>> weights = np.array([1., 2.])
        >>> c3w = Cn(2, inner=lambda x, y: np.vdot(weights * y, x))
        >>> x = c3w.element(x)  # elements must be cast (no copy)
        >>> y = c3w.element(y)
        >>> c3w.inner(x, y) == 1*(5+1j)*1 + 2*(-2j)*(1-1j)
        True
        """
        return self._space_funcs.inner(x1, x2)

    def _multiply(self, z, x1, x2):
        """The entry-wise product of two vectors, assigned to `z`.

        Parameters
        ----------
        z : `Cn.Vector`
            The result vector
        x1, x2 : `Cn.Vector`
            Factors in the product

        Returns
        -------
        None

        Examples
        --------
        >>> c3 = Cn(3)
        >>> x = c3.element([5+1j, 3, 2-2j])
        >>> y = c3.element([1, 2+1j, 3-1j])
        >>> z = c3.element()
        >>> c3.multiply(z, x, y)
        >>> z
        Cn(3).element([(5+1j), (6+3j), (4-8j)])
        """
        if z is x1 and z is x2:  # z = z*z
            z.data[:] *= z.data
        elif z is x1:  # z = z*x2
            z.data[:] *= x2.data
        elif z is x2:  # z = z*x1
            z.data[:] *= x1.data
        else:  # z = x1*x2
            z.data[:] = x1.data
            z.data[:] *= x2.data

    def zero(self):
        """Create a vector of zeros.

        Examples
        --------
        >>> c3 = Cn(3)
        >>> x = c3.zero()
        >>> x
        Cn(3).element([0j, 0j, 0j])
        """
        return self.element(np.zeros(self.size, dtype=self.dtype))

    def __eq__(self, other):
        """`s.__eq__(other) <==> s == other`.

        Returns
        -------
        equals : bool
            `True` if `other` is an instance of this space's type
            with the same `size` and `dtype`, and **identical**
            distance function, otherwise `False`.

        Examples
        --------
        >>> from numpy.linalg import norm
        >>> def dist(x, y, ord):
        ...     return norm(x - y, ord)

        >>> from functools import partial
        >>> dist2 = partial(dist, ord=2)
        >>> c3 = Cn(3, dist=dist2)
        >>> c3_same = Cn(3, dist=dist2)
        >>> c3  == c3_same
        True

        Different `dist` functions result in different spaces - the
        same applies for `norm` and `inner`:

        >>> dist1 = partial(dist, ord=1)
        >>> c3_1 = Cn(3, dist=dist1)
        >>> c3_2 = Cn(3, dist=dist2)
        >>> c3_1 == c3_2
        False

        Be careful with Lambdas - they result in non-identical function
        objects:

        >>> c3_lambda1 = Cn(3, dist=lambda x, y: norm(x-y, ord=1))
        >>> c3_lambda2 = Cn(3, dist=lambda x, y: norm(x-y, ord=1))
        >>> c3_lambda1 == c3_lambda2
        False

        An `Fn` space with the same data type is considered equal:

        >>> c3 = Cn(3)
        >>> f3_cdouble = Fn(3, dtype='complex128')
        >>> c3 == f3_cdouble
        True
        """
        if other is self:
            return True

        return (isinstance(other, Fn) and
                self.size == other.size and
                self.dtype == other.dtype and
                self.field == other.field and
                self._space_funcs == other._space_funcs)

    def __repr__(self):
        """s.__repr__() <==> repr(s)."""
        inner_fstr = '{}, {}'
        if self._space_funcs._dist_using_inner:
            inner_fstr += ', dist_using_inner=True'
        if isinstance(self._space_funcs, FnCustomInnerProduct):
            inner_fstr += ', inner=<custom inner>'
        elif isinstance(self._space_funcs, FnCustomNorm):
            inner_fstr += ', norm=<custom norm>'
        elif isinstance(self._space_funcs, FnCustomDist):
            inner_fstr += ', norm=<custom dist>'
        elif isinstance(self._space_funcs, FnConstWeighting):
            weight = self._space_funcs.const
            if weight != 1.0:
                inner_fstr += ', weight={weight}'
        elif isinstance(self._space_funcs, FnMatrixWeighting):
            inner_fstr += ', weight={weight!r}'
            weight = self._space_funcs.matrix

        inner_str = inner_fstr.format(self.size, dtype_repr(self.dtype),
                                      weight=weight)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    class Vector(FnBase.Vector, Ntuples.Vector):

        """Representation of an `Fn` element.

        See also
        --------
        See the module documentation for attributes, methods etc.
        """

        def __init__(self, space, data):
            """Initialize a new instance."""
            if not isinstance(space, Fn):
                raise TypeError('{!r} not an `Fn` instance.'
                                ''.format(space))

            if not isinstance(data, np.ndarray):
                raise TypeError('data {!r} not a `numpy.ndarray` instance.'
                                ''.format(data))
            super().__init__(space, data)


class Cn(Fn):

    """The complex vector space :math:`C^n` with vector multiplication.

    Its elements are represented as instances of the inner `Cn.Vector`
    class.

    See also
    --------
    Fn : n-tuples over a field :math:`F` with arbitrary scalar data type
    """

    def __init__(self, size, dtype=np.complex128, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        size : positive int
            The number of dimensions of the space
        dtype : object
            The data type of the storage array. Can be provided in any
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.

            Only complex floating-point data types are allowed.
        kwargs : {'weight', 'dist', 'norm', 'inner', 'dist_using_inner'}
            See `Fn`
        """
        super().__init__(size, dtype, **kwargs)

        if not is_complex_dtype(self._dtype):
            raise TypeError('data type {} not a complex floating-point type.'
                            ''.format(dtype))
        self._real_dtype = _TYPE_MAP_C2R[self._dtype]

    @property
    def real_dtype(self):
        """The corresponding real data type of this space."""
        return self._real_dtype

    def __repr__(self):
        """s.__repr__() <==> repr(s)."""
        inner_fstr = '{}'
        if self.dtype != np.complex128:
            inner_fstr += ', {dtype}'
        if self._space_funcs._dist_using_inner:
            inner_fstr += ', dist_using_inner=True'
        if isinstance(self._space_funcs, FnCustomInnerProduct):
            inner_fstr += ', inner=<custom inner>'
        elif isinstance(self._space_funcs, FnCustomNorm):
            inner_fstr += ', norm=<custom norm>'
        elif isinstance(self._space_funcs, FnCustomDist):
            inner_fstr += ', norm=<custom dist>'
        elif isinstance(self._space_funcs, FnConstWeighting):
            weight = self._space_funcs.const
            if weight != 1.0:
                inner_fstr += ', weight={weight}'
        elif isinstance(self._space_funcs, FnMatrixWeighting):
            inner_fstr += ', weight={weight!r}'
            weight = self._space_funcs.matrix

        inner_str = inner_fstr.format(self.size, dtype=dtype_repr(self.dtype),
                                      weight=weight)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`cn.__str__() <==> str(cn)`."""
        if self.dtype == np.complex128:
            return 'Cn({})'.format(self.size)
        else:
            return 'Cn({}, {})'.format(self.size, self.dtype)

    class Vector(Fn.Vector):
        """Representation of a `Cn` element.

        See also
        --------
        See the module documentation for attributes, methods etc.
        """

        @property
        def real(self):
            """The real part of this vector.

            Returns
            -------
            real : `Rn.Vector` view
                The real part this vector as a vector in `Rn`

            Examples
            --------
            >>> c3 = Cn(3)
            >>> x = c3.element([5+1j, 3, 2-2j])
            >>> x.real
            Rn(3).element([5.0, 3.0, 2.0])

            The `Rn` vector is really a view, so changes affect
            the original array:

            >>> x.real *= 2
            >>> x
            Cn(3).element([(10+1j), (6+0j), (4-2j)])
            """
            rn = Rn(self.space.size, self.space.real_dtype)
            return rn.element(self.data.real)

        @real.setter
        def real(self, newreal):
            """The setter for the real part.

            This method is invoked by `vec.real = other`.

            Parameters
            ----------
            newreal : array-like or scalar
                The new real part for this vector.

            Examples
            --------
            >>> c3 = Cn(3)
            >>> x = c3.element([5+1j, 3, 2-2j])
            >>> a = Rn(3).element([0, 0, 0])
            >>> x.real = a
            >>> x
            Cn(3).element([1j, 0j, -2j])

            Other array-like types and broadcasting:

            >>> x.real = 1.0
            >>> x
            Cn(3).element([(1+1j), (1+0j), (1-2j)])
            >>> x.real = [0, 2, -1]
            >>> x
            Cn(3).element([1j, (2+0j), (-1-2j)])
            """
            self.real.data[:] = newreal

        @property
        def imag(self):
            """The imaginary part of this vector.

            Returns
            -------
            imag : `Rn.Vector`
                The imaginary part this vector as a vector in `Rn`

            Examples
            --------
            >>> c3 = Cn(3)
            >>> x = c3.element([5+1j, 3, 2-2j])
            >>> x.imag
            Rn(3).element([1.0, 0.0, -2.0])

            The `Rn` vector is really a view, so changes affect
            the original array:

            >>> x.imag *= 2
            >>> x
            Cn(3).element([(5+2j), (3+0j), (2-4j)])
            """
            rn = Rn(self.space.size, self.space.real_dtype)
            return rn.element(self.data.imag)

        @imag.setter
        def imag(self, newimag):
            """The setter for the imaginary part.

            This method is invoked by `vec.imag = other`.

            Parameters
            ----------
            newreal : array-like or scalar
                The new imaginary part for this vector.

            Examples
            --------
            >>> x = Cn(3).element([5+1j, 3, 2-2j])
            >>> a = Rn(3).element([0, 0, 0])
            >>> x.imag = a; print(x)
            [(5+0j), (3+0j), (2+0j)]

            Other array-like types and broadcasting:

            >>> x.imag = 1.0; print(x)
            [(5+1j), (3+1j), (2+1j)]
            >>> x.imag = [0, 2, -1]; print(x)
            [(5+0j), (3+2j), (2-1j)]
            """
            self.imag.data[:] = newimag


class Rn(Fn):

    """The real vector space :math:`R^n` with vector multiplication.

    Its elements are represented as instances of the inner `Rn.Vector`
    class.

    See also
    --------
    Fn : n-tuples over a field :math:`F` with arbitrary scalar data type
    """

    def __init__(self, size, dtype=np.float64, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        size : positive int
            The number of dimensions of the space
        dtype : object
            The data type of the storage array. Can be provided in any
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.

            Only real floating-point data types are allowed.
        kwargs : {'weight', 'dist', 'norm', 'inner', 'dist_using_inner'}
            See `Fn`
        """
        super().__init__(size, dtype, **kwargs)

        if not is_real_dtype(self._dtype):
            raise TypeError('data type {} not a real floating-point type.'
                            ''.format(dtype))

    def __repr__(self):
        """s.__repr__() <==> repr(s)."""
        inner_fstr = '{}'
        if self.dtype != np.float64:
            inner_fstr += ', {dtype}'
        if self._space_funcs._dist_using_inner:
            inner_fstr += ', dist_using_inner=True'
        if isinstance(self._space_funcs, FnCustomInnerProduct):
            inner_fstr += ', inner=<custom inner>'
        elif isinstance(self._space_funcs, FnCustomNorm):
            inner_fstr += ', norm=<custom norm>'
        elif isinstance(self._space_funcs, FnCustomDist):
            inner_fstr += ', norm=<custom dist>'
        elif isinstance(self._space_funcs, FnConstWeighting):
            weight = self._space_funcs.const
            if weight != 1.0:
                inner_fstr += ', weight={weight}'
        elif isinstance(self._space_funcs, FnMatrixWeighting):
            inner_fstr += ', weight={weight!r}'
            weight = self._space_funcs.matrix

        inner_str = inner_fstr.format(self.size, dtype=dtype_repr(self.dtype),
                                      weight=weight)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`rn.__str__() <==> str(rn)`."""
        if self.dtype == np.float64:
            return 'Rn({})'.format(self.size)
        else:
            return 'Rn({}, {})'.format(self.size, self.dtype)


class MatVecOperator(LinearOperator):

    """Operator :math:`F^n -> F^m` represented by a matrix."""

    def __init__(self, dom, ran, matrix):
        """Initialize a new instance.

        Parameters
        ----------
        dom : `Fn`
            Space on whose elements the matrix acts. Its dtype must be
            castable to the range dtype.
        ran : `Fn`
            Space to which the matrix maps
        matrix : array-like or scipy.sparse.spmatrix
            Matrix representing the linear operator. Its shape must be
            `(m, n)`, where `n` is the size of `dom` and `m` the size
            of `ran`. Its dtype must be castable to the range dtype.
        """
        super().__init__(dom, ran)
        if not isinstance(dom, Fn):
            raise TypeError('domain {!r} is not an `Fn` instance.'
                            ''.format(dom))
        if not isinstance(ran, Fn):
            raise TypeError('range {!r} is not an `Fn` instance.'
                            ''.format(ran))
        if not np.can_cast(dom.dtype, ran.dtype):
            raise TypeError('domain data type {} cannot be safely cast to '
                            'range data type {}.'
                            ''.format(dom.dtype, ran.dtype))

        if isinstance(matrix, sp.sparse.spmatrix):
            self._matrix = matrix
        else:
            self._matrix = np.asmatrix(matrix)

        if self._matrix.shape != (ran.size, dom.size):
            raise ValueError('matrix shape {} does not match the required '
                             'shape {} of a matrix {} --> {}.'
                             ''.format(self._matrix.shape,
                                       (ran.size, dom.size),
                                       dom, ran))
        if not np.can_cast(self._matrix.dtype, ran.dtype):
            raise TypeError('matrix data type {} cannot be safely cast to '
                            'range data type {}.'
                            ''.format(matrix.dtype, ran.dtype))

    @property
    def matrix(self):
        """Matrix representing this operator."""
        return self._matrix

    @property
    def matrix_issparse(self):
        """Whether the representing matrix is sparse or not."""
        return isinstance(self.matrix, sp.sparse.spmatrix)

    @property
    def adjoint(self):
        """Adjoint operator represented by the adjoint matrix."""
        if self.domain.field != self.range.field:
            raise NotImplementedError('adjoint not defined since fields '
                                      'of domain and range differ ({} != {}).'
                                      ''.format(self.domain.field,
                                                self.range.field))
        return MatVecOperator(self.range, self.domain, self.matrix.H)

    def _call(self, inp):
        """Raw call method on input, producing a new output."""
        return self.range.element(
            np.asarray(self.matrix.dot(inp.data)).squeeze())

    def _apply(self, inp, outp):
        """Raw apply method on input, writing to given output."""
        if self.matrix_issparse:
            # Unfortunately, there is no native in-place dot product for
            # sparse matrices
            outp.data[:] = np.asarray(self.matrix.dot(inp.data)).squeeze()
        else:
            self.matrix.dot(inp.data, out=outp.data)

    # TODO: repr and str


class FnWeightingBase(with_metaclass(ABCMeta, object)):

    """Abstract base class for weighting of `FnBase` spaces.

    This class and its subclasses serve as a simple means to evaluate
    and compare weighted inner products, norms and metrics semantically
    rather than by identity on a pure function level.

    The functions are implemented similarly to `Operator` but without
    extra type checks of input parameters - this is done in the callers
    of the `LinearSpace` instance where these functions used.
    """

    def __init__(self, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        dist_using_inner : bool, optional
            Calculate `dist` using the formula

            norm(x-y)**2 = norm(x)**2 + norm(y)**2 - 2*inner(x, y).real

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) `x` and `y`.
        """
        self._dist_using_inner = bool(dist_using_inner)

    @abstractmethod
    def __eq__(self, other):
        """`w.__eq__(other) <==> w == other`.

        Returns
        -------
        equal : bool
            `True` if `other` is a `FnWeightingBase` instance
            represented by the **identical** matrix, `False` otherwise.

        Notes
        -----
        This operation must be computationally cheap, i.e. no large
        arrays may be compared element-wise. That is the task of the
        `equiv` method.
        """

    def equiv(self, other):
        """Test if `other` is an equivalent inner product.

        Returns
        -------
        equivalent : bool
            `True` if `other` is a `FnWeightingBase` instance which
            yields the same result as this inner product for any
            input, `False` otherwise. This is checked by entry-wise
            comparison of this instance's matrix with the matrix of
            `other`.
        """
        raise NotImplementedError

    @abstractmethod
    def inner(self, x1, x2):
        """Calculate the inner product of two vectors.

        Parameters
        ----------
        x1, x2 : `FnBase.Vector`
            Vectors whose inner product is calculated

        Returns
        -------
        inner : float or complex
            The inner product of the two provided vectors
        """

    def norm(self, x):
        """Calculate the norm of a vector.

        This is the standard implementation using `inner`. Subclasses
        should override it for optimization purposes.

        Parameters
        ----------
        x1 : `FnBase.Vector`
            Vector whose norm is calculated

        Returns
        -------
        norm : float
            The norm of the vector
        """
        return float(sqrt(self.inner(x, x)))

    def dist(self, x1, x2):
        """Calculate the distance between two vectors.

        This is the standard implementation using `norm`. Subclasses
        should override it for optimization purposes.

        Parameters
        ----------
        x1, x2 : `FnBase.Vector`
            Vectors whose mutual distance is calculated

        Returns
        -------
        dist : float
            The distance between the vectors
        """
        if self._dist_using_inner:
            dist_squared = (self.norm(x1)**2 + self.norm(x2)**2 -
                            2 * self.inner(x1, x2).real)
            if dist_squared < 0:  # Compensate for numerical error
                dist_squared = 0.0
            return float(sqrt(dist_squared))
        else:
            return self.norm(x1 - x2)


def _norm_default(x):
    if _blas_is_applicable(x):
        norm = get_blas_funcs('nrm2', dtype=x.space.dtype)
    else:
        norm = np.linalg.norm
    return norm(x.data)


def _inner_default(x1, x2):
    if _blas_is_applicable(x1, x2):
        dot = get_blas_funcs('dotc', dtype=x1.space.dtype)
    elif is_real_dtype(x1.space.dtype):
        dot = np.dot  # still much faster than vdot
    else:
        dot = np.vdot  # slowest alternative
    # x2 as first argument because we want linearity in x1
    return dot(x2.data, x1.data)


class FnWeighting(with_metaclass(ABCMeta, FnWeightingBase)):

    """Abstract base class for `Fn` weighting."""


class FnMatrixWeighting(FnWeighting):

    """Matrix-weighting for `Fn`.

    The weighted inner product with matrix :math:`G` is defined as

    :math:`<a, b> := b^H G a`

    with :math:`b^H` standing for transposed complex conjugate. The
    matrix must be Hermitian and posivive definite, otherwise it does
    not define an inner product. This is not checked during
    initialization.

    Norm and distance are implemented as in the base class by default.
    """

    def __init__(self, matrix, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        matrix : array-like or scipy.sparse.spmatrix
            Weighting matrix of the inner product. Its shape must be
            `(n, n)`, where `n` is the size of `space`.
        dist_using_inner : bool, optional
            Calculate `dist(x, y)` as

            `sqrt(norm(x)**2 + norm(y)**2 - 2*inner(x, y).real)`

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it is not guaranteed to
            evaluate to exactly zero for equal (but not identical)
            `x` and `y`.
        """
        super().__init__(dist_using_inner)
        if isinstance(matrix, sp.sparse.spmatrix):
            self._matrix = matrix
        else:
            self._matrix = np.asmatrix(matrix)

        if self._matrix.shape[0] != self._matrix.shape[1]:
            raise ValueError('matrix with shape {} not square.'
                             ''.format(self._matrix.shape))

    @property
    def matrix(self):
        """Weighting matrix of this inner product."""
        return self._matrix

    @property
    def matrix_issparse(self):
        """Whether the representing matrix is sparse or not."""
        return isinstance(self.matrix, sp.sparse.spmatrix)

    def __eq__(self, other):
        """`inner.__eq__(other) <==> inner == other`.

        Returns
        -------
        equals : bool
            `True` if `other` is an `FnMatrixWeighting` instance with
            **identical** matrix, `False` otherwise.

        See also
        --------
        equiv : test for equivalent inner products
        """
        if other is self:
            return True

        return (isinstance(other, FnMatrixWeighting) and
                self.matrix is other.matrix)

    def equiv(self, other):
        """Test if `other` is an equivalent weighting.

        Returns
        -------
        equivalent : bool
            `True` if `other` is an `FnWeighting` instance which
            yields the same result as this inner product for any
            input, `False` otherwise. This is checked by entry-wise
            comparison of this inner product's matrix with the matrix
            or constant of `other`.
        """
        # Optimization for equality
        if self == other:
            return True

        elif isinstance(other, FnMatrixWeighting):
            if self.matrix_issparse:
                if other.matrix_issparse:
                    # Optimization for different number of nonzero elements
                    if self.matrix.nnz != other.matrix.nnz:
                        return False
                    return (self.matrix != other.matrix).nnz == 0
                else:
                    return np.array_equal(self.matrix.todense(), other.matrix)

            else:  # matrix of `self` is dense
                if other.matrix_issparse:
                    return np.array_equal(self.matrix, other.matrix.todense())
                else:
                    return np.array_equal(self.matrix, other.matrix)

        elif isinstance(other, FnConstWeighting):
            return np.array_equiv(self.matrix.diagonal(), other.const)

        else:
            return False

    def matvec(self, inp, outp=None):
        """The matvec operation of this inner product."""
        if outp is not None:
            if self.matrix_issparse:
                # Unfortunately, there is no native in-place dot product for
                # sparse matrices
                outp[:] = np.asarray(self.matrix.dot(inp)).squeeze()
            else:
                self.matrix.dot(inp, out=outp)
        else:
            return np.asarray(self.matrix.dot(inp)).squeeze()

    def inner(self, x1, x2):
        """Calculate the matrix-weighted inner product of two vectors.

        Parameters
        ----------
        x1, x2 : `Fn.Vector`
            Vectors whose inner product is calculated

        Returns
        -------
        inner : float or complex
            The inner product of the two provided vectors
        """
        return _inner_default(self.matvec(x1), x2)

    def __repr__(self):
        """`w.__repr__() <==> repr(w)`."""
        if self.matrix_issparse:
            inner_fstr = ('<{shape} sparse matrix, format {fmt!r}, {nnz} '
                          'stored entries>')
            fmt = self.matrix.format
            nnz = self.matrix.nnz
            if self._dist_using_inner:
                inner_fstr += ', dist_using_inner=True'
        else:
            inner_fstr = '\n{matrix!r}'
            fmt = ''
            nnz = 0
            if self._dist_using_inner:
                inner_fstr += ',\n dist_using_inner=True'
            else:
                inner_fstr += '\n'

        inner_str = inner_fstr.format(shape=self.matrix.shape, fmt=fmt,
                                      nnz=nnz, matrix=self.matrix)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`inner.__repr__() <==> repr(inner)`."""
        return 'Weighting: matrix =\n{}'.format(self.matrix)


class FnConstWeighting(FnWeighting):

    """Weighting of `Fn` by a constant.

    The weighted inner product with constant `c` is defined as

    :math:`<a, b> := b^H c a`

    with :math:`b^H` standing for transposed complex conjugate.
    """

    def __init__(self, constant, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        constant : float
            Weighting constant of the inner product.
        dist_using_inner : bool, optional
            Calculate `dist(x, y)` as

            `sqrt(norm(x)**2 + norm(y)**2 - 2*inner(x, y).real)`

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it is not guaranteed to
            evaluate to exactly zero for equal (but not identical)
            `x` and `y`.
        """
        super().__init__(dist_using_inner)
        self._const = float(constant)

    @property
    def const(self):
        """Weighting constant of this inner product."""
        return self._const

    def __eq__(self, other):
        """`inner.__eq__(other) <==> inner == other`.

        Returns
        -------
        equal : bool
            `True` if `other` is an `FnConstWeighting`
            instance with the same constant, `False` otherwise.
        """
        return (isinstance(other, FnConstWeighting) and
                self.const == other.const)

    def equiv(self, other):
        """Test if `other` is an equivalent weighting.

        Returns
        -------
        equivalent : bool
            `True` if `other` is an `FnWeighting` instance which
            yields the same result as this inner product for any
            input, `False` otherwise. This is the same as equality
            if `other` is an `FnConstWeighting` instance, otherwise
            by entry-wise comparison of this inner product's constant
            with the matrix of `other`.
        """
        if isinstance(other, FnConstWeighting):
            return self == other
        elif isinstance(other, FnWeighting):
            return other.equiv(self)
        else:
            return False

    def inner(self, x1, x2):
        """Calculate the constant-weighted inner product of two vectors.

        Parameters
        ----------
        x1, x2 : `Fn.Vector`
            Vectors whose inner product is calculated

        Returns
        -------
        inner : float or complex
            The inner product of the two provided vectors
        """
        return self.const * float(_inner_default(x1, x2))

    def norm(self, x):
        """Calculate the constant-weighted norm of a vector.

        Parameters
        ----------
        x1 : `Fn.Vector`
            Vector whose norm is calculated

        Returns
        -------
        norm : float
            The norm of the vector
        """
        return sqrt(abs(self.const)) * float(_norm_default(x))

    def dist(self, x1, x2):
        """Calculate the constant-weighted distance between two vectors.

        Parameters
        ----------
        x1, x2 : `Fn.Vector`
            Vectors whose mutual distance is calculated

        Returns
        -------
        dist : float
            The distance between the vectors
        """
        if self._dist_using_inner:
            dist_squared = (_norm_default(x1)**2 + _norm_default(x2)**2 -
                            2 * _inner_default(x1, x2).real)
            if dist_squared < 0.0:  # Compensate for numerical error
                dist_squared = 0.0
            return sqrt(abs(self.const)) * float(sqrt(dist_squared))
        else:
            return sqrt(abs(self.const)) * self.norm(x1 - x2)

    def __repr__(self):
        """`w.__repr__() <==> repr(w)`."""
        inner_fstr = '{}'
        if self._dist_using_inner:
            inner_fstr += ', dist_using_inner=True'

        inner_str = inner_fstr.format(self.const)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`w.__str__() <==> str(w)`."""
        return 'Weighting: constant = {:.4}'.format(self.const)


class FnNoWeighting(FnConstWeighting):

    """Weighting of `Fn` with constant 1.

    The unweighted inner product is defined as

    :math:`<a, b> := b^H a`

    with :math:`b^H` standing for transposed complex conjugate.
    This is the CPU implementation using NumPy.
    """

    def __init__(self, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        dist_using_inner : bool, optional
            Calculate `dist` using the formula

            norm(x-y)**2 = norm(x)**2 + norm(y)**2 - 2*inner(x, y).real

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) `x` and `y`.
        """
        super().__init__(1.0, dist_using_inner)

    def __repr__(self):
        """`w.__repr__() <==> repr(w)`."""
        if self._dist_using_inner:
            inner_str = 'dist_using_inner=True'
        else:
            inner_str = ''

        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`w.__str__() <==> str(w)`."""
        return self.__class__.__name__


class FnCustomInnerProduct(FnWeighting):

    """Custom inner product on `Fn`."""

    def __init__(self, inner, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        inner : callable
            The inner product implementation. It must accept two
            `Fn.Vector` arguments, return a complex number and
            satisfy the following conditions for all vectors `x`,
            `y` and `z` and scalars `s`:

             - `inner(x, y) == conjugate(inner(y, x))`
             - `inner(s * x, y) == s * inner(x, y)`
             - `inner(x + z, y) == inner(x, y) + inner(z, y)`
             - `inner(x, x) == 0` (approx.) only if `x == 0`
               (approx.)

        dist_using_inner : bool, optional
            Calculate `dist` using the formula

            norm(x-y)**2 = norm(x)**2 + norm(y)**2 - 2*inner(x, y).real

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) `x` and `y`.
        """
        super().__init__(dist_using_inner)

        if not callable(inner):
            raise TypeError('inner product function {!r} not callable.'
                            ''.format(inner))
        self._inner_impl = inner

    @property
    def inner(self):
        """Custom inner product of this instance.."""
        return self._inner_impl

    def __eq__(self, other):
        """`inner.__eq__(other) <==> inner == other`.

        Returns
        -------
        equal : bool
            `True` if `other` is an `FnCustomInnerProduct`
            instance with the same inner product, `False` otherwise.
        """
        return (isinstance(other, FnCustomInnerProduct) and
                self.inner == other.inner)

    def __repr__(self):
        """`w.__repr__() <==> repr(w)`."""
        inner_fstr = '{!r}'
        if self._dist_using_inner:
            inner_fstr += ', dist_using_inner=True'

        inner_str = inner_fstr.format(self.inner)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`w.__str__() <==> str(w)`."""
        return self.__repr__()  # TODO: prettify?


class FnCustomNorm(FnWeighting):

    """Custom norm on `Fn`, removes `inner`."""

    def __init__(self, norm):
        """Initialize a new instance.

        Parameters
        ----------
        norm : callable
            The norm implementation. It must accept an `Fn.Vector`
            argument, return a `RealNumber` and satisfy the
            following properties:

            - `norm(x) >= 0`
            - `norm(x) == 0` (approx.) only if `x == 0` (approx.)
            - `norm(s * x) == abs(s) * norm(x)` for `s` scalar
            - `norm(x + y) <= norm(x) + norm(y)`
        """
        super().__init__(dist_using_inner=False)

        if not callable(norm):
            raise TypeError('norm function {!r} not callable.'
                            ''.format(norm))
        self._norm_impl = norm

    @property
    def norm(self):
        """Custom norm of this instance.."""
        return self._norm_impl

    def inner(self, x1, x2):
        """Inner product is not defined for custom norm."""
        raise NotImplementedError

    def __eq__(self, other):
        """`inner.__eq__(other) <==> inner == other`.

        Returns
        -------
        equal : bool
            `True` if `other` is an `FnCustomNorm`
            instance with the same norm, `False` otherwise.
        """
        return (isinstance(other, FnCustomNorm) and
                self.norm == other.norm)

    def __repr__(self):
        """`w.__repr__() <==> repr(w)`."""
        inner_fstr = '{!r}'
        inner_str = inner_fstr.format(self.norm)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`w.__str__() <==> str(w)`."""
        return self.__repr__()  # TODO: prettify?


class FnCustomDist(FnWeighting):

    """Custom distance on `Fn`, removes `norm` and `inner`."""

    def __init__(self, dist):
        """Initialize a new instance.

        Parameters
        ----------
        dist : callable
            The distance function defining a metric on :math:`F^n`.
            It must accept two `Fn.Vector` arguments and fulfill the
            following conditions for any vectors `x`, `y` and `z`:

            - `dist(x, y) == dist(y, x)`
            - `dist(x, y) >= 0`
            - `dist(x, y) == 0` (approx.) if and only if `x == y`
              (approx.)
            - `dist(x, y) <= dist(x, z) + dist(z, y)`
        """
        super().__init__(dist_using_inner=False)

        if not callable(dist):
            raise TypeError('distance function {!r} not callable.'
                            ''.format(dist))
        self._dist_impl = dist

    @property
    def dist(self):
        """Custom distance of this instance.."""
        return self._dist_impl

    def inner(self, x1, x2):
        """Inner product is not defined for custom distance."""
        raise NotImplementedError

    def norm(self, x):
        """Norm is not defined for custom distance."""
        raise NotImplementedError

    def __eq__(self, other):
        """`inner.__eq__(other) <==> inner == other`.

        Returns
        -------
        equal : bool
            `True` if `other` is an `FnCustomDist`
            instance with the same norm, `False` otherwise.
        """
        return (isinstance(other, FnCustomDist) and
                self.dist == other.dist)

    def __repr__(self):
        """`w.__repr__() <==> repr(w)`."""
        inner_fstr = '{!r}'
        inner_str = inner_fstr.format(self.dist)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`w.__repr__() <==> repr(w)`."""
        return self.__repr__()  # TODO: prettify?


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
