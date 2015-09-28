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
import numpy as np
import scipy as sp
import ctypes
from scipy.linalg.blas import get_blas_funcs
from numbers import Integral
import platform

# ODL imports
from odl.operator.operator import LinearOperator
from odl.set.sets import Set, RealNumbers, ComplexNumbers
from odl.set.space import LinearSpace, UniversalSpace
from odl.util.utility import array1d_repr, array1d_str, dtype_repr


__all__ = ('NtuplesBase', 'FnBase', 'Ntuples', 'Fn', 'Cn', 'Rn',
           'MatVecOperator',
           'ConstWeightedInnerProduct', 'MatrixWeightedInnerProduct')


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
        contains : `bool`
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
        equals : `bool`
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
        def size(self):
            """Length of this vector, equal to space size."""
            return self.space.size

        @property
        def shape(self):
            """Shape of this vector, equals `(size,)`."""
            return (self.size,)

        @abstractmethod
        def copy(self):
            """Create an identical (deep) copy of this vector."""

        @abstractmethod
        def asarray(self, start=None, stop=None, step=None):
            """Extract the data of this array as a numpy array.

            Parameters
            ----------
            start : `int`, Optional (default: `None`)
                Start position. None means the first element.
            start : `int`, Optional (default: `None`)
                One element past the last element to be extracted.
                None means the last element.
            start : `int`, Optional (default: `None`)
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
            equals : `bool`
                `True` if all entries of `other` are equal to this
                vector's entries, `False` otherwise.
            """

        @abstractmethod
        def __getitem__(self, indices):
            """Access values of this vector.

            Parameters
            ----------
            indices : `int` or `slice`
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
            indices : `int` or `slice`
                The position(s) that should be set
            values : {scalar, array-like, `Ntuples.Vector`}
                The value(s) that are to be assigned.

                If `index` is an `int`, `value` must be single value.

                If `index` is a `slice`, `value` must be broadcastable
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
            start : `int`, Optional (default: `None`)
                Start position. None means the first element.
            start : `int`, Optional (default: `None`)
                One element past the last element to be extracted.
                None means the last element.
            start : `int`, Optional (default: `None`)
                Step length. None means 1.
            out : `ndarray`, Optional (default: `None`)
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
                return self.data[start:stop:step].copy()
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
            equals :  `bool`
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
            indices : `int` or `slice`
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
            indices : `int` or `slice`
                The position(s) that should be set
            values : {scalar, array-like, `Ntuples.Vector`}
                The value(s) that are to be assigned.

                If `indices` is an `int`, `value` must be single value.

                If `indices` is a `slice`, `value` must be
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

        dummy = np.empty(0, dtype=self._dtype)
        if np.isrealobj(dummy):
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
    def _multiply(self, z, x, y):
        """The entry-wise product of two vectors, assigned to `z`."""

    class Vector(with_metaclass(ABCMeta, NtuplesBase.Vector,
                                LinearSpace.Vector)):

        """Abstract class for representation of :math:`F^n` vectors.

        Defines abstract attributes and concrete ones which are
        independent of data representation.
        """


def _lincomb(z, a, x, b, y, dtype):
    """Raw linear combination depending on data type."""
    def fallback_axpy(x, y, n, a):
        """Fallback axpy implementation avoiding copy."""
        if a != 0:
            y /= a
            y += x
            y *= a
        return y

    def fallback_scal(a, x, n):
        """Fallback scal implementation."""
        x *= a
        return x

    def fallback_copy(x, y, n):
        """Fallback copy implementation."""
        y[...] = x[...]
        return y

    if (dtype in (np.float32, np.float64, np.complex64, np.complex128) and
            all(a.flags.contiguous for a in (x.data, y.data, z.data))):
        # pylint: disable=unbalanced-tuple-unpacking
        axpy, scal, copy = get_blas_funcs(
            ['axpy', 'scal', 'copy'], arrays=(x.data, y.data))
    else:
        axpy, scal, copy = (fallback_axpy, fallback_scal, fallback_copy)

    if x is y and b != 0:
        # x is aligned with y -> z = (a+b)*x
        _lincomb(z, a+b, x, 0, x, dtype)
    elif z is x and z is y:
        # All the vectors are aligned -> z = (a+b)*z
        scal(a+b, z.data, len(z))
    elif z is x:
        # z is aligned with x -> z = a*z + b*y
        if a != 1:
            scal(a, z.data, len(z))
        if b != 0:
            axpy(y.data, z.data, len(z), b)
    elif z is y:
        # z is aligned with y -> z = a*x + b*z
        if b != 1:
            scal(b, z.data, len(z))
        if a != 0:
            axpy(x.data, z.data, len(z), a)
    else:
        # We have exhausted all alignment options, so x != y != z
        # We now optimize for various values of a and b
        if b == 0:
            if a == 0:  # Zero assignment -> z = 0
                z.data[:] = 0
            else:  # Scaled copy -> z = a*x
                copy(x.data, z.data, len(z))
                if a != 1:
                    scal(a, z.data, len(z))
        else:
            if a == 0:  # Scaled copy -> z = b*y
                copy(y.data, z.data, len(z))
                if b != 1:
                    scal(b, z.data, len(z))

            elif a == 1:  # No scaling in x -> z = x + b*y
                copy(x.data, z.data, len(z))
                axpy(y.data, z.data, len(z), b)
            else:  # Generic case -> z = a*x + b*y
                copy(y.data, z.data, len(z))
                if b != 1:
                    scal(b, z.data, len(z))
                axpy(x.data, z.data, len(z), a)


def _dist_not_impl(x, y):
    raise NotImplementedError('no distance function provided.')


def _norm_not_impl(x):
    raise NotImplementedError('no norm function provided.')


def _inner_not_impl(x, y):
    raise NotImplementedError('no inner product function provided.')


# TODO: optimize?
def _dist_default(x, y):
    return (x-y).norm()


def _norm_default(x):
    return np.sqrt(x.inner(x))


def _inner_default(x, y):
    # y as first argument because we want linearity in x
    return np.vdot(y.data, x.data)


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
        size : int
            The number of dimensions of the space
        dtype : object
            The data type of the storage array. Can be provided in any
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.
            Only scalar data types are allowed.

        kwargs : {'dist', 'norm', 'inner'}
            'dist' : callable, optional (Default: `norm(x-y)`)
                The distance function defining a metric on :math:`F^n`. It
                must accept two array arguments and fulfill the following
                conditions for any vectors `x`, `y` and `z`:

                - `dist(x, y) == dist(y, x)`
                - `dist(x, y) >= 0`
                - `dist(x, y) == 0` (approx.) if and only if `x == y`
                  (approx.)
                - `dist(x, y) <= dist(x, z) + dist(z, y)`

            'norm' : callable, optional (Default: `sqrt(inner(x,y))`)
                The norm implementation. It must accept an array-like
                argument, return a `RealNumber` and satisfy the following
                properties:

                - `norm(x) >= 0`
                - `norm(x) == 0` (approx.) only if `x == 0` (approx.)
                - `norm(s * x) == abs(s) * norm(x)` for `s` scalar
                - `norm(x + y) <= norm(x) + norm(y)`

            'inner' : callable, optional
                The inner product implementation. It must accept two
                array-like arguments, return a complex number and satisfy
                the following conditions for all vectors `x`, `y` and `z`
                and scalars `s`:

                 - `inner(x, y) == conjugate(inner(y, x))`
                 - `inner(s * x, y) == s * inner(x, y)`
                 - `inner(x + z, y) == inner(x, y) + inner(z, y)`
                 - `inner(x, x) == 0` (approx.) only if `x == 0` (approx.)
        """
        super().__init__(size, dtype)

        dist = kwargs.get('dist', None)
        norm = kwargs.get('norm', None)
        inner = kwargs.get('inner', None)

        if dist is not None:
            if norm is not None:
                raise ValueError('custom norm cannot be combined with '
                                 'custom distance.')
            if inner is not None:
                raise ValueError('custom inner product cannot be combined '
                                 'with custom distance.')
            norm = _norm_not_impl
            inner = _inner_not_impl
        elif norm is not None:
            if inner is not None:
                raise ValueError('custom inner product cannot be combined '
                                 'with custom norm.')
            inner = _inner_not_impl
            dist = _dist_default
        elif inner is not None:
            dist = _dist_default
            norm = _norm_default
        else:
            dist = _dist_default
            norm = _norm_default
            inner = _inner_default

        if not callable(dist):
            raise TypeError('distance function {!r} not callable.'
                            ''.format(dist))
        if not callable(norm):
            raise TypeError('norm function {!r} not callable.'.format(norm))
        if not callable(inner):
            raise TypeError('inner product function {!r} not callable.'
                            ''.format(inner))
        self._norm_impl = norm
        self._dist_impl = dist
        self._inner_impl = inner

    def _lincomb(self, z, a, x, b, y):
        """Linear combination of `x` and `y`.

        Calculate `z = a * x + b * y` using optimized BLAS routines if
        possible.

        Parameters
        ----------
        z : `Fn.Vector`
            The Vector that the result is written to.
        a, b : `field` element
            Scalar to multiply `x` and `y` with.
        x, y : `Fn.Vector`
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
        _lincomb(z, a, x, b, y, self.dtype)

    def _dist(self, x, y):
        """Calculate the distance between two vectors.

        Parameters
        ----------
        x, y : `Fn.Vector`
            The vectors whose mutual distance is calculated

        Returns
        -------
        dist : `float`
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
        return self._dist_impl(x, y)

    def _norm(self, x):
        """Calculate the norm of a vector.

        Parameters
        ----------
        x : `Fn.Vector`
            The vector whose norm is calculated

        Returns
        -------
        norm : `float`
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
        return self._norm_impl(x)

    def _inner(self, x, y):
        """Raw inner product of two vectors.

        Parameters
        ----------

        x, y : `Cn.Vector`
            The vectors whose inner product is calculated

        Returns
        -------
        inner : `complex`
            Inner product of `x` and `y`.

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
        return self._inner_impl(x, y)

    def _multiply(self, z, x, y):
        """The entry-wise product of two vectors, assigned to `z`.

        Parameters
        ----------
        z : `Cn.Vector`
            The result vector
        x : `Cn.Vector`
            First factor
        y : `Cn.Vector`
            Second factor, used to store the result

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
        if z is x and z is y:  # z = z*z
            z.data[:] *= z.data
        elif z is x:  # z = z*y
            z.data[:] *= y.data
        elif z is y:  # z = z*x
            z.data[:] *= x.data
        else:  # z = x*y
            z.data[:] = x.data
            z.data[:] *= y.data

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
        equals : `bool`
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
                self._dist_impl == other._dist_impl and
                self._norm_impl == other._norm_impl and
                self._inner_impl == other._inner_impl)

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

    Its elements are represented as instances of the inner `Fn.Vector`
    class.

    See also
    --------
    See the module documentation for attributes, methods etc.
    """

    def __init__(self, size, dtype=np.complex128, **kwargs):
        """Initialize a new instance.

        Only complex floating-point data types are allowed.
        """
        dist = kwargs.pop('dist', None)
        norm = kwargs.pop('norm', None)
        inner = kwargs.pop('inner', None)

        super().__init__(size, dtype, dist=dist, norm=norm, inner=inner,
                         **kwargs)

        if not np.iscomplexobj(np.empty(0, dtype=self._dtype)):
            raise TypeError('data type {} not a complex floating-point type.'
                            ''.format(dtype))
        self._real_dtype = _TYPE_MAP_C2R[self._dtype]

    @property
    def real_dtype(self):
        """The corresponding real data type of this space."""
        return self._real_dtype

    def __repr__(self):
        """`cn.__repr__() <==> repr(cn)`."""
        if self.dtype == np.complex128:
            return 'Cn({})'.format(self.size)
        else:
            return 'Cn({}, {!r})'.format(self.size, self.dtype)

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
    See the module documentation for attributes, methods etc.
    """

    def __init__(self, size, dtype=np.float64, **kwargs):
        """Initialize a new instance.

        Only real floating-point types are allowed.
        """
        dist = kwargs.pop('dist', None)
        norm = kwargs.pop('norm', None)
        inner = kwargs.pop('inner', None)

        super().__init__(size, dtype, dist=dist, norm=norm, inner=inner,
                         **kwargs)

        if not np.isrealobj(np.empty(0, dtype=self._dtype)):
            raise TypeError('data type {} not a real floating-point type.'
                            ''.format(dtype))

    def __repr__(self):
        """`rn.__repr__() <==> repr(rn)`."""
        if self.dtype == np.float64:
            return 'Rn({})'.format(self.size)
        else:
            return 'Rn({}, {!r})'.format(self.size, self.dtype)

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


class InnerProductBase(with_metaclass(ABCMeta, object)):

    """Abstract base class for raw (weighted) inner products.

    This class and its subclasses serve as a simple means to evaluate
    and compare weighted inner products semantically rather than by
    identity on a pure function level.

    It is implemented similarly to `Operator` but without extra type
    checks of input parameters - this is done in the caller `inner()`
    of the `LinearSpace` instance where this inner product is used.
    """

    @abstractmethod
    def __eq__(self, other):
        """`inner.__eq__(other) <==> inner == other`.

        Returns
        -------
        equal : bool
            `True` if `other` is a `WeightedInnerBase` instance
            represented by the **identical** matrix, `False` otherwise.

        Notes
        -----
        This operation must be computationally cheap, i.e. no arrays
        may be compared element-wise. That is the task of the `equiv`
        method.
        """

    @abstractmethod
    def equiv(self, other):
        """Test if `other` is an equivalent inner product.

        Returns
        -------
        equivalent : bool
            `True` if `other` is a `WeightedInnerBase` instance which
            yields the same result as this inner product for any
            input, `False` otherwise. This is checked by entry-wise
            comparison of this inner product's matrix with the matrix
            or constant of `other`.
        """

    @abstractmethod
    def __call__(self, x1, x2):
        """`inner.__call__(x1, x2) <==> inner(x1, x2)`.

        Parameters
        ----------
        x1, x2 : `FnBase.Vector`
            Vectors whose inner product is calculated

        Returns
        -------
        inner : float or complex
            The inner product of the two provided vectors
        """


class InnerProduct(with_metaclass(ABCMeta, InnerProductBase)):

    """Abstract base class for raw inner products, NumPy version."""


class MatrixWeightedInnerProduct(InnerProduct):

    """Matrix-weighted :math:`F^n` inner products in NumPy.

    The weighted inner product with matrix :math:`G` is defined as

    :math:`<a, b> := b^H G a`

    with :math:`b^H` standing for transposed complex conjugate. The
    matrix must be Hermitian and posivive definite, otherwise it does
    not define an inner product. This is not checked during
    initialization.
    """

    def __init__(self, matrix):
        """Initialize a new instance.

        Parameters
        ----------
        matrix : array-like or scipy.sparse.spmatrix
            Weighting matrix of the inner product. Its shape must be
            `(n, n)`, where `n` is the size of `space`.
        """
        super().__init__()
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
            `True` if `other` is a `MatrixWeightedInner` instance with
            **identical** matrix, `False` otherwise.

        See also
        --------
        equiv : test for equivalent inner products
        """
        if other is self:
            return True

        return (isinstance(other, MatrixWeightedInnerProduct) and
                self.matrix is other.matrix)

    def equiv(self, other):
        """Test if `other` is an equivalent inner product.

        Returns
        -------
        equivalent : bool
            `True` if `other` is a `WeightedInner` instance which
            yields the same result as this inner product for any
            input, `False` otherwise. This is checked by entry-wise
            comparison of this inner product's matrix with the matrix
            or constant of `other`.
        """
        # Optimization for equality
        if self == other:
            return True

        elif isinstance(other, MatrixWeightedInnerProduct):
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

        elif isinstance(other, ConstWeightedInnerProduct):
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

    def __call__(self, x1, x2):
        """Raw call method of this inner product.`

        Calculate the inner product of `x1` and `x2` weighted by the
        matrix of this instance.
        """
        # vdot conjugates the first argument if complex
        return np.vdot(x2.data, self.matvec(x1.data))

    def __repr__(self):
        """`inner.__repr__() <==> repr(inner)`."""
        if self.matrix_issparse:
            return ('MatrixWeightedInnerProduct(<{} sparse matrix, '
                    'format {!r}, {} stored entries>)'
                    ''.format(self.matrix.shape, self.matrix.format,
                              self.matrix.nnz))
        else:
            return 'MatrixWeightedInnerProduct(\n{!r}\n)'.format(self.matrix)

    def __str__(self):
        """`inner.__repr__() <==> repr(inner)`."""
        return '(x, y) --> y^H G x,  G =\n{}'.format(self.matrix)


class ConstWeightedInnerProduct(InnerProduct):

    """Operator for constant-weighted :math:`F^n` inner products.

    The weighted inner product with constant :math:`c` is defined as

    :math:`<a, b> := b^H c a`

    with :math:`b^H` standing for transposed complex conjugate.
    This is the CPU implementation using NumPy.
    """

    def __init__(self, constant):
        """Initialize a new instance.

        Parameters
        ----------
        constant : float
            Weighting constant of the inner product.
        """
        super().__init__()
        self._const = float(constant)

    @property
    def const(self):
        """Weighting constant of this inner product."""
        return self._const

    def __call__(self, x1, x2):
        """Raw call method of this inner product.

        Calculate the inner product of `x1` and `x2` weighted by the
        constant of this instance.
        """
        # vdot conjugates the first argument if complex
        return np.vdot(x2.data, x1.data) * self.const

    def __eq__(self, other):
        """`inner.__eq__(other) <==> inner == other`.

        Returns
        -------
        equal : bool
            `True` if `other` is a `ConstWeightedInnerProduct`
            instance with the same constant, `False` otherwise.
        """
        return (isinstance(other, ConstWeightedInnerProduct) and
                self.const == other.const)

    def equiv(self, other):
        """Test if `other` is an equivalent inner product.

        Returns
        -------
        equivalent : bool
            `True` if `other` is a `WeightedInner` instance which
            yields the same result as this inner product for any
            input, `False` otherwise. This is the same as equality
            if `other` is a `ConstWeightedInner` instance, otherwise
            by entry-wise comparison of this inner product's constant
            with the matrix of `other`.
        """
        if isinstance(other, ConstWeightedInnerProduct):
            return self == other
        elif isinstance(other, MatrixWeightedInnerProduct):
            return other.equiv(self)
        else:
            return False

    def __repr__(self):
        """`inner.__repr__() <==> repr(inner)`."""
        return '{}({})'.format(self.__class__.__name__, self.const)

    def __str__(self):
        """`inner.__repr__() <==> repr(inner)`."""
        return '(x, y) --> {:.4} * y^H x'.format(self.const)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
