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
|`Rn`         |`Ntuples`,    |`n`-tuples of real numbers with         |
|             |`Algebra`     |vector-vector multiplication            |
+-------------+--------------+----------------------------------------+
|`MetricRn`   |`Rn`,         |`Rn` with a metric, i.e. a function to  |
|             |`MetricSpace` |measure the distance between elements   |
+-------------+--------------+----------------------------------------+
|`NormedRn`   |`MetricRn`,   |`MetricRn` with a norm. The metric is   |
|             |`NormedSpace` |induced by the via the relation         |
|             |              |`dist(x, y) = norm(x - y)`              |
+-------------+--------------+----------------------------------------+
|`HilbertRn`  |`NormedRn`,   |`NormedRn` with an inner product. The   |
|             |`HilbertSpace`|norm is induced by the inner product via|
|             |              |the relation                            |
|             |              |`norm(x) = sqrt(inner(x, x))`           |
+-------------+--------------+----------------------------------------+
|`EuclideanRn`|`HilbertRn`   |`HilbertRn` with the standard inner     |
|             |              |(dot) product                           |
+-------------+--------------+----------------------------------------+
|`Cn`         |`Ntuples`,    |`n`-tuples of complex numbers with      |
|             |`Algebra`     |vector-vector multiplication            |
+-------------+--------------+----------------------------------------+
|`MetricCn`   |`Cn`,         |`Cn` with a metric, i.e. a function to  |
|             |`MetricSpace` |measure the distance between elements   |
+-------------+--------------+----------------------------------------+
|`NormedCn`   |`MetricCn`,   |`MetricCn` with a norm. The metric is   |
|             |`NormedSpace` |induced by the via the relation         |
|             |              |`dist(x, y) = norm(x - y)`              |
+-------------+--------------+----------------------------------------+
|`HilbertCn`  |`NormedCn`,   |`NormedCn` with an inner product. The   |
|             |`HilbertSpace`|norm is induced by the inner product via|
|             |              |the relation                            |
|             |              |`norm(x) = sqrt(inner(x, x))`           |
+-------------+--------------+----------------------------------------+
|`EuclideanCn`|`HilbertCn`   |`HilbertRn` with the standard inner     |
|             |              |(dot) product (with complex conjugation)|
+-------------+--------------+----------------------------------------+

Space attributes and methods
----------------------------
The following tables summarize all attributes and methods of spaces in
this module. Each table reflects the *added* features for the
respective class.

**`Ntuples` and subclasses**

Attributes:

+----------+-------------+------------------------------------------+
|Name      |Type         |Description                               |
+==========+=============+==========================================+
|`dim`     |`int`        |The number of entries per tuple           |
+----------+-------------+------------------------------------------+
|`dtype`   |`type`       |The data dype of each tuple entry         |
+----------+-------------+------------------------------------------+

Methods:

+-----------+---------------------+-----------------------------------+
|Signature  |Return type          |Description                        |
+===========+=====================+===================================+
|`element   |`<space type>.Vector`|Create a space element. If `inp` is|
|(inp=None)`|                     |`None`, merely memory is allocated.|
|           |                     |Otherwise, the element is created  |
|           |                     |from `inp`.                        |
+-----------+---------------------+-----------------------------------+

Magic methods:



Vector attributes and methods
-----------------------------
Similarly, the following tables incrementally summarize all attributes
and methods of vectors in this module.

**`Ntuples.Vector` and subclasses**

+----------+-------------+------------------------------------------+
|Name      |Type         |Description                               |
+==========+=============+==========================================+
|`data`    |`type`       |The data dype of each tuple entry         |
+----------+-------------+------------------------------------------+
|`data_ptr`|`type`       |The data dype of each tuple entry         |
+----------+-------------+------------------------------------------+

TODO: continue here

"""

# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from builtins import super
from future import standard_library

# External module imports
import numpy as np
import scipy as sp
from numbers import Integral
from math import sqrt

# ODL imports
from odl.space.set import Set, RealNumbers, ComplexNumbers
from odl.space.space import MetricSpace, NormedSpace, HilbertSpace, Algebra
from odl.utility.utility import errfmt, array1d_repr
try:
    from odl.space.cuda import CudaRn
    try:
        CudaRn(1).element()
    except MemoryError:
        print(errfmt("""
        Warning: Your GPU seems to be misconfigured. Skipping CUDA-dependent
        modules."""))
        CUDA_AVAILABLE = False
    else:
        CUDA_AVAILABLE = True
except ImportError:
    CudaRn = None
    CUDA_AVAILABLE = False

standard_library.install_aliases()


class Ntuples(Set):

    """The set of `n`-tuples of arbitrary type.

    Attributes
    ----------


    Methods
    -------


    Magic methods
    -------------

    +----------------------+----------------+--------------------+
    |Signature             |Provides syntax |Implementation      |
    +======================+================+====================+
    |`__eq__(other)`       |`self == other` |`equals(other)`     |
    +----------------------+----------------+--------------------+
    |`__ne__(other)`       |`self != other` |`not equals(other)` |
    +----------------------+----------------+--------------------+
    |`__contains__(other)` |`other in self` |`contains(other)`   |
    +----------------------+----------------+--------------------+
    """

    def __init__(self, dim, dtype):
        """Initialize a new instance.

        Parameters
        ----------
        dim : `int`
            The number entries per tuple
        dtype : `object`
            The data type for each tuple entry. Can be provided in any
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.
        """
        if not isinstance(dim, Integral) or dim < 1:
            raise TypeError(errfmt('''
            `dim` {} is not a positive integer.'''.format(dim)))
        self._dim = dim
        self._dtype = np.dtype(dtype)

    def element(self, inp=None):
        """Create a new element.

        Parameters
        ----------
        inp : array-like or scalar, optional
            Input to initialize the new element.

            If `inp` is `None`, an empty element is created with no
            guarantee of its state (memory allocation only).

            If `inp` is a `numpy.ndarray` of shape `(dim,)` and the
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
        >>> strings3 = Ntuples(3, dtype='S1')  # 1-char strings
        >>> x = strings3.element(['w', 'b', 'w'])
        >>> x
        Ntuples(3, dtype('S1')).element(['w', 'b', 'w'])
        >>> y = strings3.element()
        >>> y.assign(x)
        >>> y == x
        True
        >>> y = strings3.element('b'); print(y)
        ['b', 'b', 'b']

        Array views are preserved:

        >>> strings2 = Ntuples(2, dtype='S1')  # 1-char strings
        >>> x = strings3.element(['w', 'b', 'w'])
        >>> y = strings2.element(x[::2])  # view into x
        >>> y[:] = 'x'
        >>> print(x)
        ['x', 'b', 'x']
        """
        if inp is None:
            inp = np.empty(self.dim, dtype=self.dtype)
        elif isinstance(inp, Ntuples.Vector):
            return self.element(inp.data)
        else:
            inp = np.atleast_1d(inp).astype(self.dtype, copy=False)

            if inp.shape == (1,):
                inp = np.repeat(inp, self.dim)
            elif inp.shape == (self.dim,):
                pass
            else:
                raise ValueError(errfmt('''
                `inp` shape {} not broadcastable to shape ({}).
                '''.format(inp.shape, self.dim)))

        return self.Vector(self, inp)

    @property
    def dtype(self):
        """The data type of each entry.

        Examples
        --------
        >>> int_3 = Ntuples(3, dtype=int)
        >>> int_3.dtype
        dtype('int64')
        """
        return self._dtype

    @property
    def dim(self):
        """The dimension of this space.

        Examples
        --------
        >>> int_3 = Ntuples(3, dtype=int)
        >>> int_3.dim
        3
        """
        return self._dim

    def equals(self, other):
        """Test if `other` is equal to this space.

        Parameters
        ----------
        `other` : `object`
            The object to check for equality

        Returns
        -------
        equals : boolean
            `True` if `other` is an instance of this space's type
            with the same `dim` and `dtype`, otherwise `False`.

        Examples
        --------
        >>> int_3 = Ntuples(3, dtype=int)
        >>> int_3.equals(int_3)
        True

        Equality is not identity:

        >>> int_3a, int_3b = Ntuples(3, int), Ntuples(3, int)
        >>> int_3a.equals(int_3b)
        True
        >>> int_3a is int_3b
        False

        >>> int_3, int_4 = Ntuples(3, int), Ntuples(4, int)
        >>> int_3.equals(int_4)
        False
        >>> int_3, str_3 = Ntuples(3, 'int'), Ntuples(3, 'string')
        >>> int_3.equals(str_3)
        False

        Equality can also be checked with "==":

        >>> int_3, int_4 = Ntuples(3, int), Ntuples(4, int)
        >>> int_3 == int_3
        True
        >>> int_3 == int_4
        False
        >>> int_3 != int_4
        True
        """
        return (isinstance(other, type(self)) and
                self.dim == other.dim and
                self.dtype == other.dtype)

    def contains(self, other):
        """Test if `other` is contained in this space.

        Parameters
        ----------
        other : `object`
            The object to be tested for membership

        Returns
        -------
        contains : `bool`
            `True` if `other` is an instance of the same `Vector`
            class, `other.space.dim` equals this space's `dim` and
            `other.space.dtype` can be safely cast to this space's
            data type. `False` otherwise.

        Examples
        --------
        >>> long_3 = Ntuples(3, dtype='int64')
        >>> double_3 = Ntuples(3, dtype='float64')
        >>> long_vec = long_3.element([1, 2, 3])
        >>> long_vec in double_3
        True
        >>> int_3 = Ntuples(3, dtype='int32')
        >>> float_3 = Ntuples(3, dtype='float32')
        >>> int_vec = int_3.element([1, 2, 3])
        >>> int_vec in float_3  # Unsafe cast
        False
        """
        return (isinstance(other, type(self).Vector) and
                len(other) == self.dim and
                np.can_cast(other.space.dtype, self.dtype))

    def __repr__(self):
        """s.__repr__() <==> repr(s)."""
        return 'Ntuples({}, {!r})'.format(self.dim, self.dtype)

    def __str__(self):
        """s.__str__() <==> str(s)."""
        return 'Ntuples({}, {})'.format(self.dim, self.dtype)

    class Vector(object):

        """Representation of an `Ntuples` element.

        Attributes
        ----------

        +-----------+---------------+---------------------------------+
        |Name       |Type           |Description                      |
        +===========+===============+=================================+
        |`space`    |`Set`          |The set to which this vector     |
        |           |               |belongs                          |
        +-----------+---------------+---------------------------------+
        |`data`     |`numpy.ndarray`|The container for the vector     |
        |           |               |entries                          |
        +-----------+---------------+---------------------------------+
        |`data_ptr` |`int`          |A raw memory pointer to the data |
        |           |               |container. Can be processed with |
        |           |               |the `ctypes` module in Python.   |
        +-----------+---------------+---------------------------------+

        Methods
        -------

        +----------------+--------------------+-----------------------+
        |Signature       |Return type         |Description            |
        +================+====================+=======================+
        |`equals(other)` |`boolean`           |Test if `other` is     |
        |                |                    |equal to this vector.  |
        +----------------+--------------------+-----------------------+
        |`assign(other)` |`None`              |Copy the values of     |
        |                |                    |`other` to this vector.|
        +----------------+--------------------+-----------------------+
        |`copy()`        |`LinearSpace.Vector`|Create a (deep) copy of|
        |                |                    |this vector.           |
        +----------------+--------------------+-----------------------+

        Magic methods
        -------------

        +------------------+----------------+-------------------------+
        |Signature         |Provides syntax |Implementation           |
        +==================+================+=========================+
        |`__eq__(other)`   |`self == other` |`equals(other)`          |
        +------------------+----------------+-------------------------+
        |`__ne__(other)`   |`self != other` |`not equals(other)`      |
        +------------------+----------------+-------------------------+

        """

        def __init__(self, space, data):
            """Initialize a new instance."""
            if not isinstance(space, Ntuples):
                raise TypeError(errfmt('''
                `space` {!r} not an instance of `Ntuples`.
                '''.format(space)))

            if not isinstance(data, np.ndarray):
                raise TypeError(errfmt('''
                `data` {!r} not an instance of `numpy.ndarray`.
                '''.format(data)))

            if data.dtype != space.dtype:
                raise TypeError(errfmt('''
                `data.dtype` {} not equal to `space.dtype` {}.
                '''.format(data.dtype, space.dtype)))

            if data.shape != (space.dim,):
                raise ValueError(errfmt('''
                `data.shape` {} not equal to `(space.dim,)` {}.
                '''.format(data.shape, (space.dim,))))

            self._space = space
            self._data = data

        @property
        def space(self):
            """The space this vector belongs to."""
            return self._space

        @property
        def data(self):
            """The vector's data representation, a `numpy.ndarray`.

            Examples
            --------
            >>> vec = Ntuples(3, int).element([1, 2, 3])
            >>> vec.data
            array([1, 2, 3])
            """
            return self._data

        @property
        def data_ptr(self):
            """A raw pointer to the data container.

            Examples
            --------
            >>> import ctypes
            >>> vec = Ntuples(3, int).element([1, 2, 3])
            >>> arr_type = ctypes.c_int64 * 3
            >>> buffer = arr_type.from_address(vec.data_ptr)
            >>> arr = np.frombuffer(buffer, dtype=int)
            >>> arr
            array([1, 2, 3])

            In-place modification via pointer:

            >>> arr[0] = 5
            >>> vec
            Ntuples(3, dtype('int64')).element([5, 2, 3])
            """
            return self._data.ctypes.data

        def equals(self, other):
            """Test if `other` is equal to this vector.

            Parameters
            ----------
            other : `object`
                Object to compare to this vector

            Returns
            -------
            equals : `boolean`
                `True` if `other` is an element of this vector's
                space with equal entries, `False` otherwise.

            See also
            --------
            Check the `contains()` documentation for rules of space
            membership.

            Examples
            --------
            >>> vec1 = Ntuples(3, int).element([1, 2, 3])
            >>> vec2 = Ntuples(3, int).element([-1, 2, 0])
            >>> vec1.equals(vec2)
            False
            >>> vec2 = Ntuples(3, int).element([1, 2, 3])
            >>> vec1.equals(vec2)
            True
            >>> vec1 == vec2  # equivalent
            True

            Equality can hold for different data types as long as
            casts are safe in at least one direction:

            >>> vec2 = Ntuples(3, float).element([1, 2, 3])
            >>> vec1.equals(vec2) and vec2.equals(vec1)
            True
            >>> vec2 = Ntuples(3, 'float32').element([1, 2, 3])
            >>> vec1.equals(vec2) or vec2.equals(vec1)
            False
            """
            return (hasattr(other, 'space') and
                    (other in self.space or self in other.space) and
                    np.all(self.data == other.data))

        # Convenience functions
        def assign(self, other):
            """Assign the values of `other` to this vector.

            Parameters
            ----------
            other : `Ntuples.Vector`
                The values to be copied to this vector. `other`
                must be an element of this vector's space.

            Returns
            -------
            `None`

            Examples
            --------
            >>> vec1 = Ntuples(3, int).element([1, 2, 3])
            >>> vec2 = Ntuples(3, int).element([-1, 2, 0])
            >>> vec1.assign(vec2)
            >>> vec1
            Ntuples(3, dtype('int64')).element([-1, 2, 0])
            """
            if other not in self.space:
                raise TypeError(errfmt('''
                `other` {!r} not in `space` {}'''.format(other, self.space)))
            self.data[:] = other.data[:]

        def copy(self):
            """Create an identical (deep) copy of this vector.

            Parameters
            ----------
            None

            Returns
            -------
            copy : `Ntuples.Vector`

            Examples
            --------
            >>> vec1 = Ntuples(3, int).element([1, 2, 3])
            >>> vec2 = vec1.copy()
            >>> vec2
            Ntuples(3, dtype('int64')).element([1, 2, 3])
            >>> vec1 == vec2
            True
            >>> vec1 is vec2
            False
            """
            return self.space.element(self.data.copy())

        def __len__(self):
            """The dimension this vector's space.

            Examples
            --------
            >>> len(Ntuples(3, int).element())
            3
            """
            return self.space.dim

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
            >>> str_3 = Ntuples(3, dtype='S6')  # 6-char strings
            >>> x = str_3.element(['a', 'Hello!', '0'])
            >>> x[0]
            'a'
            >>> x[1:3]
            Ntuples(2, dtype('S6')).element(['Hello!', '0'])
            """
            try:
                return self.data[int(indices)]  # single index
            except TypeError:
                arr = self.data[indices]
                return Ntuples(len(arr), self.space.dtype).element(arr)

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

            Returns
            -------
            None

            Examples
            --------
            >>> int_3 = Ntuples(3, int)
            >>> x = int_3.element([1, 2, 3])
            >>> x[0] = 5
            >>> x
            Ntuples(3, dtype('int64')).element([5, 2, 3])

            Assignment from array-like structures or another
            vector:

            >>> y = Ntuples(2, 'short').element([-1, 2])
            >>> x[:2] = y
            >>> x
            Ntuples(3, dtype('int64')).element([-1, 2, 3])
            >>> x[1:3] = [7, 8]
            >>> x
            Ntuples(3, dtype('int64')).element([-1, 7, 8])
            >>> x[:] = np.array([0, 0, 0])
            >>> x
            Ntuples(3, dtype('int64')).element([0, 0, 0])

            Broadcasting is also supported:

            >>> x[1:3] = -2.
            >>> x
            Ntuples(3, dtype('int64')).element([0, -2, -2])

            Be aware of unsafe casts and over-/underflows, there
            will be warnings at maximum.

            >>> x = Ntuples(2, 'int8').element([0, 0])
            >>> maxval = 127  # maximum signed 8-bit int
            >>> x[0] = maxval + 1
            >>> x
            Ntuples(2, dtype('int8')).element([-128, 0])
            >>> x[:] = np.arange(2, dtype='int64')
            >>> x
            Ntuples(2, dtype('int8')).element([0, 1])
            """
            self.data[indices] = values

        def __eq__(self, other):
            """`vec.__eq__(other) <==> vec == other`."""
            return self.equals(other)

        def __ne__(self, other):
            """`vec.__ne__(other) <==> vec != other`."""
            return not self.equals(other)

        def __str__(self):
            """`vec.__str__() <==> str(vec)`."""
            return array1d_repr(self.data)

        def __repr__(self):
            """`vec.__repr__() <==> repr(vec)`."""
            return '{!r}.element({})'.format(self.space,
                                             array1d_repr(self.data))


def _lincomb(z, a, x, b, y, dtype):
    """Raw linear combination depending on data type."""
    def fallback_axpy(a, x, y):
        """Fallback axpy implementation avoiding copy."""
        if a != 0:
            y /= a
            y += x
            y *= a
        return y

    def fallback_scal(a, x):
        """Fallback scal implementation."""
        x *= a
        return x

    def fallback_copy(x, y):
        """Fallback copy implementation."""
        y[...] = x[...]
        return y

    # pylint: disable=unbalanced-tuple-unpacking
    blas_axpy, blas_scal, blas_copy = sp.linalg.blas.get_blas_funcs(
        ['axpy', 'scal', 'copy'], dtype=dtype)

    if (dtype in (np.float32, np.float64, np.complex64, np.complex128) and
            all(a.flags.contiguous for a in (x.data, y.data, z.data))):
        axpy, scal, copy = (blas_axpy, blas_scal, blas_copy)
    else:
        axpy, scal, copy = (fallback_axpy, fallback_scal, fallback_copy)

    if x is y and b != 0:
        # x is aligned with y -> z = (a+b)*x
        _lincomb(z, a+b, x, 0, x, dtype)
    elif z is x and z is y:
        # All the vectors are aligned -> z = (a+b)*z
        scal(a+b, z.data)
    elif z is x:
        # z is aligned with x -> z = a*z + b*y
        if a != 1:
            scal(a, z.data)
        if b != 0:
            axpy(y.data, z.data, len(z), b)
    elif z is y:
        # z is aligned with y -> z = a*x + b*z
        if b != 1:
            scal(b, z.data)
        if a != 0:
            axpy(x.data, z.data, len(z), a)
    else:
        # We have exhausted all alignment options, so x != y != z
        # We now optimize for various values of a and b
        if b == 0:
            if a == 0:  # Zero assignment -> z = 0
                z.data[:] = 0
            else:  # Scaled copy -> z = a*x
                copy(x.data, z.data)
                if a != 1:
                    scal(a, z.data)
        else:
            if a == 0:  # Scaled copy -> z = b*y
                copy(y.data, z.data)
                if b != 1:
                    scal(b, z.data)

            elif a == 1:  # No scaling in x -> z = x + b*y
                copy(x.data, z.data)
                axpy(y.data, z.data, len(z), b)
            else:  # Generic case -> z = a*x + b*y
                copy(y.data, z.data)
                if b != 1:
                    scal(b, z.data)
                axpy(x.data, z.data, len(z), a)


class Rn(Ntuples, Algebra):

    """The real vector space :math:`R^n` with vector multiplication.

    Its elements are represented as instances of the inner `Rn.Vector`
    class.

    See also
    --------
    See `Ntuples`, `Algebra` and `LinearSpace` for attributes and
    methods as well as further help.
    """

    def __init__(self, dim, dtype=float):
        """Initialize a new instance.

        Parameters
        ----------
        `dim` : `int`
            The dimension of the space
        `dtype` : `object`, optional  (Default: `float`)
            The data type for each vector entry. Can be provided in any
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.
            Only real floating-point types are allowed.
        """
        if not isinstance(dim, Integral) or dim < 1:
            raise TypeError(errfmt('''
            `dim` {} is not a positive integer.'''.format(dim)))

        dtype_ = np.dtype(dtype)
        if dtype_ not in (np.float16, np.float32, np.float64, np.float128):
            raise TypeError(errfmt('''
            `dtype` {} not a real floating-point type.'''.format(dtype)))

        super().__init__(dim, dtype_)
        self._field = RealNumbers()

    def _lincomb(self, z, a, x, b, y):
        """Linear combination of `x` and `y`.

        Calculate z = a * x + b * y using optimized BLAS routines
        if possible.

        Parameters
        ----------
        z : `Rn.Vector`
            The Vector to which the result is written.
        a, b : `RealNumber`
            Scalars to multiply `x` and `y` with.
        x, y : `Rn.Vector`
            The summands

        Returns
        -------
        None

        Examples
        --------
        >>> r3 = Rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([4, 5, 6])
        >>> z = r3.element()
        >>> r3.lincomb(z, 2, x, 3, y)
        >>> z
        Rn(3).element([14.0, 19.0, 24.0])
        """
        _lincomb(z, a, x, b, y, self.dtype)

    def zero(self):
        """Create a vector of zeros.

        Examples
        --------
        >>> r3 = Rn(3)
        >>> x = r3.zero()
        >>> x
        Rn(3).element([0.0, 0.0, 0.0])
        """
        return self.element(np.zeros(self.dim, dtype=self.dtype))

    @property
    def field(self):
        """The field of :math:`R^n`, i.e. the real numbers.

        Examples
        --------
        >>> r3 = Rn(3)
        >>> r3.field
        RealNumbers()
        """
        return self._field

    def _multiply(self, x, y):
        """The entry-wise product of two vectors, assigned to `y`.

        Parameters
        ----------
        x : `Rn.Vector`
            First factor
        y : `Rn.Vector`
            Second factor, used to store the result

        Returns
        -------
        None

        Examples
        --------
        >>> rn = Rn(3)
        >>> x = rn.element([5, 3, 2])
        >>> y = rn.element([1, 2, 3])
        >>> rn.multiply(x, y)
        >>> y
        Rn(3).element([5.0, 6.0, 6.0])
        """
        y.data[:] = x.data * y.data

    def __repr__(self):
        """`rn.__repr__() <==> repr(rn)`."""
        if self.dtype == np.float64:
            return 'Rn({})'.format(self.dim)
        else:
            return 'Rn({}, {!r})'.format(self.dim, self.dtype)

    def __str__(self):
        """`rn.__str__() <==> str(rn)`."""
        if self.dtype == np.float64:
            return 'Rn({})'.format(self.dim)
        else:
            return 'Rn({}, {})'.format(self.dim, self.dtype)

    class Vector(Ntuples.Vector, Algebra.Vector):

        """Representation of an `Rn` element.

        See also
        --------
        See `Ntuples.Vector`, `Algebra.Vector` and
        `LinearSpace.Vector` for attributes and methods.
        """

        def __init__(self, space, data):
            """Initialize a new instance."""
            if not isinstance(space, Rn):
                raise TypeError(errfmt('''
                `space` {!r} not an instance of `Rn`.
                '''.format(space)))

            super().__init__(space, data)


class MetricRn(Rn, MetricSpace):

    """The real space :math:`R^n` as a metric space.

    Its elements are represented as instances of the inner `Rn.Vector`
    class.

    See also
    --------
    See `Ntuples`, `Algebra`, `LinearSpace` and `MetricSpace` for
    attributes and methods as well as further help.
    """

    def __init__(self, dim, dist, dtype=float):
        """Initialize a new instance.

        Parameters
        ----------
        dim : int
            The dimension of the space
        dist : callable
            The distance function defining a metric on :math:`R^n`. It
            must accept two array arguments and fulfill the following
            conditions for any vectors `x`, `y` and `z`:

            - `dist(x, y) == dist(y, x)`
            - `dist(x, y) >= 0`
            - `dist(x, y) == 0` (approx.) if and only if `x == y`
              (approx.)
            - `dist(x, y) <= dist(x, z) + dist(z, y)`

        dtype : `object`, optional
            The data type for each vector entry. Can be provided in any
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.
            Only real floating-point types are allowed.
        """
        if not callable(dist):
            raise TypeError('`dist` {!r} not callable.'.format(dist))

        self._custom_dist = dist
        super().__init__(dim, dtype)

    def _dist(self, x, y):
        return self._custom_dist(x, y)

    def __repr__(self):
        """`rn.__repr__() <==> repr(rn)`."""
        if self.dtype == np.float64:
            return 'MetricRn({}, {!r})'.format(self.dim, self._dist)
        else:
            return 'MetricRn({}, {!r}, {!r})'.format(self.dim, self._dist,
                                                     self.dtype)

    def __str__(self):
        """`rn.__str__() <==> str(rn)`."""
        if self.dtype == np.float64:
            return 'MetricRn({})'.format(self.dim)
        else:
            return 'MetricRn({}, {})'.format(self.dim, self.dtype)

    class Vector(Rn.Vector, MetricSpace.Vector):

        """A MetricRn vector represented by a NumPy array.

        Attributes
        ----------

        +-----------+---------------+---------------------------------+
        |Name       |Type           |Description                      |
        +===========+===============+=================================+
        |`data`     |`numpy.ndarray`|The container for the vector     |
        |           |               |entries                          |
        +-----------+---------------+---------------------------------+
        |`data_ptr` |`int`          |A raw memory pointer to the data |
        |           |               |container. Can be processed with |
        |           |               |the `ctypes` module in Python.   |
        +-----------+---------------+---------------------------------+
        |`space`    |`Rn`           |The `LinearSpace` the vector     |
        |           |               |lives in                         |
        +-----------+---------------+---------------------------------+

        Methods
        -------

        +----------------+-----------+--------------------------------+
        |Signature       |Return type|Description                     |
        +================+===========+================================+
        |`assign(other)` |`None`     |Copy the values of `other` to   |
        |                |           |this vector.                    |
        +----------------+-----------+--------------------------------+
        |`copy()`        |`Rn.Vector`|Create a new vector and fill it |
        |                |           |with this vector's values.      |
        +----------------+-----------+--------------------------------+
        |`set_zero()`    |`None`     |Set all entries to 0.           |
        +----------------+-----------+--------------------------------+
        |`lincomb(a, x,  |`None`     |Assign the values of the linear |
        |b=None, y=None)`|           |combination `a * x + b * y` to  |
        |                |           |this vector.                    |
        +----------------+-----------+--------------------------------+
        |`equals(other)` |`boolean`  |Test if all entries of this     |
        |                |           |vector and `other` are equal.   |
        +----------------+-----------+--------------------------------+
        |`dist(y)`       |`float`    |The distance of this vector to  |
        |                |           |`y` as measured in the space    |
        |                |           |metric.                         |
        +----------------+-----------+--------------------------------+

        See also
        --------
        See `LinearSpace.Vector` and `MetricSpace.Vector` for a full
        list of attributes and methods.
        """

        def __init__(self, space, data):
            """Initialize a new instance."""
            if not isinstance(space, MetricRn):
                raise TypeError(errfmt('''
                `space` {!r} not an instance of `MetricRn`.
                '''.format(space)))

            super().__init__(space, data)


class NormedRn(MetricRn, NormedSpace):

    """The real space :math:`R^n` with the p-norm or a custom norm.

    # TODO: document public interface
    """

    def __init__(self, dim, norm, dtype=float):
        """Create a new NormedRn instance.

        Parameters
        ----------

        dim : `int`
            The dimension of the space
        norm : callable
            The norm implementation. It must accept an array-like
            argument, return a `RealNumber` and satisfy the following
            properties:

            - `norm(x) >= 0`
            - `norm(x) == 0` (approx.) only if `x == 0` (approx.)
            - `norm(s * x) == abs(s) * norm(x)` for `s` scalar
            - `norm(x + y) <= norm(x) + norm(y)`

        dtype : `object`, optional
            The data type for each vector entry. Can be provided in any
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.
            Only real floating-point types are allowed.
        """
        if not callable(norm):
            raise TypeError('`norm` {} is not callable.'.format(norm))

        def induced_dist(x, y):
            """The distance induced by the norm.

            Examples
            --------
            >>> import numpy as np
            >>> from functools import partial
            >>> r2 = NormedRn(2, norm=partial(np.linalg.norm, ord=1))
            >>> x = r2.element([3, 4])
            >>> y = r2.element([-1, 2])
            >>> r2.dist(x, y)
            6.0
            """
            return float(norm(x - y))

        super().__init__(dim, induced_dist, dtype)
        self._norm_impl = norm

    def _norm(self, x):
        """Calculate the norm of a vector.

        Parameters
        ----------
        x : `NormedRn.Vector`
            The vector whose norm is calculated

        Returns
        -------
        norm : `float`
            Norm of the vector

        Examples
        --------
        >>> import numpy as np
        >>> r2_2 = NormedRn(2, norm=np.linalg.norm)  # 2-norm
        >>> x = r2_2.element([3, 4])
        >>> r2_2.norm(x)
        5.0

        >>> from functools import partial
        >>> r2_1 = NormedRn(2, partial(np.linalg.norm, ord=1))
        >>> x = r2_1.element([3, 4])
        >>> r2_1.norm(x)
        7.0

        >>> r2_0 = NormedRn(2, partial(np.linalg.norm, ord=0))
        >>> x = r2_0.element([3, 0])
        >>> r2_0.norm(x)
        1.0
        """
        return float(self._norm_impl(x))

    def __repr__(self):
        """`rn.__repr__() <==> repr(rn)`."""
        if self.dtype == np.float64:
            return 'NormedRn({}, {!r})'.format(self.dim, self._norm)
        else:
            return 'NormedRn({}, {!r}, {!r})'.format(self.dim, self._norm,
                                                     self.dtype)

    def __str__(self):
        """`rn.__str__() <==> str(rn)`."""
        if self.dtype == np.float64:
            return 'NormedRn({})'.format(self.dim)
        else:
            return 'NormedRn({}, {})'.format(self.dim, self.dtype)

    class Vector(MetricRn.Vector, NormedSpace.Vector):

        """Representation of a `NormedRn` element."""

        def __init__(self, space, data):
            """Initialize a new instance."""
            if not isinstance(space, NormedRn):
                raise TypeError(errfmt('''
                `space` {!r} not an instance of `NormedRn`.
                '''.format(space)))

            super().__init__(space, data)


class HilbertRn(NormedRn, HilbertSpace):

    """The real space :math:`R^n` with an inner product.

    # TODO: document public interface
    """

    def __init__(self, dim, inner, dtype=float):
        """Initialize a new `HilbertRn` instance.

        Parameters
        ----------

        dim : `int`
            The dimension of the space
        inner : callable
            Create a HilbertRn with this inner product. It must take
            two `Rn` vectors as arguments, return a `RealNumber` and
            satisfy the following conditions for all vectors `x`, `y`
            and `z` and scalars `s`:

             - `inner(x, y) == inner(y, x)`
             - `inner(s * x, y) == s * inner(x, y)`
             - `inner(x + z, y) == inner(x, y) + inner(z, y)`
             - `inner(x, x) == 0` (approx.) only if `x == 0` (approx.)

        dtype : `object`, optional
            The data type for each vector entry. Can be provided in any
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.
            Only real floating-point types are allowed.
        """
        if not callable(inner):
            raise TypeError(errfmt('''
            `inner` {} not callable.'''.format(inner)))

        def induced_norm(x):
            """The norm induced by the inner product.

            Examples
            --------
            >>> import numpy as np
            >>> r2 = HilbertRn(2, inner=np.vdot)
            >>> x = r2.element([3, 4])
            >>> r2.norm(x)
            5.0
            """
            return sqrt(float(inner(x, x)))

        super().__init__(dim, induced_norm, dtype)
        self._innerprod = inner

    def _inner(self, x, y):
        """Raw inner product of two vectors.

        Parameters
        ----------

        x, y : `HilbertRn.Vector`
            The vectors whose inner product is calculated

        Returns
        -------
        inner : `float`
            Inner product of `x` and `y`.

        Examples
        --------
        >>> import numpy as np
        >>> r3 = HilbertRn(3, inner=np.vdot)
        >>> x = r3.element([5, 3, 2])
        >>> y = r3.element([1, 2, 3])
        >>> r3.inner(x, y) == 5*1 + 3*2 + 2*3
        True
        >>> weights = np.array([1., 2., 1.])
        >>> r3w = HilbertRn(3, lambda x, y: np.vdot(weights * x, y))
        >>> r3w.inner(x, y) == 1*5*1 + 2*3*2 + 1*2*3
        True
        """
        return float(self._innerprod(x, y))

    def __repr__(self):
        """`rn.__repr__() <==> repr(rn)`."""
        if self.dtype == np.float64:
            return 'HilbertRn({}, {!r})'.format(self.dim, self._inner)
        else:
            return 'HilbertRn({}, {!r}, {!r})'.format(self.dim, self._inner,
                                                      self.dtype)

    def __str__(self):
        """`rn.__str__() <==> str(rn)`."""
        if self.dtype == np.float64:
            return 'HilbertRn({})'.format(self.dim)
        else:
            return 'HilbertRn({}, {})'.format(self.dim, self.dtype)

    class Vector(NormedRn.Vector, HilbertSpace.Vector):

        """Representation of a `HilbertRn` element."""

        def __init__(self, space, data):
            """Initialize a new instance."""
            if not isinstance(space, HilbertRn):
                raise TypeError(errfmt('''
                `space` {!r} not an instance of `HilbertRn`.
                '''.format(space)))

            super().__init__(space, data)


class EuclideanRn(HilbertRn):

    """The `n`-dimensional standard Euclidean space."""

    def __init__(self, dim, dtype=float):
        """Initialize a new `EuclideanRn` instance.

        Parameters
        ----------

        dim : `int`
            The dimension of the space
        dtype : `object`, optional
            The data type for each vector entry. Can be provided in any
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.
            Only real floating-point types are allowed.

        Examples
        --------
        >>> r3 = EuclideanRn(3)
        >>> x = r3.element([5, 3, 2])
        >>> y = r3.element([1, 2, 3])
        >>> r3.inner(x, y) == 5*1 + 3*2 + 2*3
        True
        """
        super().__init__(dim, inner=np.vdot, dtype=dtype)

    def __repr__(self):
        """`rn.__repr__() <==> repr(rn)`."""
        if self.dtype == np.float64:
            return 'EuclideanRn({})'.format(self.dim)
        else:
            return 'EuclideanRn({}, {!r})'.format(self.dim, self.dtype)

    def __str__(self):
        """`rn.__str__() <==> str(rn)`."""
        if self.dtype == np.float64:
            return 'EuclideanRn({})'.format(self.dim)
        else:
            return 'EuclideanRn({}, {})'.format(self.dim, self.dtype)

    class Vector(HilbertRn.Vector):

        """Representation of a `EuclideanRn` element."""

        def __init__(self, space, data):
            """Initialize a new instance."""
            if not isinstance(space, EuclideanRn):
                raise TypeError(errfmt('''
                `space` {!r} not an instance of `EuclideanRn`.
                '''.format(space)))

            super().__init__(space, data)


class Cn(Ntuples, Algebra):

    """The complex vector space :math:`C^n` with vector multiplication.

    Its elements are represented as instances of the inner `Cn.Vector`
    class.

    Differences to `LinearSpace`
    ----------------------------

    Attributes
    ----------

    +--------+-------------+------------------------------------------+
    |Name    |Type         |Description                               |
    +========+=============+==========================================+
    |`dim`   |`int`        |The dimension `n` of the space :math:`C^n`|
    +--------+-------------+------------------------------------------+
    |`field` |             |`ComplexNumbers()`                        |
    +--------+-------------+------------------------------------------+
    |`dtype` |`type`       |The data dype of each vector entry        |
    +--------+-------------+------------------------------------------+

    Methods
    -------

    +-----------------+-----------+-----------------------------------+
    |Signature        |Return type|Description                        |
    +=================+===========+===================================+
    |`element         |`Cn.Vector`|Create an element in `Cn`. If      |
    |(inp=None)`      |           |`inp` is `None`, merely memory is  |
    |                 |           |allocated. Otherwise, the element  |
    |                 |           |is created from `inp`.             |
    +-----------------+-----------+-----------------------------------+
    |`zero()`         |`Cn.Vector`|Create the zero element, i.e., the |
    |                 |           |element where each entry is 0.     |
    +-----------------+-----------+-----------------------------------+
    |`multiply(x, y)` |`None`     |Calculate the entry-wise product of|
    |                 |           |`x` and `y` and assign the result  |
    |                 |           |to `y`.                            |
    +-----------------+-----------+-----------------------------------+

    See also
    --------
    See `LinearSpace` for a list of additional attributes and methods
    as well as further help.

    -------------------------------------------------------------------
    """

    def __init__(self, dim, dtype=complex):
        """Initialize a new `Cn` instance.

        Parameters
        ----------
        `dim` : `int`
            The dimension of the space
        `dtype` : `type`, optional  (Default: `complex`)
            The data type of the storage array. Can be provided in any
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.
            Only complex floating-point data types are allowed.
        """
        if not isinstance(dim, Integral) or dim < 1:
            raise TypeError(errfmt('''
            `dim` {} is not a positive integer.'''.format(dim)))

        # TODO: support separate storage of real and imag parts?
        dtype_ = np.dtype(dtype)
        if dtype_ not in (np.complex64, np.complex128, np.complex256):
            raise TypeError(errfmt('''
            `dtype` {} not a complex floating-point type.
            '''.format(dtype)))

        super().__init__(dim, dtype_)
        self._field = ComplexNumbers()

    def _lincomb(self, z, a, x, b, y):
        """Linear combination of `x` and `y`.

        Calculate z = a * x + b * y using optimized BLAS routines.

        Parameters
        ----------
        z : `Cn.Vector`
            The Vector that the result is written to.
        a, b : `ComplexNumber`
            Scalar to multiply `x` and `y` with.
        x, y : `Cn.Vector`
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

    def zero(self):
        """Create a vector of zeros.

        Examples
        --------
        >>> c3 = Cn(3)
        >>> x = c3.zero()
        >>> x
        Cn(3).element([0j, 0j, 0j])
        """
        return self.element(np.zeros(self.dim, dtype=self.dtype))

    @property
    def field(self):
        """The field of :math:`C^n`, i.e. the complex numbers.

        Examples
        --------
        >>> c3 = Cn(3)
        >>> c3.field
        ComplexNumbers()
        """
        return self._field

    def equals(self, other):
        """Check if `other` is a Cn instance of the same dimension.

        Parameters
        ----------
        other : any object
            The object to check for equality

        Returns
        -------
        equals : boolean

        Examples
        --------

        >>> c3 = Cn(3)
        >>> c3.equals(c3)
        True

        Equality is not identity:

        >>> c3a, c3b = Cn(3), Cn(3)
        >>> c3a.equals(c3b)
        True
        >>> c3a is c3b
        False

        >>> c3, c4 = Cn(3), Cn(4)
        >>> c3.equals(c4)
        False
        >>> c3_double, c3_single = Cn(3), Cn(3, dtype='csingle')
        >>> c3_double.equals(c3_single)
        False

        Equality can also be checked with "==":

        >>> c3, c4 = Cn(3), Cn(4)
        >>> c3 == c3
        True
        >>> c3 == c4
        False
        >>> c3 != c4
        True
        """
        return (isinstance(other, Cn) and
                self.dim == other.dim and
                self.dtype == other.dtype)

    def _multiply(self, x, y):
        """The entry-wise product of two vectors, assigned to `y`.

        Parameters
        ----------
        x : `Rn.Vector`
            First factor
        y : `Rn.Vector`
            Second factor, used to store the result

        Returns
        -------
        None

        Examples
        --------
        >>> c3 = Cn(3)
        >>> x = c3.element([5+1j, 3, 2-2j])
        >>> y = c3.element([1, 2+1j, 3-1j])
        >>> c3.multiply(x, y)
        >>> y
        Cn(3).element([(5+1j), (6+3j), (4-8j)])
        """
        y.data[:] *= x.data

    def __repr__(self):
        """repr() implementation."""
        if self.dtype == np.complex128:
            return 'Cn({})'.format(self.dim)
        else:
            return 'Cn({}, {!r})'.format(self.dim, self.dtype)

    def __str__(self):
        """str() implementation."""
        if self.dtype == np.complex128:
            return 'Cn({})'.format(self.dim)
        else:
            return 'Cn({}, {})'.format(self.dim, self.dtype)

    class Vector(Ntuples.Vector, Algebra.Vector):

        """A `Cn` vector represented with a NumPy array.

        Differences to `LinearSpace.Vector`
        -----------------------------------

        Attributes
        ----------

        +-----------+---------------+---------------------------------+
        |Name       |Type           |Description                      |
        +===========+===============+=================================+
        |`data`     |`numpy.ndarray`|The container for the vector     |
        |           |               |entries                          |
        +-----------+---------------+---------------------------------+
        |`data_ptr` |`int`          |A raw memory pointer to the data |
        |           |               |container. Can be processed with |
        |           |               |the `ctypes` module in Python.   |
        +-----------+---------------+---------------------------------+
        |`real`     |`Rn.Vector`    |Real part of this vector as an   |
        |           |(view)         |`Rn` vector view. This attribute |
        |           |               |is writable.                     |
        +-----------+---------------+---------------------------------+
        |`imag`     |`Rn.Vector`    |Imaginary part of this vector as |
        |           |(view)         |an `Rn` vector view. This        |
        |           |               |attribute is writable.           |
        +-----------+---------------+---------------------------------+

        Methods
        -------

        +----------------+--------------------+-----------------------+
        |Signature       |Return type         |Description            |
        +================+====================+=======================+
        |`equals(other)` |`boolean`           |Test if `other` is     |
        |                |                    |equal to this vector.  |
        +----------------+--------------------+-----------------------+

        See also
        --------
        See `LinearSpace.Vector` for a list of further attributes and
        methods.

        ---------------------------------------------------------------
        """

        def __init__(self, space, data):
            """Initialize a new `Cn.Vector` instance.

            Parameters
            ----------
            space : `Cn`
                Space instance this vector lives in
            data : `numpy.ndarray`
                Array that will be used as data representation. Its
                dtype must be equal to `space.dtype`, and its shape
                must be `(space.dim,)`.
            """
            if not isinstance(space, Cn):
                raise TypeError(errfmt('''
                `space` type {} is not `Cn`.
                '''.format(type(space))))

            super().__init__(space, data)

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
            return Rn(self.space.dim).element(self.data.real)

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
            return Rn(self.space.dim).element(self.data.imag)

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


# TODO: move - the requirement of CUDA for this module is bad!
#def cartesian(dim, impl='numpy', **kwargs):
#    """Create an n-dimensional Cartesian space, by default Euclidean.
#
#    Parameters
#    ----------
#
#    dim : int
#        The dimension of the space to be created
#    impl : str, optional
#        'numpy' : Use NumPy as backend for data storage and operations.
#                  This is the default.
#        'cuda' : Use CUDA as backend for data storage and operations
#                 (requires odlpp).
#    kwargs : {'dist', 'norm', 'norm_p', 'inner', 'weights'}
#        'dist' : callable or False
#            If False, create a plain Rn without further structure.
#            Otherwise, create a MetricRn with this distance function
#        'norm' : callable
#            Create a NormedRn with this norm function.
#            Cannot be combined with 'dist'.
#        'norm_p' : RealNumber
#            Create a NormedRn with the p-norm associated with 'norm_p'.
#            Cannot be combined with 'dist' or 'norm'.
#        'inner' : callable
#            Create a HilbertRn with this inner product function.
#            Cannot be combined with 'dist', 'norm' or 'norm_p'.
#        'weights' : array-like or float
#            Create a HilbertRn with the weighted dot product as inner
#            product, i.e., <x, y> = dot(x, weights*y).
#            Cannot be combined with 'dist', 'norm' or 'inner'.
#
#    Returns
#    -------
#
#    rn : instance of Rn, MetricRn, NormedRn or HilbertRn
#
#    See also
#    --------
#    Rn, MetricRn, NormedRn, HilbertRn
#
#    """
#    try:
#        impl = impl.lower()
#    except AttributeError:
#        raise TypeError("'impl' must be a string")
#
#    dist = kwargs.get('dist', None)
#    norm = kwargs.get('norm', None)
#    norm_p = kwargs.get('norm_p', None)
#    inner = kwargs.get('inner', None)
#    weights = kwargs.get('weights', None)
#
#    # Check if the parameter combination makes sense. The checks for
#    # correct types or values are delegated to the class initializers
#    if impl == 'numpy':
#        # 'dist' is processed first since int short-cuts to 'Rn' or
#        # 'MetricRn' if provided
#        if dist is False:
#            return Rn(dim)
#        elif dist is not None:
#            if norm is not None:
#                raise ValueError(errfmt('''
#                'dist' cannot be combined with 'norm' '''))
#            if norm_p is not None:
#                raise ValueError(errfmt('''
#                'dist' cannot be combined with 'norm_p' '''))
#            if inner is not None:
#                raise ValueError(errfmt('''
#                'dist' cannot be combined with 'inner' '''))
#            if weights is not None:
#                raise ValueError(errfmt('''
#                'dist' cannot be combined with 'weights' '''))
#
#            return MetricRn(dim, dist=dist)
#
#        # 'dist' not specified, continuing with 'norm' and 'norm_p'
#        if norm is not None and weights is not None:
#            raise ValueError(errfmt('''
#            'norm' cannot be combined with 'weights' '''))
#        elif norm is not None or norm_p is not None:
#            if inner is not None:
#                raise ValueError(errfmt('''
#                'norm' or 'norm_p' cannot be combined with 'inner' '''))
#
#            return NormedRn(dim, norm=norm, p=norm_p, weights=weights)
#        else:
#            # neither 'dist' nor 'norm' nor 'norm_p' specified,
#            # assuming inner product space
#            if inner is not None and weights is not None:
#                raise ValueError(errfmt('''
#                'inner' cannot be combined with 'weights' '''))
#            return HilbertRn(dim, inner=inner, weights=weights)
#
#    if impl == 'cuda':
#        if not CUDA_AVAILABLE:
#            raise ValueError(errfmt('''
#            CUDA implementation not available'''))
#
#        # TODO: move to CudaRn.__init__
#        if norm_p is not None:
#            raise NotImplementedError(errfmt('''
#            p-norms for p != 2.0 in CUDA spaces not implemented'''))
#
#        if norm is not None:
#            raise ValueError(errfmt('''
#            Custom norm implementation not possible for CUDA spaces'''))
#        if inner is not None:
#            raise ValueError(errfmt('''
#            Custom inner product implementation not possible for CUDA
#            spaces'''))
#
#        # TODO: move to CudaRn.__init__
#        if weights is not None:
#            raise NotImplementedError(errfmt('''
#            Weighted CUDA spaces not implemented'''))
#
#        return CudaRn(dim)
#
#    else:
#        raise ValueError(errfmt('''
#        Invalid value {} for 'impl' '''.format(impl)))


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
