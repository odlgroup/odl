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

"""CPU implementations of n-dimensional Cartesian spaces.

This is a default implementation of R^n and the corresponding MetricRn,
NormedRn and EuclideanRn. The data is represented by NumPy arrays.

+-------------+-------------------------------------------------------+
|Class name   |Description                                            |
+=============+=======================================================+
|`Rn`         |Basic class for spaces of `n`-tuples of real numbers   |
+-------------+-------------------------------------------------------+
|`MetricRn`   |`Rn` with a metric, i.e. a function to measure distance|
+-------------+-------------------------------------------------------+
|`NormedRn`   |`MetricRn` with a norm. The metric is derived from the |
|             |norm by the distance function                          |
|             |`dist(x, y) = norm(x - y)`                             |
+-------------+-------------------------------------------------------+
|`EuclideanRn`|`NormedRn` with an inner product. The norm is derived  |
|             |from the inner product according to the relation       |
|             |`norm(x) = sqrt(inner(x, x))`                          |
+-------------+-------------------------------------------------------+

In addition, a factory function for simple creation of Cartesian spaces
is provided:

+-------------------+-------------------------------------------------+
|Signature          |Description                                      |
+===================+=================================================+
|`cartesian(dim,    |Create a Cartesian space of R^n type. By default,|
|impl='numpy',      |the standard Euclidean space with the 2-norm and |
|**kwargs)`         |NumPy backend is created. See the function doc   |
|                   |for further options.                             |
+-------------------+-------------------------------------------------+
"""
# TODO: add other data types

# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from builtins import super
from future import standard_library

# External module imports
import numpy as np
from scipy.linalg.blas import get_blas_funcs
from numbers import Integral, Real
from math import sqrt

# ODL imports
from odl.space.set import Set, RealNumbers
from odl.space.space import (LinearSpace, MetricSpace, NormedSpace,
                             HilbertSpace, Algebra)
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

    +--------+-------------+------------------------------------------+
    |Name    |Type         |Description                               |
    +========+=============+==========================================+
    |`dim`   |`int`        |The number of entries per tuple           |
    +--------+-------------+------------------------------------------+
    |`dtype` |`type`       |The data dype of each tuple entry         |
    +--------+-------------+------------------------------------------+

    Methods
    -------

    +------------+----------------+-----------------------------------+
    |Signature   |Return type     |Description                        |
    +============+================+===================================+
    |`element    |`Ntuples.Vector`|Create an element in `Ntuples`. If |
    |(data=None)`|                |`data` is `None`, merely memory is |
    |            |                |allocated. Otherwise, the element  |
    |            |                |is created from `data`.            |
    +------------+----------------+-----------------------------------+

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
        """Initialize a new `Ntuples` instance.

        Parameters
        ----------
        `dim` : `int`
            The number entries per tuple
        `dtype` : `object`
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

    def element(self, data=None):
        """Create an `Ntuples` element.

        Parameters
        ----------
        data : array-like or `Ntuples.Vector`, optional
            The value(s) to fill the new array with.

            If an array is provided, it must have a shape of either
            `(1,)`, in which case the value is broadcasted, or
            `(dim,)`.

            If an `Ntuples.Vector` is given, its data container
            `data` is used as input array.

            Note that copying is avoided whenever possible, i.e.
            when the input data type matches the `Ntuples` data
            dype.

            **If data dypes match, the input array is only wrapped,
            not copied. Otherwise, the values are cast to the
            `Ntuples` data type.**

        Returns
        -------
        element : `Ntuples.Vector`

        Note
        ----
        If called without arguments, the values of the returned vector
        may be **anything**.

        Examples
        --------
        """
        if data is None:
            data = np.empty(self.dim, dtype=self.dtype)
        elif isinstance(data, Ntuples.Vector):
            return self.element(data.data)
        else:
            data = np.atleast_1d(data).astype(self.dtype, copy=False)

            if data.shape == (1,):
                data = np.repeat(data, self.dim)
            elif data.shape == (self.dim,):
                pass
            else:
                raise ValueError(errfmt('''
                `data` shape {} not broadcastable to shape ({}).
                '''.format(data.shape, self.dim)))

        return self.Vector(self, data)

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
        """Test if `other` is equal to this `Ntuples`.

        Parameters
        ----------
        `other` : `object`
            The object to check for equality

        Returns
        -------
        equals : boolean
            `True` if `other` is an `Ntuples` instance with the same
            `dim` and `dtype`, otherwise `False`.

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
        return (isinstance(other, Ntuples) and
                self.dim == other.dim and
                self.dtype == other.dtype)

    def contains(self, other):
        """Test if `other` is contained."""
        return isinstance(other, Ntuples.Vector) and other.space == self

    def __repr__(self):
        """repr() implementation."""
        return 'Ntuples({}, {!r})'.format(self.dim, self.dtype)

    def __str__(self):
        """str() implementation."""
        return 'Ntuples({}, {})'.format(self.dim, self.dtype)

    class Vector(object):

        """An `Ntuples` vector represented with a NumPy array.

        Not intended for creation of vectors, use the space's
        `element()` method instead.

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
            """Initialize a new `Ntuples.Vector` instance.

            Parameters
            ----------
            space : `Ntuples`
                Space instance this vector lives in
            data : `numpy.ndarray`
                Array that will be used as data representation. Its
                dtype must be equal to `space.dtype`, and its shape
                must be `(space.dim,)`.
            """
            if not isinstance(space, Ntuples):
                raise TypeError(errfmt('''
                `space` type {} is not `Ntuples`.
                '''.format(type(space))))

            if not isinstance(data, np.ndarray):
                raise TypeError(errfmt('''
                `data` type {} is not `numpy.ndarray`.
                '''.format(type(data))))

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

            In-place modification:

            >>> arr[0] = 5
            >>> vec
            Ntuples(3, dtype('int64')).element([5, 2, 3])
            """
            return self._data.ctypes.data

        def equals(self, other):
            """Test if `other` is equal to this vector.

            Parameters
            ----------
            `other` : `object`
                Object to compare to this vector

            Returns
            -------
            `equals` : `boolean`
                `True` if `other` is an element of this vector's
                space with equal entries, `False` otherwise.
            """
            return other in self.space and np.all(self.data == other.data)

        # Convenience functions
        def assign(self, other):
            """Assign the values of `other` to this vector.

            Parameters
            ----------
            `other` : `Ntuples.Vector`
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
            `copy` : `Ntuples.Vector`

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
            return self.space.element(self.data[:])

        def __len__(self):
            """The dimension of the underlying space.

            Examples
            --------
            >>> len(Ntuples(3, int).element())
            3
            """
            return self.space.dim

        def __getitem__(self, index):
            """Access values of this vector.

            Parameters
            ----------

            `index` : `int` or `slice`
                The position(s) that should be accessed

            Returns
            -------
            `value`: `space.dtype` or `numpy.ndarray`
                The value(s) at the index (indices)


            Examples
            --------

            >>> str_3 = Ntuples(3, dtype='S6')  # 6-char strings
            >>> x = str_3.element(['a', 'Hello!', '0'])
            >>> x[0]
            'a'
            >>> x[1:3]
            array(['Hello!', '0'], dtype='|S6')
            """
            return self.data.__getitem__(index)

        def __setitem__(self, index, value):
            """Set values of this vector.

            Parameters
            ----------

            `index` : `int` or `slice`
                The position(s) that should be set
            `value` : `space.dtype` or array-like
                The value(s) that are to be assigned.
                If `index` is an `int`, `value` must be single value
                of type `space.dtype`.
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
            >>> x[1:3] = [7, 8]
            >>> x
            Ntuples(3, dtype('int64')).element([5, 7, 8])
            >>> x[:] = np.array([0, 0, 0])
            >>> x
            Ntuples(3, dtype('int64')).element([0, 0, 0])

            Broadcasting is also supported:

            >>> x[1:3] = -2.
            >>> x
            Ntuples(3, dtype('int64')).element([0, -2, -2])
            """
            return self.data.__setitem__(index, value)

        def __eq__(self, other):
            """`vec.__eq__(other) <==> vec == other`."""
            return self.equals(other)

        def __ne__(self, other):
            """`vec.__ne__(other) <==> vec != other`."""
            return not self.equals(other)

        def __str__(self):
            """str() implementation."""
            return array1d_repr(self.data)

        def __repr__(self):
            """repr() implementation."""
            return '{!r}.element({})'.format(self.space,
                                             array1d_repr(self.data))


class Rn(Ntuples, LinearSpace):

    """The real vector space R^n without further structure.

    Its elements are represented as instances of the inner `Rn.Vector`
    class.

    Differences to `LinearSpace`
    ----------------------------

    Attributes
    ----------

    +--------+-------------+------------------------------------------+
    |Name    |Type         |Description                               |
    +========+=============+==========================================+
    |`dim`   |`int`        |The dimension `n` of the space R^n        |
    +--------+-------------+------------------------------------------+
    |`field` |             |Equal to `RealNumbers`                    |
    +--------+-------------+------------------------------------------+
    |`dtype` |`type`       |The data dype of each vector entry        |
    +--------+-------------+------------------------------------------+

    Methods
    -------

    +-----------------+-----------+-----------------------------------+
    |Signature        |Return type|Description                        |
    +=================+===========+===================================+
    |`element         |`Rn.Vector`|Create an element in `Rn`. If      |
    |(data=None)`     |           |`data` is `None`, merely memory is |
    |                 |           |allocated. Otherwise, the element  |
    |                 |           |is created from `data`.            |
    +-----------------+-----------+-----------------------------------+
    |`zero()`         |`Rn.Vector`|Create the zero element, i.e., the |
    |                 |           |element where each entry is 0.     |
    +-----------------+-----------+-----------------------------------+

    See also
    --------
    See `LinearSpace` for a full list of attributes and methods as well
    as further help.
    """

    def __init__(self, dim, dtype=float):
        """Initialize a new `Rn` instance.

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
        # pylint: disable=unbalanced-tuple-unpacking
        if not isinstance(dim, Integral) or dim < 1:
            raise TypeError(errfmt('''
            `dim` {} is not a positive integer.'''.format(dim)))

        dtype_ = np.dtype(dtype)
        if dtype_ not in (np.float16, np.float32, np.float64, np.float128):
            raise TypeError(errfmt('''
            `dtype` {} not a real floating-point type.'''.format(dtype)))

        def fallback_axpy(a, x, y):
            """Fallback axpy implementation."""
            return a * x + y

        def fallback_scal(a, x):
            """Fallback scal implementation."""
            return a * x

        def fallback_copy(x, y):
            """Fallback copy implementation."""
            y[...] = x[...]
            return y

        blas_axpy, blas_scal, blas_copy = get_blas_funcs(
            ['axpy', 'scal', 'copy'], dtype=dtype_)

        self._dim = dim
        self._field = RealNumbers()
        self._dtype = dtype_
        if dtype_ in (np.float32, np.float64):
            self._axpy = blas_axpy
            self._scal = blas_scal
            self._copy = blas_copy
        else:
            self._axpy = fallback_axpy
            self._scal = fallback_scal
            self._copy = fallback_copy

    def _lincomb(self, z, a, x, b, y):
        """Linear combination of `x` and `y`.

        Calculate z = a*x + b*y using optimized BLAS routines.

        Parameters
        ----------
        z : Rn.Vector
            The Vector that the result is written to.
        a : RealNumber
            Scalar to multiply `x` with.
        x : Rn.Vector
            The first of the summands
        b : RealNumber
            Scalar to multiply `y` with.
        y : Rn.Vector
            The second of the summands

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
        if x is y and b != 0:
            # x is aligned with y -> z = (a+b)*x
            self._lincomb(z, a+b, x, 0, x)
        elif z is x and z is y:
            # All the vectors are aligned -> z = (a+b)*z
            self._scal(a+b, z.data)
        elif z is x:
            # z is aligned with x -> z = a*z + b*y
            if a != 1:
                self._scal(a, z.data)
            if b != 0:
                self._axpy(y.data, z.data, self.dim, b)
        elif z is y:
            # z is aligned with y -> z = a*x + b*z
            if b != 1:
                self._scal(b, z.data)
            if a != 0:
                self._axpy(x.data, z.data, self.dim, a)
        else:
            # We have exhausted all alignment options, so x != y != z
            # We now optimize for various values of a and b
            if b == 0:
                if a == 0:  # Zero assignment -> z = 0
                    z.data[:] = 0
                else:  # Scaled copy -> z = a*x
                    self._copy(x.data, z.data)
                    if a != 1:
                        self._scal(a, z.data)
            else:
                if a == 0:  # Scaled copy -> z = b*y
                    self._copy(y.data, z.data)
                    if b != 1:
                        self._scal(b, z.data)

                elif a == 1:  # No scaling in x -> z = x + b*y
                    self._copy(x.data, z.data)
                    self._axpy(y.data, z.data, self.dim, b)
                else:  # Generic case -> z = a*x + b*y
                    self._copy(y.data, z.data)
                    if b != 1:
                        self._scal(b, z.data)
                    self._axpy(x.data, z.data, self.dim, a)

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
        """The field of R^n, i.e. the real numbers.

        Examples
        --------

        >>> r3 = Rn(3)
        >>> r3.field
        RealNumbers()
        """
        return self._field

    def equals(self, other):
        """Check if `other` is an Rn instance of the same dimension.

        Parameters
        ----------
        other : any object
            The object to check for equality

        Returns
        -------
        equals : boolean

        Examples
        --------

        >>> r3 = Rn(3)
        >>> r3.equals(r3)
        True

        Equality is not identity:

        >>> r3a, r3b = Rn(3), Rn(3)
        >>> r3a.equals(r3b)
        True
        >>> r3a is r3b
        False

        >>> r3, r4 = Rn(3), Rn(4)
        >>> r3.equals(r4)
        False
        >>> r3_double, r3_single = Rn(3), Rn(3, dtype='single')
        >>> r3_double.equals(r3_single)
        False

        Equality can also be checked with "==":

        >>> r3, r4 = Rn(3), Rn(4)
        >>> r3 == r3
        True
        >>> r3 == r4
        False
        >>> r3 != r4
        True
        """
        return (isinstance(other, Rn) and
                self.dim == other.dim and
                self.dtype == other.dtype)

    def __repr__(self):
        """repr() implementation."""
        if self.dtype == np.float64:
            return 'Rn({})'.format(self.dim)
        else:
            return 'Rn({}, {!r})'.format(self.dim, self.dtype)

    def __str__(self):
        """str() implementation."""
        if self.dtype == np.float64:
            return 'Rn({})'.format(self.dim)
        else:
            return 'Rn({}, {})'.format(self.dim, self.dtype)

    class Vector(Ntuples.Vector, LinearSpace.Vector):

        """An `Rn` vector represented with a NumPy array.

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
            """Initialize a new `Rn.Vector` instance.

            Parameters
            ----------
            space : `Rn`
                Space instance this vector lives in
            data : `numpy.ndarray`
                Array that will be used as data representation. Its
                dtype must be equal to `space.dtype`, and its shape
                must be `(space.dim,)`.
            """
            if not isinstance(space, Rn):
                raise TypeError(errfmt('''
                `space` type {} is not `Rn`.
                '''.format(type(space))))

            super().__init__(space, data)

        def __str__(self):
            """str() implementation."""
            return array1d_repr(self.data)

        def __repr__(self):
            """repr() implementation."""
            return '{!r}.element({})'.format(self.space, array1d_repr(self))


class MetricRn(Rn, MetricSpace):

    """The real space R^n as a metric space without norm.

    Its elements are represented as instances of the inner `Rn.Vector`
    class.

    Attributes
    ----------

    +--------+-------------+------------------------------------------+
    |Name    |Type         |Description                               |
    +========+=============+==========================================+
    |`dim`   |`int`        |The dimension `n` of the space R^n        |
    +--------+-------------+------------------------------------------+
    |`field` |`RealNumbers`|The type of scalars upon which the space  |
    |        |             |is built                                  |
    +--------+-------------+------------------------------------------+

    Methods
    -------

    +-----------------+-----------+-----------------------------------+
    |Signature        |Return type|Description                        |
    +=================+===========+===================================+
    |`element         |`Rn.Vector`|Create an element in `Rn`. If      |
    |(data=None)`     |           |`data` is `None`, merely memory is |
    |                 |           |allocated. Otherwise, the element  |
    |                 |           |is created from `data`.            |
    +-----------------+-----------+-----------------------------------+
    |`zero()`         |`Rn.Vector`|Create the zero element, i.e., the |
    |                 |           |element where each entry is 0.     |
    +-----------------+-----------+-----------------------------------+
    |`dist(x, y)`     |`float`    |Distance between two elements      |
    +-----------------+-----------+-----------------------------------+
    |`equals(other)`  |`boolean`  |Implements `self == other`.        |
    +-----------------+-----------+-----------------------------------+
    |`contains(other)`|`boolean`  |Implements `other in self`.        |
    +-----------------+-----------+-----------------------------------+
    |`lincomb(z, a, x,|`None`     |Linear combination `a * x + b * y`,|
    |b=None, y=None)` |           |stored in `z`.                     |
    +-----------------+-----------+-----------------------------------+
    """

    def __init__(self, dim, dist):
        """Create a MetricRn.

        Parameters
        ----------

        dim : int
            The dimension of the space
        dist : callable
            The distance function defining a metric on R^n. It must accept
            two array arguments and fulfill the following conditions:

            - `dist(x, y) = dist(y, x)`
            - `dist(x, y) >= 0`
            - `dist(x, y) == 0` (or close to 0) if and only if `x == y`
        """
        if not callable(dist):
            raise TypeError('`dist` {!r} not callable.'.format(dist))

        self._custom_dist = dist
        super().__init__(dim)

    def _dist(self, x, y):
        return self._custom_dist(x, y)

    def __repr__(self):
        """repr() implementation."""
        return 'MetricRn({})'.format(self.dim)

    def __str__(self):
        """str() implementation."""
        return self.__repr__()

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

        ---------------------------------------------------------------

        Parameters
        ----------

        space : Rn
            Space instance this vector lives in
        data : numpy.ndarray
            Array that will be used as data representation. Its dtype
            must be float64, and its shape must be (dim,).
        """


class NormedRn(Rn, NormedSpace):

    """The real space R^n with the p-norm or a custom norm.

    # TODO: document public interface
    """

    def __init__(self, dim, p=None, **kwargs):
        """Create a new NormedRn instance.

        Parameters
        ----------

        dim : int
            The dimension of the space
        p : float, optional
            The order of the norm. Default: 2.0
        kwargs: {'weights', 'norm'}
                'weights': array-like, optional
                    Array of weights to be used in the norm calculation.
                    It must be broadcastable to size (n,). All entries
                    must be positive.
                    Cannot be combined with 'norm'.
                'norm': callable, optional
                    A custom norm to use instead of the p-norm. Cannot be
                    combined with 'p'.

        Notes
        -----

        The following values for `p` can be specified.
        Note that any value of p < 1 only gives a pseudonorm.

        =====  ====================================================
        p      Definition
        =====  ====================================================
        inf    max(abs(x[0]), ..., abs(x[n-1]))
        -inf   min(abs(x[0]), ..., abs(x[n-1]))
        0      (x[0] != 0 + ... + x[n-1] != 0)
        other  (abs(x[0])**p + ... + abs(x[n-1])**p)**(1/p)
        =====  ====================================================
        """
        weights = kwargs.get('weights', None)
        norm = kwargs.get('norm', None)

        if p is not None and norm is not None:
            raise ValueError(errfmt('''
            'p' and 'norm' cannot be combined.'''))

        p = p if p is not None else 2.0
        if not isinstance(p, Real):
            raise TypeError(errfmt('''
            'p' ({}) must be a real number.'''.format(p)))

        if weights is not None:
            if norm is not None:
                raise ValueError(errfmt('''
                'weights' and 'norm' cannot be combined.'''))
            try:
                weights = np.atleast_1d(weights)
            except TypeError:
                raise TypeError(errfmt('''
                'weights' ({}) must be array-like.'''.format(weights)))

            if not np.all(weights > 0):
                raise ValueError(errfmt('''
                'weights' must be all positive'''))

        if norm is not None and not callable(norm):
            raise TypeError("'norm' must be callable.")

        self._p = p
        self._sqrt_weights = np.sqrt(weights) if weights is not None else None
        self._custom_norm = norm

        super().__init__(dim)

    def _norm(self, vector):
        """Calculate the norm of a vector.

        Parameters
        ----------

        vector : NormedRn.Vector

        Returns
        -------
        norm : float
            Norm of the vector

        Examples
        --------

        >>> r2_2 = NormedRn(2, p=2)
        >>> x = r2_2.element([3, 4])
        >>> r2_2.norm(x)
        5.0

        >>> r2_1 = NormedRn(2, p=1)
        >>> x = r2_1.element([3, 4])
        >>> r2_1.norm(x)
        7.0

        >>> r2_0 = NormedRn(2, p=0)
        >>> x = r2_0.element([3, 0])
        >>> r2_0.norm(x)
        1.0

        """
        if self._custom_norm is not None:
            return self._custom_norm(vector.data)
        elif self._sqrt_weights is None:
            return np.linalg.norm(vector.data, ord=self._p)
        else:
            return np.linalg.norm(vector.data * self._sqrt_weights,
                                  ord=self._p)

    def __repr__(self):
        """repr() implementation."""
        return 'NormedRn({})'.format(self.dim)

    def __str__(self):
        """str() implementation."""
        return self.__repr__()

    class Vector(Rn.Vector, NormedSpace.Vector):

        """A NormedRn vector represented by a NumPy array.

        Parameters
        ----------

        space : NormedRn
            Space instance this vector lives in
        data : numpy.ndarray
            Array that will be used as data representation. Its dtype
            must be float64, and its shape must be (n,).
        """


class EuclideanRn(Rn, HilbertSpace, Algebra):

    """The real space R^n with the an inner product.

    # TODO: document public interface
    """

    def __init__(self, dim, **kwargs):
        """Create a new EuclideanRn instance.

        Parameters
        ----------

        dim : int
            The dimension of the space

        kwargs : {'inner', 'weights'}
            'inner' : callable
                Create a EuclideanRn with this inner product. It must take
                two Rn vectors as arguments and return a RealNumber.
            'weights' : array-like or float
                Create a EuclideanRn with the weighted dot product as inner
                product, i.e., <x, y> = dot(x, weights*y).
                'weights' must be broadcastable to shape (n,) and all
                entries must be positive.
                Cannot be combined with 'inner'.
        """
        weights = kwargs.get('weights', None)
        inner = kwargs.get('inner', None)

        if weights is not None:
            if inner is not None:
                raise ValueError(errfmt('''
                'weights' and 'inner' cannot be combined.'''))
            try:
                weights = np.atleast_1d(weights)
            except TypeError:
                raise TypeError(errfmt('''
                'weights' ({}) must be array-like.'''.format(weights)))

            if not np.all(weights > 0):
                raise ValueError(errfmt('''
                'weights' must be all positive'''))

        if inner is not None and not callable(inner):
            raise TypeError("'inner' must be callable.")

        self._weights = weights
        self._custom_inner = inner
        self._dot = get_blas_funcs(['dot'])[0]

        super().__init__(dim)

    def _norm(self, x):
        """Calculate the norm of a vector.

        norm(x) := sqrt(inner(x, x)).

        Parameters
        ----------

        x : EuclideanRn.Vector

        Returns
        -------
        norm : float
               Norm of the vector

        Examples
        --------

        >>> rn = EuclideanRn(2)
        >>> x = rn.element([3, 4])
        >>> rn.norm(x)
        5.0

        """
        return sqrt(self._inner(x, x))

    def _inner(self, x, y):
        """Calculate the inner product of two vectors.

        This is defined as:

        inner(x,y) := x[0]*y[0] + x[1]*y[1] + ... x[n-1]*y[n-1]

        Parameters
        ----------

        x : EuclideanRn.Vector
        y : EuclideanRn.Vector

        Returns
        -------
        inner : float
            Inner product of x and y.

        Examples
        --------

        >>> rn = EuclideanRn(3)
        >>> x = rn.element([5, 3, 2])
        >>> y = rn.element([1, 2, 3])
        >>> rn.inner(x, y) == 5*1 + 3*2 + 2*3
        True

        """
        if self._custom_inner is not None:
            return self._custom_inner(x.data, y.data)
        elif self._weights is None:
            return float(self._dot(x.data, y.data))
        else:
            return float(self._dot(x.data, self._weights * y.data))

    def _multiply(self, x, y):
        """The pointwise product of two vectors, assigned to `y`.

        This is defined as:

        multiply(x,y) := [x[0]*y[0], x[1]*y[1], ..., x[n-1]*y[n-1]]

        Parameters
        ----------

        x : EuclideanRn.Vector
            read from
        y : EuclideanRn.Vector
            read from and written to

        Returns
        -------
        None

        Examples
        --------

        >>> rn = EuclideanRn(3)
        >>> x = rn.element([5, 3, 2])
        >>> y = rn.element([1, 2, 3])
        >>> rn.multiply(x, y)
        >>> y
        EuclideanRn(3).element([5.0, 6.0, 6.0])
        """
        y.data[:] = x.data * y.data

    def __repr__(self):
        """repr() implementation."""
        return 'EuclideanRn({})'.format(self.dim)

    def __str__(self):
        """str() implementation."""
        return self.__repr__()

    class Vector(Rn.Vector, HilbertSpace.Vector, Algebra.Vector):

        """A EuclideanRn-vector represented using numpy.

        Parameters
        ----------

        space : EuclideanRn
            Space instance this vector lives in
        data : numpy.ndarray
            Array that will be used as data representation. Its dtype
            must be float64, and its shape must be (dim,).
        """


def cartesian(dim, impl='numpy', **kwargs):
    """Create an n-dimensional Cartesian space, by default Euclidean.

    Parameters
    ----------

    dim : int
        The dimension of the space to be created
    impl : str, optional
        'numpy' : Use NumPy as backend for data storage and operations.
                  This is the default.
        'cuda' : Use CUDA as backend for data storage and operations
                 (requires odlpp).
    kwargs : {'dist', 'norm', 'norm_p', 'inner', 'weights'}
        'dist' : callable or False
            If False, create a plain Rn without further structure.
            Otherwise, create a MetricRn with this distance function
        'norm' : callable
            Create a NormedRn with this norm function.
            Cannot be combined with 'dist'.
        'norm_p' : RealNumber
            Create a NormedRn with the p-norm associated with 'norm_p'.
            Cannot be combined with 'dist' or 'norm'.
        'inner' : callable
            Create a EuclideanRn with this inner product function.
            Cannot be combined with 'dist', 'norm' or 'norm_p'.
        'weights' : array-like or float
            Create a EuclideanRn with the weighted dot product as inner
            product, i.e., <x, y> = dot(x, weights*y).
            Cannot be combined with 'dist', 'norm' or 'inner'.

    Returns
    -------

    rn : instance of Rn, MetricRn, NormedRn or EuclideanRn

    See also
    --------
    Rn, MetricRn, NormedRn, EuclideanRn

    """
    try:
        impl = impl.lower()
    except AttributeError:
        raise TypeError("'impl' must be a string")

    dist = kwargs.get('dist', None)
    norm = kwargs.get('norm', None)
    norm_p = kwargs.get('norm_p', None)
    inner = kwargs.get('inner', None)
    weights = kwargs.get('weights', None)

    # Check if the parameter combination makes sense. The checks for
    # correct types or values are delegated to the class initializers
    if impl == 'numpy':
        # 'dist' is processed first since int short-cuts to 'Rn' or
        # 'MetricRn' if provided
        if dist is False:
            return Rn(dim)
        elif dist is not None:
            if norm is not None:
                raise ValueError(errfmt('''
                'dist' cannot be combined with 'norm' '''))
            if norm_p is not None:
                raise ValueError(errfmt('''
                'dist' cannot be combined with 'norm_p' '''))
            if inner is not None:
                raise ValueError(errfmt('''
                'dist' cannot be combined with 'inner' '''))
            if weights is not None:
                raise ValueError(errfmt('''
                'dist' cannot be combined with 'weights' '''))

            return MetricRn(dim, dist=dist)

        # 'dist' not specified, continuing with 'norm' and 'norm_p'
        if norm is not None and weights is not None:
            raise ValueError(errfmt('''
            'norm' cannot be combined with 'weights' '''))
        elif norm is not None or norm_p is not None:
            if inner is not None:
                raise ValueError(errfmt('''
                'norm' or 'norm_p' cannot be combined with 'inner' '''))

            return NormedRn(dim, norm=norm, p=norm_p, weights=weights)
        else:
            # neither 'dist' nor 'norm' nor 'norm_p' specified,
            # assuming inner product space
            if inner is not None and weights is not None:
                raise ValueError(errfmt('''
                'inner' cannot be combined with 'weights' '''))
            return EuclideanRn(dim, inner=inner, weights=weights)

    if impl == 'cuda':
        if not CUDA_AVAILABLE:
            raise ValueError(errfmt('''
            CUDA implementation not available'''))

        # TODO: move to CudaRn.__init__
        if norm_p is not None:
            raise NotImplementedError(errfmt('''
            p-norms for p != 2.0 in CUDA spaces not implemented'''))

        if norm is not None:
            raise ValueError(errfmt('''
            Custom norm implementation not possible for CUDA spaces'''))
        if inner is not None:
            raise ValueError(errfmt('''
            Custom inner product implementation not possible for CUDA
            spaces'''))

        # TODO: move to CudaRn.__init__
        if weights is not None:
            raise NotImplementedError(errfmt('''
            Weighted CUDA spaces not implemented'''))

        return CudaRn(dim)

    else:
        raise ValueError(errfmt('''
        Invalid value {} for 'impl' '''.format(impl)))


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
