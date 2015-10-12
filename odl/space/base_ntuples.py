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

"""Base classes for implementation of n-tuples."""

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

# ODL imports
from odl.set.sets import Set, RealNumbers, ComplexNumbers
from odl.set.space import LinearSpace
from odl.util.utility import array1d_repr, array1d_str, dtype_repr
from odl.util.utility import is_real_dtype


__all__ = ()


class NtuplesBase(with_metaclass(ABCMeta, Set)):

    """Base class for sets of n-tuples independent of implementation."""

    def __init__(self, size, dtype):
        """Initialize a new instance.

        Parameters
        ----------
        size : non-negative int
            The number of entries per tuple
        dtype : object
            The data type for each tuple entry. Can be provided in any
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.
        """
        self._size = int(size)
        if self._size < 0:
            raise TypeError('size {} is not non-negative.'.format(size))
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
        >>> from odl import Ntuples
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
        >>> from odl import Ntuples
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

        return ((isinstance(self, type(other)) or
                 isinstance(other, type(self))) and
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

        @property
        def itemsize(self):
            """The size in bytes on one element of this type."""
            return self.dtype.itemsize

        @property
        def nbytes(self):
            """The number of bytes this vector uses in memory."""
            return self.size * self.itemsize

        @abstractmethod
        def copy(self):
            """Create an identical (deep) copy of this vector."""

        @abstractmethod
        def asarray(self, start=None, stop=None, step=None):
            """Extract the data of this array as a numpy array.

            Parameters
            ----------
            start : int, optional
                Start position. `None` means the first element.
            start : int, optional
                One element past the last element to be extracted.
                `None` means the last element.
            start : int, optional
                Step length. `None` means 1.

            Returns
            -------
            asarray : `numpy.ndarray`
                Numpy array of the same type as the space.
            """

        def __len__(self):
            """`v.__len__() <==> len(v)`.

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
            raise TypeError('{!r} is not a scalar data type.'.format(dtype))

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


class _FnWeightingBase(with_metaclass(ABCMeta, object)):

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
        raise NotImplementedError

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
        return float(sqrt(self.inner(x, x).real))

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


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
