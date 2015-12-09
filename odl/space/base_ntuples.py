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
from builtins import int

# External module imports
from abc import ABCMeta, abstractmethod
from math import sqrt
import numpy as np
import platform

# ODL imports
from odl.set.sets import Set, RealNumbers, ComplexNumbers
from odl.set.space import LinearSpace, LinearSpaceVector
from odl.util.utility import (
    array1d_repr, array1d_str, dtype_repr, with_metaclass,
    is_scalar_dtype, is_real_dtype, is_floating_dtype)
from odl.util.ufuncs import NtuplesBaseVectorUFuncs


__all__ = ('NtuplesBase', 'NtuplesBaseVector',
           'FnBase', 'FnBaseVector',
           'FnWeightingBase')


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


class NtuplesBase(Set):

    """Base class for sets of n-tuples independent of implementation."""

    def __init__(self, size, dtype):
        """Initialize a new instance.

        Parameters
        ----------
        size : non-negative int
            The number of entries per tuple
        dtype : `object`
            The data type for each tuple entry. Can be provided in any
            way the `numpy.dtype` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.
        """
        self._size = int(size)
        if self.size < 0:
            raise TypeError('size {!r} is not non-negative.'.format(size))
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
        """Return ``other in self``.

        Returns
        -------
        contains : `bool`
            `True` if ``other`` is an `NtuplesBaseVector` instance and
            ``other.space`` is equal to this space, `False` otherwise.

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
        return getattr(other, 'space', None) == self

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : `bool`
            `True` if ``other`` is an instance of this space's type
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
        """Return ``repr(self)``."""
        return '{}({}, {})'.format(self.__class__.__name__, self.size,
                                   dtype_repr(self.dtype))

    def __str__(self):
        """Return ``str(self)``."""
        return '{}({}, {})'.format(self.__class__.__name__, self.size,
                                   dtype_repr(self.dtype))

    @property
    def element_type(self):
        """ `NtuplesBaseVector` """
        return NtuplesBaseVector


class NtuplesBaseVector(with_metaclass(ABCMeta, object)):

    """Abstract class for representation of `NtuplesBase` elements.

    Defines abstract attributes and concrete ones which are
    independent of data representation.
    """

    # Give a `Vector` a higher priority than any NumPy array type. This
    # forces the usage of `__op__` of `Vector` if the other operand
    # is a NumPy object (applies also to scalars!).
    __array_priority__ = 1000000.0

    def __init__(self, space, *args, **kwargs):
        """Initialize a new instance."""
        self._space = space

    @abstractmethod
    def copy(self):
        """Create an identical (deep) copy of this vector."""

    @abstractmethod
    def asarray(self, start=None, stop=None, step=None, out=None):
        """Extract the data of this array as a numpy array.

        Parameters
        ----------
        start : `int`, optional
            Start position. `None` means the first element.
        start : `int`, optional
            One element past the last element to be extracted.
            `None` means the last element.
        start : `int`, optional
            Step length. `None` means 1.
        out : `numpy.ndarray`
            Array to write result to.

        Returns
        -------
        asarray : `numpy.ndarray`
            Numpy array of the same type as the space.
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
        values : `NtuplesBase.dtype` or `NtuplesBaseVector`
            The value(s) at the index (indices)
        """

    @abstractmethod
    def __setitem__(self, indices, values):
        """Set values of this vector.

        Parameters
        ----------
        indices : `int` or `slice`
            The position(s) that should be set
        values : {scalar, array-like, `NtuplesBaseVector`}
            The value(s) that are to be assigned.

            If ``index`` is an integer, ``value`` must be single value.

            If ``index`` is a slice, ``value`` must be broadcastable
            to the size of the slice (same size, shape (1,)
            or single value).
        """

    @abstractmethod
    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : `bool`
            `True` if all entries of ``other`` are equal to this
            vector's entries, `False` otherwise.
        """

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
        """Shape of this vector, equals ``(size,)``."""
        return (self.size,)

    @property
    def itemsize(self):
        """The size in bytes on one element of this type."""
        return self.dtype.itemsize

    @property
    def nbytes(self):
        """The number of bytes this vector uses in memory."""
        return self.size * self.itemsize

    def __len__(self):
        """Return ``len(self)``.

        Return the number of space dimensions.
        """
        return self.space.size

    def __array__(self, dtype=None):
        """Return a numpy array of this ntuple.

        Parameters
        ----------
        dtype : `object`
            Specifier for the data type of the output array

        Returns
        -------
        array : `numpy.ndarray`
        """
        if dtype is None:
            return self.asarray()
        else:
            return self.asarray().astype(dtype, copy=False)

    def __array_wrap__(self, obj):
        """Return a new vector from the data in obj.

        Parameters
        ----------
        obj : `numpy.ndarray`
            The array that should be wrapped

        Returns
        -------
            vector : `NtuplesBaseVector`
        """
        if obj.ndim == 0:
            return self.space.field.element(obj)
        else:
            return self.space.element(obj)

    def __ne__(self, other):
        """Return ``self != other``."""
        return not self.__eq__(other)

    def __str__(self):
        """Return ``str(self)``."""
        return array1d_str(self)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{!r}.element({})'.format(self.space,
                                         array1d_repr(self))

    @property
    def ufunc(self):
        """`NtuplesBaseVectorUFuncs`, access to numpy style ufuncs.

        These are always available, but may or may not be optimized for
        the specific space in use.
        """
        return NtuplesBaseVectorUFuncs(self)


class FnBase(NtuplesBase, LinearSpace):

    """Base class for :math:`F^n` independent of implementation."""

    def __init__(self, size, dtype):
        """Initialize a new instance.

        Parameters
        ----------
        size : `int`
            The number of dimensions of the space
        dtype : `object`
            The data type of the storage array. Can be provided in any
            way the `numpy.dtype` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.
            Only scalar data types (numbers) are allowed.
        """
        NtuplesBase.__init__(self, size, dtype)
        if not is_scalar_dtype(self.dtype):
            raise TypeError('{!r} is not a scalar data type.'.format(dtype))

        if is_real_dtype(self.dtype):
            field = RealNumbers()
            self._real_dtype = self.dtype
            self._is_real = True
        else:
            field = ComplexNumbers()
            self._real_dtype = _TYPE_MAP_C2R[self.dtype]
            self._is_real = False

        self._is_floating = is_floating_dtype(self.dtype)

        LinearSpace.__init__(self, field)

    @property
    def real_dtype(self):
        """The corresponding real data type of this space."""
        return self._real_dtype

    @property
    def is_rn(self):
        """If the space represents the set :math:`R^n`.

        Tuples of real numbers.
        """
        return self._is_real and self._is_floating

    @property
    def is_cn(self):
        """If the space represents the set :math:`C^n`.

        Tuples of complex numbers.
        """
        return (not self._is_real) and self._is_floating

    @abstractmethod
    def zero(self):
        """Create a vector of zeros."""

    @abstractmethod
    def one(self):
        """Create a vector of ones."""

    @abstractmethod
    def _multiply(self, x1, x2, out):
        """The entry-wise product of two vectors, assigned to ``out``."""

    @abstractmethod
    def _divide(self, x1, x2, out):
        """The entry-wise division of two vectors, assigned to ``out``."""

    @property
    def element_type(self):
        """ `FnBaseVector` """
        return FnBaseVector


class FnBaseVector(NtuplesBaseVector, LinearSpaceVector):

    """Abstract class for representation of `FnBase` vectors.

    Defines abstract attributes and concrete ones which are
    independent of data representation.
    """

    def __eq__(self, other):
        return LinearSpaceVector.__eq__(self, other)

    def copy(self):
        return LinearSpaceVector.copy(self)


class FnWeightingBase(object):

    """Abstract base class for weighting of `FnBase` spaces.

    This class and its subclasses serve as a simple means to evaluate
    and compare weighted inner products, norms and metrics semantically
    rather than by identity on a pure function level.

    The functions are implemented similarly to `Operator`,
    but without extra type checks of input parameters - this is done in
    the callers of the `LinearSpace` instance where these
    functions used.
    """

    def __init__(self, impl, exponent=2.0, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        impl : `str`
            Specifier for the implementation backend
        exponent : positive `float`
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
        dist_using_inner : `bool`, optional
            Calculate `dist` using the formula

            :math:`\lVert x-y \\rVert^2 = \lVert x \\rVert^2 +
            \lVert y \\rVert^2 - 2\Re \langle x, y \\rangle`.

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) :math:`x` and
            :math:`y`.

            This option can only be used if ``exponent`` is 2.0.

            Default: `False`.
        """
        self._dist_using_inner = bool(dist_using_inner)
        self._exponent = float(exponent)
        self._impl = str(impl).lower()
        if self._exponent <= 0:
            raise ValueError('only positive exponents or inf supported, '
                             'got {}.'.format(exponent))
        elif self._exponent != 2.0 and self._dist_using_inner:
            raise ValueError('`dist_using_inner` can only be used if the '
                             'exponent is 2.0.')

    @property
    def impl(self):
        """Implementation backend of this weighting."""
        return self._impl

    @property
    def exponent(self):
        """Exponent of this weighting."""
        return self._exponent

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equal : `bool`
            `True` if ``other`` is a the same weighting, `False`
            otherwise.

        Notes
        -----
        This operation must be computationally cheap, i.e. no large
        arrays may be compared element-wise. That is the task of the
        `equiv` method.
        """
        return (self.exponent == other.exponent and
                self._dist_using_inner == other._dist_using_inner and
                self.impl == other.impl)

    def equiv(self, other):
        """Test if ``other`` is an equivalent inner product.

        Should be overwritten, default tests for equality.

        Returns
        -------
        equivalent : `bool`
            `True` if ``other`` is a `FnWeightingBase` instance which
            yields the same result as this inner product for any
            input, `False` otherwise.
        """
        return self == other

    def inner(self, x1, x2):
        """Calculate the inner product of two vectors.

        Parameters
        ----------
        x1, x2 : `FnBaseVector`
            Vectors whose inner product is calculated

        Returns
        -------
        inner : `float` or `complex`
            The inner product of the two provided vectors
        """
        raise NotImplementedError

    def norm(self, x):
        """Calculate the norm of a vector.

        This is the standard implementation using `inner`.
        Subclasses should override it for optimization purposes.

        Parameters
        ----------
        x1 : `FnBaseVector`
            Vector whose norm is calculated

        Returns
        -------
        norm : `float`
            The norm of the vector
        """
        return float(sqrt(self.inner(x, x).real))

    def dist(self, x1, x2):
        """Calculate the distance between two vectors.

        This is the standard implementation using `norm`.
        Subclasses should override it for optimization purposes.

        Parameters
        ----------
        x1, x2 : `FnBaseVector`
            Vectors whose mutual distance is calculated

        Returns
        -------
        dist : `float`
            The distance between the vectors
        """
        if self._dist_using_inner:
            dist_squared = (self.norm(x1) ** 2 + self.norm(x2) ** 2 -
                            2 * self.inner(x1, x2).real)
            if dist_squared < 0:  # Compensate for numerical error
                dist_squared = 0.0
            return float(sqrt(dist_squared))
        else:
            return self.norm(x1 - x2)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
