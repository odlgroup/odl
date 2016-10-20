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

"""Base classes for implementations of n-tuples."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import int

from abc import ABCMeta, abstractmethod
import numpy as np

from odl.set import (Set, RealNumbers, ComplexNumbers, LinearSpace,
                     LinearSpaceElement)
from odl.util.ufuncs import NtuplesBaseUFuncs
from odl.util.utility import (
    array1d_repr, array1d_str, dtype_repr, with_metaclass,
    is_scalar_dtype, is_real_dtype, is_floating_dtype,
    complex_dtype, real_dtype)


__all__ = ('NtuplesBase', 'NtuplesBaseVector', 'FnBase', 'FnBaseVector')


class NtuplesBase(Set):

    """Base class for sets of n-tuples independent of implementation."""

    def __init__(self, size, dtype):
        """Initialize a new instance.

        Parameters
        ----------
        size : non-negative int
            Number of entries in a tuple.
        dtype :
            Data type for each tuple entry. Can be provided in any
            way the `numpy.dtype` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.
        """
        self.__size = int(size)
        if self.size < 0:
            raise ValueError('`size` must be non-negative, got {}'
                             ''.format(size))
        self.__dtype = np.dtype(dtype)

    @property
    def dtype(self):
        """Data type of each entry."""
        return self.__dtype

    @property
    def size(self):
        """Number of entries per tuple."""
        return self.__size

    @property
    def shape(self):
        """Shape ``(size,)`` of this space."""
        return (self.size,)

    def __contains__(self, other):
        """Return ``other in self``.

        Returns
        -------
        contains : bool
            ``True`` if ``other`` is an `NtuplesBaseVector` instance and
            ``other.space`` is equal to this space, ``False`` otherwise.

        Examples
        --------
        >>> long_3 = odl.ntuples(3, dtype='int64')
        >>> long_3.element() in long_3
        True
        >>> long_3.element() in odl.ntuples(3, dtype='int32')
        False
        >>> long_3.element() in odl.ntuples(3, dtype='float64')
        False
        """
        return getattr(other, 'space', None) == self

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if ``other`` is an instance of this space's type
            with the same `size` and `dtype`, ``False`` otherwise.

        Examples
        --------
        >>> int_3 = odl.ntuples(3, dtype=int)
        >>> int_3 == int_3
        True

        Equality is not identity:

        >>> int_3a, int_3b = odl.ntuples(3, int), odl.ntuples(3, int)
        >>> int_3a == int_3b
        True
        >>> int_3a is int_3b
        False

        >>> int_3, int_4 = odl.ntuples(3, int), odl.ntuples(4, int)
        >>> int_3 == int_4
        False
        >>> int_3, str_3 = odl.ntuples(3, 'int'), odl.ntuples(3, 'S2')
        >>> int_3 == str_3
        False
        """
        # Optimization for simple cases
        if other is self:
            return True
        elif other is None:
            return False

        return ((isinstance(self, type(other)) or
                 isinstance(other, type(self))) and
                self.size == other.size and
                self.dtype == other.dtype)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({}, {})'.format(self.__class__.__name__, self.size,
                                   dtype_repr(self.dtype))

    @property
    def element_type(self):
        """Type of elements in this space."""
        raise NotImplementedError('abstract method')

    @staticmethod
    def available_dtypes():
        """Available data types for this space type.

        Returns
        -------
        available_dtypes : sequence
        """
        raise NotImplementedError('abstract method')


class NtuplesBaseVector(with_metaclass(ABCMeta, object)):

    """Abstract class for `NtuplesBase` elements.

    Do not use this class directly -- to create an element of a vector
    space, call the space's `LinearSpace.element` method instead.
    """

    def __init__(self, space, *args, **kwargs):
        """Initialize a new instance."""
        self.__space = space

    @abstractmethod
    def copy(self):
        """Return an identical (deep) copy of this vector."""

    @abstractmethod
    def asarray(self, start=None, stop=None, step=None, out=None):
        """Return the data of this vector as a numpy array.

        Parameters
        ----------
        start : int, optional
            Index of the first vector entry to be included in
            the extracted array. ``None`` is equivalent to 0.
        stop : int, optional
            Index of the first vector entry to be excluded from
            the extracted array. ``None`` is equivalent to `size`.
        step : int, optional
            Vector index step between consecutive array ellements.
            ``None`` is equivalent to 1.
        out : `numpy.ndarray`
            Array to write the result to.

        Returns
        -------
        out : `numpy.ndarray`
            Numpy array of the same `dtype` as this vector. If ``out``
            was given, the returned object is a reference to it.
        """

    @abstractmethod
    def __getitem__(self, indices):
        """Return ``self[indices]``.

        Parameters
        ----------
        indices : int or `slice`
            The position(s) that should be accessed. An integer results
            in a single entry to be returned. For a slice, the output
            is a vector of the same type.

        Returns
        -------
        values : `NtuplesBase.dtype` or `NtuplesBaseVector`
            Extracted entries according to ``indices``.
        """

    @abstractmethod
    def __setitem__(self, indices, values):
        """Implement ``self[indices] = values``.

        Parameters
        ----------
        indices : int or `slice`
            The position(s) that should be assigned to.
        values : scalar, `array-like` or `NtuplesBaseVector`
            The value(s) that are to be assigned.

            If ``index`` is an integer, ``value`` must be a single
            value.

            If ``index`` is a slice, ``value`` must be broadcastable
            to the shape of the slice, i.e. same size, shape ``(1,)``
            or a single value.
        """

    @abstractmethod
    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if all entries of ``other`` are equal to this
            vector's entries, False otherwise.
        """

    def __ne__(self, other):
        """Return ``self != other``."""
        return not self.__eq__(other)

    @property
    def space(self):
        """Space to which this vector belongs."""
        return self.__space

    @property
    def ndim(self):
        """Number of dimensions of this vector's space, always 1."""
        return 1

    @property
    def dtype(self):
        """Data type of this vector's space."""
        return self.space.dtype

    @property
    def size(self):
        """Length of this vector, equal to space size."""
        return self.space.size

    def __len__(self):
        """Return ``len(self)``.

        Equal to the number of space dimensions.
        """
        return self.space.size

    @property
    def shape(self):
        """Number of entries per axis, equals ``(size,)``."""
        return self.space.shape

    @property
    def itemsize(self):
        """Size in bytes of one element of this vector."""
        return self.dtype.itemsize

    @property
    def nbytes(self):
        """Number of bytes this vector uses in memory."""
        return self.size * self.itemsize

    def __array__(self, dtype=None):
        """Return a Numpy array containing this vector's data.

        Parameters
        ----------
        dtype :
            Specifier for the data type of the output array.

        Returns
        -------
        array : `numpy.ndarray`
        """
        if dtype is None:
            return self.asarray()
        else:
            return self.asarray().astype(dtype, copy=False)

    def __array_wrap__(self, obj):
        """Return a new vector from the data in ``obj``.

        Parameters
        ----------
        obj : `numpy.ndarray`
            Array that should be wrapped.

        Returns
        -------
        vector : `NtuplesBaseVector`
            Numpy array wrapped back into this vector's element type.
        """
        if obj.ndim == 0:
            return self.space.field.element(obj)
        else:
            return self.space.element(obj)

    def __int__(self):
        """Return ``int(self)``.

        Returns
        -------
        int : int
            Integer representing this vector.

        Raises
        ------
        TypeError : If the vector is of `size` != 1.
        """
        if self.size != 1:
            raise TypeError('only size 1 vectors can be converted to int')
        return int(self[0])

    def __long__(self):
        """Return ``long(self)``.

        The `long` method is only available in Python 2.

        Returns
        -------
        long : `long`
            Integer representing this vector.

        Raises
        ------
        TypeError : If the vector is of `size` != 1.
        """
        if self.size != 1:
            raise TypeError('only size 1 vectors can be converted to long')
        return long(self[0])

    def __float__(self):
        """Return ``float(self)``.

        Returns
        -------
        float : float
            Floating point number representing this vector.

        Raises
        ------
        TypeError : If the vector is of `size` != 1.
        """
        if self.size != 1:
            raise TypeError('only size 1 vectors can be converted to float')
        return float(self[0])

    def __complex__(self):
        """Return ``complex(self)``.

        Returns
        -------
        complex : `complex`
            Complex floating point number representing this vector.

        Raises
        ------
        TypeError : If the vector is of `size` != 1.
        """
        if self.size != 1:
            raise TypeError('only size 1 vectors can be converted to complex')
        return complex(self[0])

    def __str__(self):
        """Return ``str(self)``."""
        return array1d_str(self)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{!r}.element({})'.format(self.space,
                                         array1d_repr(self))

    @property
    def ufunc(self):
        """Internal class for access to Numpy style universal functions.

        These default ufuncs are always available, but may or may not be
        optimized for the specific space in use.
        """
        return NtuplesBaseUFuncs(self)

    def show(self, title=None, method='scatter', show=False, fig=None,
             **kwargs):
        """Display this vector graphically.

        Parameters
        ----------
        title : string, optional
            Set the title of the figure

        method : string, optional
            The following plotting methods are available:

            'scatter' : point plot

            'plot' : graph plot

        show : bool, optional
            If ``True``, the plot is shown immediately. Otherwise, display is
            deferred to a later point in time.
        fig : `matplotlib.figure.Figure`, optional
            Figure to draw into. Expected to be of same "style" as
            the figure given by this function. The most common use case
            is that ``fig`` is the return value of an earlier call to
            this function.
        kwargs : {'figsize', 'saveto', ...}
            Extra keyword arguments passed on to the display method.
            See the Matplotlib functions for documentation of extra
            options.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Resulting figure. If ``fig`` was given, the returned object
            is a reference to it.

        See Also
        --------
        odl.util.graphics.show_discrete_data : Underlying implementation
        """
        from odl.util.graphics import show_discrete_data
        from odl.discr import RegularGrid
        grid = RegularGrid(0, self.size - 1, self.size)
        return show_discrete_data(self.asarray(), grid, title=title,
                                  method=method, show=show, fig=fig, **kwargs)

    @property
    def impl(self):
        """Implementation of this vector's space."""
        return self.space.impl


class FnBase(NtuplesBase, LinearSpace):

    """Base class for n-tuples over a field independent of implementation."""

    def __init__(self, size, dtype):
        """Initialize a new instance.

        Parameters
        ----------
        size : non-negative int
            Number of entries in a tuple.
        dtype :
            Data type for each tuple entry. Can be provided in any
            way the `numpy.dtype` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.
            Only scalar data types (numbers) are allowed.
        """
        NtuplesBase.__init__(self, size, dtype)

        if not is_scalar_dtype(self.dtype):
            raise TypeError('{!r} is not a scalar data type'.format(dtype))

        if is_real_dtype(self.dtype):
            field = RealNumbers()
            self.__is_real = True
            self.__real_dtype = self.dtype
            self.__real_space = self
            try:
                self.__complex_dtype = complex_dtype(self.dtype)
            except ValueError:
                self.__complex_dtype = None
            self.__complex_space = None  # Set in first call of astype
        else:
            field = ComplexNumbers()
            self.__is_real = False
            try:
                self.__real_dtype = real_dtype(self.dtype)
            except ValueError:
                self.__real_dtype = None
            self.__real_space = None  # Set in first call of astype
            self.__complex_dtype = self.dtype
            self.__complex_space = self

        self.__is_floating = is_floating_dtype(self.dtype)
        LinearSpace.__init__(self, field)

    @property
    def is_rn(self):
        """``True`` if the space represents R^n, i.e. real tuples."""
        return self.__is_real and self.__is_floating

    @property
    def is_cn(self):
        """``True`` if the space represents C^n, i.e. complex tuples."""
        return (not self.__is_real) and self.__is_floating

    @property
    def real_dtype(self):
        """The real dtype corresponding to this space's `dtype`."""
        return self.__real_dtype

    @property
    def complex_dtype(self):
        """The complex dtype corresponding to this space's `dtype`."""
        return self.__complex_dtype

    @property
    def real_space(self):
        """The space corresponding to this space's `real_dtype`."""
        return self.astype(self.real_dtype)

    @property
    def complex_space(self):
        """The space corresponding to this space's `complex_dtype`."""
        return self.astype(self.complex_dtype)

    def _astype(self, dtype):
        """Internal helper for ``astype``. Can be overridden by subclasses."""
        return type(self)(self.size, dtype=dtype, weight=self.weighting)

    def astype(self, dtype):
        """Return a copy of this space with new ``dtype``.

        Parameters
        ----------
        dtype :
            Data type of the returned space. Can be given in any way
            `numpy.dtype` understands, e.g. as string ('complex64')
            or data type (complex).

        Returns
        -------
        newspace : `FnBase`
            The version of this space with given data type.
        """
        if dtype is None:
            # Need to filter this out since Numpy iterprets it as 'float'
            raise ValueError('unknown data type `None`')

        dtype = np.dtype(dtype)
        if dtype == self.dtype:
            return self

        # Caching for real and complex versions (exact dtype mappings)
        if dtype == self.real_dtype:
            if self.__real_space is None:
                self.__real_space = self._astype(dtype)
            return self.__real_space
        elif dtype == self.complex_dtype:
            if self.__complex_space is None:
                self.__complex_space = self._astype(dtype)
            return self.__complex_space
        else:
            return self._astype(dtype)

    @property
    def examples(self):
        """Example random vectors."""
        # Always return the same numbers
        rand_state = np.random.get_state()
        np.random.seed(1337)

        yield ('Linspaced', self.element(np.linspace(0, 1, self.size)))

        if self.is_rn:
            yield ('Random noise', self.element(np.random.rand(self.size)))
        elif self.is_cn:
            rnd = np.random.rand(self.size) + np.random.rand(self.size) * 1j
            yield ('Random noise', self.element(rnd))

        yield ('Normally distributed random noise',
               self.element(np.random.randn(self.size)))

        np.random.set_state(rand_state)

    @abstractmethod
    def zero(self):
        """Return a vector of zeros."""

    @abstractmethod
    def one(self):
        """Return a vector of ones."""

    @abstractmethod
    def _multiply(self, x1, x2, out):
        """Implement ``out[:] = x1 * x2`` (entry-wise)."""

    @abstractmethod
    def _divide(self, x1, x2, out):
        """Implement ``out[:] = x1 / x2`` (entry-wise)."""

    @staticmethod
    def default_dtype(field=None):
        """Return the default data type for a given field.

        Parameters
        ----------
        field : `Field`, optional
            Set of numbers to be represented by a data type.

        Returns
        -------
        dtype :
            Numpy data type specifier. The returned defaults are:
        """
        raise NotImplementedError('abstract method')


class FnBaseVector(NtuplesBaseVector, LinearSpaceElement):

    """Abstract class for `NtuplesBase` elements.

    Do not use this class directly -- to create an element of a vector
    space, call the space's `LinearSpace.element` method instead.
    """

    __eq__ = LinearSpaceElement.__eq__
    copy = LinearSpaceElement.copy


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
