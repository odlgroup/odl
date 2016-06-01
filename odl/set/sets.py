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

"""Basic abstract and concrete sets."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from builtins import int, object, str, zip
from future import standard_library
from past.builtins import basestring
standard_library.install_aliases()

from abc import ABCMeta, abstractmethod
from numbers import Integral, Real, Complex
import numpy as np

from odl.util.utility import (
    is_int_dtype, is_real_dtype, is_scalar_dtype, with_metaclass)


__all__ = ('Set', 'EmptySet', 'UniversalSet', 'Field', 'Integers',
           'RealNumbers', 'ComplexNumbers', 'Strings', 'CartesianProduct')


class Set(with_metaclass(ABCMeta, object)):

    """An abstract set.

    **Abstract Methods**

    Each subclass of `Set` must implement two methods: one to
    check if an object is contained in the set and one to test if two
    sets are equal.

    **Membership test:** ``__contains__(self, other)``

    Test if ``other`` is a member of this set. This function provides
    the operator overload for `in`.

    **Parameters:**
        other :
            The object to be tested for membership

    **Returns:**
        contains : bool
            True if ``other`` is a member of this set, False
            otherwise.


    **Equality test:** ``__eq__(self, other)``

    Test if ``other`` is the same set as this set, i.e. both sets are
    of the same type and contain the same elements. This function
    provides the operator overload for ``==``.

    **Parameters:**
        other :
            The object to be tested for equality.

    **Returns:**
        equals : bool
            True if both sets are of the same type and contain the
            same elements, False otherwise.

    A default implementation of the operator overload for ``!=`` via
    ``__ne__(self, other)`` is provided as ``not self.__eq__(other)``.

    **Element creation (optional)**: ``element(self, inp=None)``

    Create an element of this set, either from scratch or from an
    input parameter.

    **Parameters:**
        inp : optional
            The object from which to create the new element

    **Returns:**
        element : member of this set
            If ``inp`` is None, return an arbitrary element.
            Otherwise, return the element created from ``inp``.
    """

    @abstractmethod
    def __contains__(self, other):
        """Return ``other in self``."""

    def contains_set(self, other):
        """Test if ``other`` is a subset of this set.

        Implementing this method is optional. Default it tests for equality.
        """
        return self == other

    def contains_all(self, other):
        """Test if all points in ``other`` are contained in this set.

        This is a default implementation and should be overridden by
        subclasses.
        """
        return all(x in self for x in other)

    @abstractmethod
    def __eq__(self, other):
        """Return ``self == other``."""

    def __ne__(self, other):
        """Return ``self != other``."""
        return not self.__eq__(other)

    def element(self, inp=None):
        """Return an element from ``inp`` or from scratch.

        Implementing this method is optional.
        """
        raise NotImplementedError('`element` method not implemented')

    @property
    def examples(self):
        """Return a `generator` with elements in the set as name-value pairs.

        Can return a finite set of examples or an infinite set.

        Optional to implement, intended to be used for diagnostics.
        By default, the generator yields ``('element()', self.element())``.
        """
        yield ('element()', self.element())


class EmptySet(Set):

    """Set with no member elements (except `None`).

    None is considered as "no element", i.e.
    ``None in EmptySet()`` is True
    """

    def __contains__(self, other):
        """Test if ``other`` is None."""
        return other is None

    def contains_set(self, other):
        """Return True for the empty set, False otherwise."""
        return isinstance(other, EmptySet)

    def __eq__(self, other):
        """Return ``self == other``."""
        return isinstance(other, EmptySet)

    def element(self, inp=None):
        """Return None."""
        return None

    def __str__(self):
        """Return ``str(self)``."""
        return "EmptySet"

    def __repr__(self):
        """Return ``repr(self)``."""
        return "EmptySet()"


class UniversalSet(Set):

    """Set of all objects.

    Forget about set theory for a moment :-).
    """

    def __contains__(self, other):
        """Return True."""
        return True

    def contains_set(self, other):
        """Return True for any set."""
        return isinstance(other, Set)

    def __eq__(self, other):
        """Return ``self == other``."""
        return isinstance(other, UniversalSet)

    def element(self, inp=None):
        """Return ``inp`` in any case."""
        return inp

    def __str__(self):
        """Return ``str(self)``."""
        return "UniversalSet"

    def __repr__(self):
        """Return ``repr(self)``."""
        return "UniversalSet()"


class Strings(Set):

    """Set of fixed-length (unicode) strings."""

    def __init__(self, length):
        """Initialize a new instance.

        Parameters
        ----------
        length : int
            The fixed length of the strings in this set. Must be
            positive.
        """
        length_ = int(length)
        if length_ <= 0:
            raise ValueError('`length` must be positive, got {}'
                             ''.format(length))
        self.__length = length_

    @property
    def length(self):
        """Length of the strings."""
        return self.__length

    def __contains__(self, other):
        """Return ``other in self``.

        True if ``other`` is a string of at max `length`
        characters, False otherwise."""
        return isinstance(other, basestring) and len(other) == self.length

    def contains_all(self, array):
        """Test if `array` is an array of strings with correct length."""
        dtype = getattr(array, 'dtype', None)
        if dtype is None:
            dtype = np.result_type(*array)
        dtype_str = np.dtype('S{}'.format(self.length))
        dtype_uni = np.dtype('<U{}'.format(self.length))
        return dtype in (dtype_str, dtype_uni)

    def __eq__(self, other):
        """Return ``self == other``."""
        return isinstance(other, Strings) and other.length == self.length

    def element(self, inp=None):
        """Return a string from ``inp`` or from scratch."""
        if inp is not None:
            s = str(inp)[:self.length]
            s += ' ' * (self.length - len(s))
            return s
        else:
            return ' ' * self.length

    @property
    def examples(self):
        """Return example strings 'hello', 'world'."""
        return [('hello', 'hello'), ('world', 'world')]

    def __str__(self):
        """Return ``str(self)``."""
        return 'Strings({})'.format(self.length)

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'Strings({})'.format(self.length)


class Field(Set):
    """Any set that satisfies the field axioms

    For example `RealNumbers`, `ComplexNumbers` or
    the finite field :math:`F_2`.
    """

    @property
    def field(self):
        """Field of scalars for a field is itself.

        Notes
        -----
        This is a hack for this to work with duck-typing
        with `LinearSpace`'s.
        """
        return self


class ComplexNumbers(Field):

    """Set of complex numbers."""

    def __contains__(self, other):
        """Test if ``other`` is a complex number."""
        return isinstance(other, Complex)

    def contains_set(self, other):
        """Test if ``other`` is a subset of the complex numbers

        Returns
        -------
        contained : bool
            True if  other is `ComplexNumbers`, `RealNumbers` or `Integers`,
            else False.

        Examples
        --------
        >>> C = ComplexNumbers()
        >>> C.contains_set(RealNumbers())
        True
        """
        if other is self:
            return True

        return (isinstance(other, ComplexNumbers) or
                isinstance(other, RealNumbers) or
                isinstance(other, Integers))

    def contains_all(self, array):
        """Test if `array` is an array of real or complex numbers."""
        dtype = getattr(array, 'dtype', None)
        if dtype is None:
            dtype = np.result_type(*array)
        return is_scalar_dtype(dtype)

    def __eq__(self, other):
        """Return ``self == other``."""
        if other is self:
            return True

        return isinstance(other, ComplexNumbers)

    def element(self, inp=None):
        """Return a complex number from ``inp`` or from scratch."""
        if inp is not None:
            return complex(inp)
        else:
            return complex(0.0, 0.0)

    @property
    def examples(self):
        """Return examples of complex numbers."""
        numbers = [-1.0, 0.5, 0.0 + 2.0j, 0.0, 0.01, 1.0 + 1.0j, 1.0j, 1.0]
        return [(str(x), x) for x in numbers]

    def __str__(self):
        """Return ``str(self)``."""
        return "ComplexNumbers"

    def __repr__(self):
        """Return ``repr(self)``."""
        return "ComplexNumbers()"


class RealNumbers(Field):

    """Set of real numbers."""

    def __contains__(self, other):
        """Test if ``other`` is a real number."""
        return isinstance(other, Real)

    def contains_set(self, other):
        """Test if ``other`` is a subset of the real numbers

        Returns
        -------
        contained : bool
            True if other is `RealNumbers` or
            `Integers` False else.

        Examples
        --------
        >>> R = RealNumbers()
        >>> R.contains_set(RealNumbers())
        True
        """
        if other is self:
            return True

        return (isinstance(other, RealNumbers) or
                isinstance(other, Integers))

    def contains_all(self, array):
        """Test if `array` is an array of real numbers."""
        dtype = getattr(array, 'dtype', None)
        if dtype is None:
            dtype = np.result_type(*array)
        return is_real_dtype(dtype)

    def __eq__(self, other):
        """Return ``self == other``."""
        if other is self:
            return True

        return isinstance(other, RealNumbers)

    def element(self, inp=None):
        """Return a real number from ``inp`` or from scratch."""
        if inp is not None:
            return float(inp)
        else:
            return 0.0

    @property
    def examples(self):
        """Return examples of real numbers."""
        numbers = [-1.0, 0.5, 0.0, 0.01, 1.0]
        return [(str(x), x) for x in numbers]

    def __str__(self):
        """Return ``str(self)``."""
        return "RealNumbers"

    def __repr__(self):
        """Return ``repr(self)``."""
        return "RealNumbers()"


class Integers(Set):

    """Set of integers."""

    def __eq__(self, other):
        """Return ``self == other``."""
        if other is self:
            return True

        return isinstance(other, Integers)

    def __contains__(self, other):
        """Test if ``other`` is an integer."""
        return isinstance(other, Integral)

    def contains_set(self, other):
        """Test if ``other`` is a subset of the real numbers

        Returns
        -------
        contained : bool
            True if  other is `Integers`, else False.

        Examples
        --------
        >>> Z = Integers()
        >>> Z.contains_set(RealNumbers())
        False
        """
        if other is self:
            return True

        return isinstance(other, Integers)

    def contains_all(self, array):
        """Test if `array` is an array of integers."""
        dtype = getattr(array, 'dtype', None)
        if dtype is None:
            dtype = np.result_type(*array)
        return is_int_dtype(dtype)

    def element(self, inp=None):
        """Return an integer from ``inp`` or from scratch."""
        if inp is not None:
            return int(inp)
        else:
            return 0

    @property
    def examples(self):
        """Return examples of integers."""
        numbers = [-1, 0, 1]
        return [(str(x), x) for x in numbers]

    def __str__(self):
        """Return ``str(self)``."""
        return "Integers"

    def __repr__(self):
        """Return ``repr(self)``."""
        return "Integers()"


class CartesianProduct(Set):

    """Cartesian product of ``n`` sets.

    The elements of this set are ``n``-tuples where the i-th entry
    is an element of the i-th set.
    """

    def __init__(self, *sets):
        """Initialize a new instance."""
        if not all(isinstance(set_, Set) for set_ in sets):
            wrong = [set_ for set_ in sets
                     if not isinstance(set_, Set)]
            raise TypeError('{!r} not Set instance(s)'.format(wrong))

        self.__sets = tuple(sets)

    @property
    def sets(self):
        """Factors (sets) as a tuple."""
        return self.__sets

    def __contains__(self, other):
        """Test if ``other`` is contained in this set.

        Returns
        -------
        contains : bool
            True if ``other`` has the same length as this Cartesian
            product and each entry is contained in the set with
            corresponding index, False otherwise.
        """
        try:
            iter(other)
        except TypeError:
            return False
        return (len(other) == len(self) and
                all(p in set_ for set_, p in zip(self.sets, other)))

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            True if ``other`` is a `CartesianProduct` instance,
            has the same length as this Cartesian product and all sets
            with the same index are equal, False otherwise.
        """
        return (isinstance(other, CartesianProduct) and
                len(other) == len(self) and
                all(so == ss for so, ss in zip(other.sets, self.sets)))

    def element(self, inp=None):
        """Create a `CartesianProduct` element.

        Parameters
        ----------
        inp : `iterable`, optional
            Collection of input values for the
            `LinearSpace.element` methods
            of all sets in the Cartesian product.

        Returns
        -------
        element : tuple
            A tuple of the given input
        """
        if inp is None:
            tpl = tuple(set_.element() for set_ in self.sets)
        else:
            tpl = tuple(set_.element(inpt)
                        for inpt, set_ in zip(inp, self.sets))

            if len(tpl) != len(self):
                raise ValueError('input provides only {} values, needed '
                                 'are {}'.format(len(tpl), len(self)))

        return tpl

    def __len__(self):
        """Return ``len(self)``."""
        return len(self.sets)

    def __getitem__(self, indcs):
        """Return ``self[indcs]``.

        Examples
        --------
        >>> emp, univ = EmptySet(), UniversalSet()
        >>> prod = CartesianProduct(emp, univ, univ, emp, emp)
        >>> prod[2]
        UniversalSet()
        >>> prod[2:4]
        CartesianProduct(UniversalSet(), EmptySet())
        """
        if isinstance(indcs, slice):
            return CartesianProduct(*self.sets[indcs])
        else:
            return self.sets[indcs]

    def __str__(self):
        """Return ``str(self)``."""
        return ' x '.join(str(set_) for set_ in self.sets)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> emp, univ = EmptySet(), UniversalSet()
        >>> CartesianProduct(emp, univ)
        CartesianProduct(EmptySet(), UniversalSet())
        """
        sets_str = ', '.join(repr(set_) for set_ in self.sets)
        return 'CartesianProduct({})'.format(sets_str)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
