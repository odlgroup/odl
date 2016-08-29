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
    the operator overload for ``in``.

    **Parameters:**
        other :
            Object to be tested for membership

    **Returns:**
        contains : bool
            ``True`` if ``other`` is a member of this set, ``False``
            otherwise.


    **Equality test:** ``__eq__(self, other)``

    Test if ``other`` is the same set as this set, i.e. both sets are
    of the same type and contain the same elements. This function
    provides the operator overload for ``==``.

    **Parameters:**
        other :
            Object to be tested for equality.

    **Returns:**
        equals : bool
            ``True`` if both sets are of the same type and contain the
            same elements, ``False`` otherwise.

    A default implementation of the operator overload for ``!=`` via
    ``__ne__(self, other)`` is provided as ``not self.__eq__(other)``.

    **Element creation (optional)**: ``element(self, inp=None)``

    Create an element of this set, either from scratch or from an
    input parameter.

    **Parameters:**
        inp : optional
            Object from which to create the new element

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

        This is a default implementation that simply tests for equality.
        It should be overridden by subclasses.

        Returns
        -------
        set_contained : bool
            ``True`` if ``other`` is contained in this set, ``False``
            otherwise.
        """
        return self == other

    def contains_all(self, other):
        """Test if all elements in ``other`` are contained in this set.

        This is a default implementation that assumes ``other`` to be
        a sequence and tests each elment of ``other`` sequentially.
        This method should be overridden by subclasses.

        Returns
        -------
        all_contained : bool
            ``True`` if all elements of ``other`` are contained in this
            set, ``False`` otherwise
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

        This method should be overridden by subclasses.
        """
        raise NotImplementedError('`element` method not implemented')

    @property
    def examples(self):
        """Generator creating name-value pairs of set elements.

        This method is mainly intended for diagnostics and yields elements,
        either a finite number of times or indefinitely.

        This default implementation returns
        ``('element()', self.element())`` and should be overridden by
        subclasses.
        """
        yield ('element()', self.element())

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}()'.format(self.__class__.__name__)

    def __str__(self):
        """Return ``str(self)``."""
        return '{}'.format(self.__class__.__name__)


class EmptySet(Set):

    """Set with no member elements (except ``None``).

    ``None`` is considered as "no element", i.e. ``None in EmptySet()``
    is the only test that evaluates to ``True``.
    """

    def __contains__(self, other):
        """Return ``other in self``, always ``False`` except for ``None``."""
        return other is None

    def contains_set(self, other):
        """Return ``True`` for the empty set, ``False`` otherwise."""
        return isinstance(other, EmptySet)

    def __eq__(self, other):
        """Return ``self == other``."""
        return isinstance(other, EmptySet)

    def element(self, inp=None):
        """Return None."""
        return None


class UniversalSet(Set):

    """Set of all objects.

    Forget about set theory for a moment :-).
    """

    def __contains__(self, other):
        """Return ``other in self``, always ``True``."""
        return True

    def contains_set(self, other):
        """Return ``True`` for any set."""
        return isinstance(other, Set)

    def __eq__(self, other):
        """Return ``self == other``."""
        return isinstance(other, UniversalSet)

    def element(self, inp=None):
        """Return ``inp`` in any case."""
        return inp


class Strings(Set):

    """Set of fixed-length (unicode) strings."""

    def __init__(self, length):
        """Initialize a new instance.

        Parameters
        ----------
        length : positive int
            Fixed length of the strings in this set.
        """
        length, length_in = int(length), length
        if length <= 0:
            raise ValueError('`length` must be positive, got {}'
                             ''.format(length_in))
        self.__length = length

    @property
    def length(self):
        """Fixed length of the strings in this set."""
        return self.__length

    def __contains__(self, other):
        """Return ``other in self``.

        Returns
        -------
        contained : bool
            ``True`` if ``other`` is a string of exactly `length`
            characters, ``False`` otherwise.
        """
        return isinstance(other, basestring) and len(other) == self.length

    def contains_all(self, other):
        """Return ``True`` if all strings in ``other`` have size `length`."""
        dtype = getattr(other, 'dtype', None)
        if dtype is None:
            dtype = np.result_type(*other)
        dtype_str = np.dtype('S{}'.format(self.length))
        dtype_uni = np.dtype('<U{}'.format(self.length))
        return dtype in (dtype_str, dtype_uni)

    def __eq__(self, other):
        """Return ``self == other``."""
        return isinstance(other, Strings) and other.length == self.length

    def element(self, inp=None):
        """Return an element from ``inp`` or from scratch."""
        if inp is not None:
            s = str(inp)[:self.length]
            s += ' ' * (self.length - len(s))
            return s
        else:
            return ' ' * self.length

    @property
    def examples(self):
        """Return example strings 'hello', 'world' (size adapted)."""
        hello_str = 'hello'[:self.length]
        hello_str += ' ' * (self.length - len(hello_str))
        world_str = 'world'[:self.length]
        world_str += ' ' * (self.length - len(world_str))
        return [('hello', hello_str), ('world', world_str)]

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'Strings({})'.format(self.length)


class Field(Set):

    """A set that satisfies the field axioms.

    Examples: `RealNumbers`, `ComplexNumbers` or
    the finite field :math:`F_2`.

    See `the Wikipedia entry on fields
    <https://en.wikipedia.org/wiki/Field_%28mathematics%29>`_ for
    further information.
    """

    @property
    def field(self):
        """Field of scalars for a field is itself.

        Notes
        -----
        This is a hack to make fields to work via duck-typing with
        `LinearSpace`'s.
        """
        return self


class ComplexNumbers(Field):

    """Set of complex numbers."""

    def __contains__(self, other):
        """Return ``other in self``."""
        return isinstance(other, Complex)

    def contains_set(self, other):
        """Return ``True`` if ``other`` is a subset of the complex numbers.

        Returns
        -------
        contained : bool
            ``True`` if  ``other`` is an instance of `ComplexNumbers`,
            `RealNumbers` or `Integers`, ``False`` otherwise.

        Examples
        --------
        >>> complex_numbers = ComplexNumbers()
        >>> complex_numbers.contains_set(RealNumbers())
        True
        """
        if other is self:
            return True

        return (isinstance(other, ComplexNumbers) or
                isinstance(other, RealNumbers) or
                isinstance(other, Integers))

    def contains_all(self, other):
        """Return ``True`` if ``other`` is a sequence of complex numbers."""
        dtype = getattr(other, 'dtype', None)
        if dtype is None:
            dtype = np.result_type(*other)
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


class RealNumbers(Field):

    """Set of real numbers."""

    def __contains__(self, other):
        """Return ``other in self``."""
        return isinstance(other, Real)

    def contains_set(self, other):
        """Return ``True`` if ``other`` is a subset of the real numbers.

        Returns
        -------
        contained : bool
            ``True`` if other is an instance of `RealNumbers` or
            `Integers` False otherwise.

        Examples
        --------
        >>> real_numbers = RealNumbers()
        >>> real_numbers.contains_set(RealNumbers())
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


class Integers(Set):

    """Set of integers."""

    def __eq__(self, other):
        """Return ``self == other``."""
        if other is self:
            return True

        return isinstance(other, Integers)

    def __contains__(self, other):
        """Return ``other in self``."""
        return isinstance(other, Integral)

    def contains_set(self, other):
        """Test if ``other`` is a subset of the integers.

        Returns
        -------
        contained : bool
            ``True`` if  other is an instance of `Integers`,
            ``False`` otherwise.

        Examples
        --------
        >>> integers = Integers()
        >>> integers.contains_set(RealNumbers())
        False
        """
        if other is self:
            return True

        return isinstance(other, Integers)

    def contains_all(self, other):
        """Return ``True`` if ``other`` is a sequence of integers."""
        dtype = getattr(other, 'dtype', None)
        if dtype is None:
            dtype = np.result_type(*other)
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


class CartesianProduct(Set):

    """Cartesian product of a finite number of sets.

    The elements of this set are tuples where the i-th entry
    is an element of the i-th set.
    """

    def __init__(self, *sets):
        """Initialize a new instance."""
        for set_ in sets:
            if not isinstance(set_, Set):
                raise TypeError('{!r} is not a Set instance.'.format(set_))

        self.__sets = tuple(sets)

    @property
    def sets(self):
        """Factors (sets) as a tuple."""
        return self.__sets

    def __contains__(self, other):
        """Return ``other in self``.

        Parameters
        ----------
        other :
            Object to be tested for membership

        Returns
        -------
        contains : bool
            ``True`` if ``other`` is a sequence with same length as this
            Cartesian product, and each entry is contained in the set with
            corresponding index, ``False`` otherwise.
        """
        try:
            len(other)
        except TypeError:
            return False
        return (len(other) == len(self) and
                all(p in set_ for set_, p in zip(self.sets, other)))

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if ``other`` is a `CartesianProduct` instance,
            has the same length as this Cartesian product and all sets
            with the same index are equal, ``False`` otherwise.
        """
        return (isinstance(other, CartesianProduct) and
                len(other) == len(self) and
                all(so == ss for so, ss in zip(other.sets, self.sets)))

    def element(self, inp=None):
        """Create a `CartesianProduct` element.

        Parameters
        ----------
        inp : iterable, optional
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

    def __getitem__(self, indices):
        """Return ``self[indices]``.

        Examples
        --------
        >>> emp, univ = EmptySet(), UniversalSet()
        >>> prod = CartesianProduct(emp, univ, univ, emp, emp)
        >>> prod[2]
        UniversalSet()
        >>> prod[2:4]
        CartesianProduct(UniversalSet(), EmptySet())
        """
        if isinstance(indices, slice):
            return CartesianProduct(*self.sets[indices])
        else:
            return self.sets[indices]

    def __str__(self):
        """Return ``str(self)``."""
        return ' x '.join(str(set_) for set_ in self.sets)

    def __repr__(self):
        """Return ``repr(self)``."""
        sets_str = ', '.join(repr(set_) for set_ in self.sets)
        return '{}({})'.format(self.__class__.__name__, sets_str)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
