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

"""Basic abstract and concrete sets."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from builtins import int, object, str, zip
from odl.util.utility import with_metaclass
from future import standard_library
standard_library.install_aliases()

# External
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real, Complex

# ODL


__all__ = ('Set', 'EmptySet', 'UniversalSet', 'Integers', 'RealNumbers',
           'ComplexNumbers', 'Strings', 'CartesianProduct')


class Set(with_metaclass(ABCMeta, object)):

    """An abstract set.

    **Abstract Methods**
    Each subclass of :class:`Set` must implement two methods: one to check if
    an object is contained in the set and one to test if two sets are
    equal.

    ``__contains__(self, other)``
    
    Test if ``other`` is a member of this set. This function provides the
    operator overload for `in`.

    **Parameters:**
        other : `object`
            The object to be tested for membership.

    **Returns:**
        contains : `bool`
            `True` if ``other`` is a member of this set, `False`
            otherwise.


    ``__eq__(self, other)``
    
    Test if ``other`` is the same set as this set, i.e. both sets are
    of the same type and contain the same elements. This function
    provides the operator overload for ``==``.

    **Parameters:**
        other : `object`
            The object to be tested for equality.

    **Returns:**
        equals : `bool`
            `True` if both sets are of the same type and contain the
            same elements, `False` otherwise.

    A default implementation of the operator overload for ``!=`` via
    ``__ne__(self, other)`` is provided as ``not self.__eq__(other)``.

    optional: ``element(inp=None)``
    
    Create an element of this set, either from scratch or from an
    input parameter.

    **Parameters:**
        inp : `object`, optional
            The object from which to create the new element.

    **Returns:**
        element : member of this set
            If ``inp == None``, return an arbitrary element.
            Otherwise, return the element created from ``inp``.
    """

    @abstractmethod
    def __contains__(self, other):
        """``s.__contains__(other) <==> other in s``."""

    def contains_set(self, other):
        """Test if ``other`` is a subset of this set.

        Implementing this method is optional.
        """
        raise NotImplementedError("'contains_set' method not implemented.")

    @abstractmethod
    def __eq__(self, other):
        """``s.__eq__(other) <==> s == other``."""

    def __ne__(self, other):
        """``s.__ne__(other) <==> s != other``."""
        return not self.__eq__(other)

    def element(self, inp=None):
        """Return an element from ``inp`` or from scratch.

        Implementing this method is optional.
        """
        raise NotImplementedError("'element' method not implemented.")


class EmptySet(Set):

    """The empty set.

    `None` is considered as "no element", i.e.
    ``None in EmptySet() is True``
    """

    def __contains__(self, other):
        """Test if ``other`` is `None`."""
        return other is None

    def contains_set(self, other):
        """Return `True` for the empty set, otherwise `False`."""
        return isinstance(other, EmptySet)

    def __eq__(self, other):
        """``s.__eq__(other) <==> s == other``."""
        return isinstance(other, EmptySet)

    def element(self, inp=None):
        """Return `None`."""
        return None

    def __str__(self):
        """``s.__str__() <==> str(s)``."""
        return "EmptySet"

    def __repr__(self):
        """``s.__repr__() <==> repr(s)``."""
        return "EmptySet()"


class UniversalSet(Set):

    """The set of all objects.

    Forget about set theory for a moment :-).
    """

    def __contains__(self, other):
        """Return `True`."""
        return True

    def contains_set(self, other):
        """Return `True` for any set."""
        return isinstance(other, Set)

    def __eq__(self, other):
        """``s.__eq__(other) <==> s == other``."""
        return isinstance(other, UniversalSet)

    def element(self, inp=None):
        """Return ``inp`` in any case."""
        return inp

    def __str__(self):
        """``s.__str__() <==> str(s)``."""
        return "UniversalSet"

    def __repr__(self):
        """``s.__repr__() <==> repr(s)``."""
        return "UniversalSet()"


class Strings(Set):

    """The set of fixed-length (unicode) strings."""

    def __init__(self, length):
        """Initialize a new instance.

        Parameters
        ----------
        length : `int`
            The fixed length of the strings in this set. Must be
            positive.
        """
        if length not in Integers():
            raise TypeError('`length` {} is not an integer.'.format(length))
        if length <= 0:
            raise ValueError('`length` {} is not positive.'.format(length))
        self._length = length

    @property
    def length(self):
        """The length attribute."""
        return self._length

    def __contains__(self, other):
        """Test if ``other`` is a string of at max :attr:`length` characters."""
        return isinstance(other, str) and len(other) <= self.length

    def __eq__(self, other):
        """``s.__eq__(other) <==> s == other``."""
        return isinstance(other, Strings) and other.length == self.length

    def element(self, inp=None):
        """Return a string from ``inp`` or from scratch."""
        if inp is not None:
            return str(inp)[:self.length]
        else:
            return ''

    def __str__(self):
        """``s.__str__() <==> str(s)``."""
        return 'Strings({})'.format(self.length)

    def __repr__(self):
        """``s.__repr__() <==> repr(s)``."""
        return 'Strings({})'.format(self.length)

    
class Field(with_metaclass(ABCMeta, Set)):
    """Any set that satisfies the field axioms

    For example :class:`RealNumbers`, :class::class:`ComplexNumbers` or 
    the finite field F2
    """

    @property
    def field(self):
        """ The field of scalars for a field is itself
        """
        return self


class ComplexNumbers(Field):

    """The set of `complex` numbers."""

    def __contains__(self, other):
        """Test if ``other`` is a `complex` number."""
        return isinstance(other, Complex)

    def contains_set(self, other):
        """Test if ``other`` is a subset of the `complex` numbers

        Returns
        -------
        contained : `bool`
            True if  other is a :class::class:`ComplexNumbers`, :class:`RealNumbers`
            or :class:`Integers`, false else.

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

    def __eq__(self, other):
        """``s.__eq__(other) <==> s == other``."""
        if other is self:
            return True
            
        return isinstance(other, ComplexNumbers)

    def element(self, inp=None):
        """Return a `complex` number from ``inp`` or from scratch."""
        if inp is not None:
            return complex(inp)
        else:
            return complex(0.0, 0.0)

    def __str__(self):
        """``s.__str__() <==> str(s)``."""
        return "ComplexNumbers"

    def __repr__(self):
        """``s.__repr__() <==> repr(s)``."""
        return "ComplexNumbers()"

    
class RealNumbers(Field):
    """The set of real numbers."""

    def __contains__(self, other):
        """Test if ``other`` is a real number."""
        return isinstance(other, Real)

    def contains_set(self, other):
        """Test if ``other`` is a subset of the real numbers

        Returns
        -------
        contained : `bool`
            `True` if other is a :class:`RealNumbers` or :class:`Integers`
            `False` else.

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

    def __eq__(self, other):
        """``s.__eq__(other) <==> s == other``."""
        if other is self:
            return True
            
        return isinstance(other, RealNumbers)

    def element(self, inp=None):
        """Return a real number from ``inp`` or from scratch."""
        if inp is not None:
            return float(inp)
        else:
            return 0.0

    def __str__(self):
        """``s.__str__() <==> str(s)``."""
        return "RealNumbers"

    def __repr__(self):
        """``s.__repr__() <==> repr(s)``."""
        return "RealNumbers()"


class Integers(Set):

    """The set of integers."""

    def __eq__(self, other):
        """``s.__eq__(other) <==> s == other``."""        
        
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
        contained : `bool`
            `True` if  other is :class:`Integers`, `False` otherwise.

        Examples
        --------
        >>> Z = Integers()
        >>> Z.contains_set(RealNumbers())
        False
        """
        if other is self:
            return True
            
        return isinstance(other, Integers)

    def element(self, inp=None):
        """Return an integer from ``inp`` or from scratch."""
        if inp is not None:
            return int(inp)
        else:
            return 0

    def __str__(self):
        """``s.__str__() <==> str(s)``."""
        return "Integers"

    def __repr__(self):
        """``s.__repr__() <==> repr(s)``."""
        return "Integers()"


class CartesianProduct(Set):

    """The Cartesian product of ``n`` sets.

    The elements of this set are ``n``-tuples where the i-th entry
    is an element of the i-th set.
    """

    def __init__(self, *sets):
        """Initialize a new instance."""
        if not all(isinstance(set_, Set) for set_ in sets):
            wrong = [set_ for set_ in sets
                     if not isinstance(set_, Set)]
            raise TypeError('{} not Set instance(s)'.format(wrong))

        self._sets = tuple(sets)

    @property
    def sets(self):
        """The factors (sets) as a `tuple`."""
        return self._sets

    def __contains__(self, other):
        """Test if ``other`` is contained in this set.

        Returns
        -------
        contains : `bool`
            `True` if ``other`` has the same length as this Cartesian
            product and each entry is contained in the set with
            corresponding index, `False` otherwise.
        """
        try:
            iter(other)
        except TypeError:
            return False
        return (len(other) == len(self) and
                all(p in set_ for set_, p in zip(self.sets, other)))

    def __eq__(self, other):
        """``s.__eq__(other) <==> s == other``.

        Returns
        -------
        equals : `bool`
            `True` if ``other`` is a :class:`CartesianProduct` instance, has
            the same length as this Cartesian product and all sets
            with the same index are equal, `False` otherwise.
        """
        return (isinstance(other, CartesianProduct) and
                len(other) == len(self) and
                all(so == ss for so, ss in zip(other.sets, self.sets)))

    def element(self, inp=None):
        """Create a :class:`CartesianProduct` element.

        Parameters
        ----------
        inp : `iterable`, optional
            Collection of input values for the 
            :meth:`~odl.set.space.LinearSpace.element()` methods
            of all sets in the Cartesian product.
        """
        if inp is None:
            tpl = tuple(set_.element() for set_ in self.sets)
        else:
            tpl = tuple(set_.element(inpt)
                        for inpt, set_ in zip(inp, self.sets))

            if len(tpl) != len(self):
                raise ValueError('input provides only {} values, needed '
                                 'are {}.'.format(len(tpl), len(self)))

        return tpl

    def __len__(self):
        """``s.__len__() <==> len(s)``."""
        return len(self.sets)

    def __getitem__(self, indcs):
        """``s.__getitem__(indcs) <==> s[indcs]``.

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
        """``s.__str__() <==> str(s)``."""
        return ' x '.join(str(set_) for set_ in self.sets)

    def __repr__(self):
        """``s.__repr__() <==> repr(s)``.

        Examples
        --------
        >>> emp, univ = EmptySet(), UniversalSet()
        >>> CartesianProduct(emp, univ)
        CartesianProduct(EmptySet(), UniversalSet())
        """
        sets_str = ', '.join(repr(set_) for set_ in self.sets)
        return 'CartesianProduct({})'.format(sets_str)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
