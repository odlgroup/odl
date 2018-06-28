﻿# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Basic abstract and concrete sets."""

from __future__ import print_function, division, absolute_import
from builtins import int, object
from numbers import Integral, Real, Complex
from past.types.basestring import basestring
import numpy as np

from odl.util import is_int_dtype, is_real_dtype, is_numeric_dtype, unique


__all__ = ('Set', 'EmptySet', 'UniversalSet', 'Field', 'Integers',
           'RealNumbers', 'ComplexNumbers', 'Strings', 'CartesianProduct',
           'SetUnion', 'SetIntersection', 'FiniteSet')


class Set(object):

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

    def __contains__(self, other):
        """Return ``other in self``."""
        raise NotImplementedError('abstract method')

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

    def __eq__(self, other):
        """Return ``self == other``."""
        raise NotImplementedError('abstract method')

    def __ne__(self, other):
        """Return ``self != other``."""
        return not self.__eq__(other)

    def __cmp__(self, other):
        """Comparsion not implemented."""
        # Stops python 2 from allowing comparsion of arbitrary objects
        raise TypeError('unorderable types: {}, {}'
                        ''.format(self.__class__.__name__, type(other)))

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

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash(type(self))

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

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash(type(self))

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

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((type(self), self.length))

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
        return is_numeric_dtype(dtype)

    def __eq__(self, other):
        """Return ``self == other``."""
        if other is self:
            return True

        return isinstance(other, ComplexNumbers)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash(type(self))

    def element(self, inp=None):
        """Return a complex number from ``inp`` or from scratch."""
        if inp is not None:
            # Workaround for missing __complex__ of numpy.ndarray
            # for Numpy version < 1.12
            # TODO: remove when Numpy >= 1.12 is required
            if isinstance(inp, np.ndarray):
                return complex(inp.reshape([1])[0])
            else:
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

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash(type(self))

    def element(self, inp=None):
        """Return a real number from ``inp`` or from scratch."""
        if inp is None:
            return 0.0
        else:
            return float(getattr(inp, 'real', inp))

    @property
    def examples(self):
        """Return examples of real numbers."""
        numbers = [-1.0, 0.5, 0.0, 0.01, 1.0]
        return [(str(x), x) for x in numbers]


class Integers(Set):

    """Set of integers."""

    def __contains__(self, other):
        """Return ``other in self``."""
        return isinstance(other, Integral)

    def contains_set(self, other):
        """Test if ``other`` is a subset of the integers.

        Returns
        -------
        contained : bool
            ``True`` if other is an instance of `Integers`,
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

    def __eq__(self, other):
        """Return ``self == other``."""
        if other is self:
            return True

        return isinstance(other, Integers)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash(type(self))

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
        """The sets of this cartesian product as a tuple."""
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
        return (type(self) == type(other) and
                self.sets == other.sets)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((type(self), self.sets))

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
        >>> reals, complexnrs = odl.RealNumbers(), odl.ComplexNumbers()
        >>> prod = odl.CartesianProduct(reals, complexnrs, complexnrs, reals)
        >>> prod[1]
        ComplexNumbers()
        >>> prod[2:4]
        CartesianProduct(ComplexNumbers(), RealNumbers())
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


class SetUnion(Set):

    """The union of several subsets.

    The elements of this set are elements of at least one of the subsets.

    This is a *lazy* union, i.e. there is no intelligence and the set is
    literally stored as the union of its subsets.
    """

    def __init__(self, *sets):
        """Initialize a new instance.

        Parameters
        ----------
        set1, ..., setN : `Set`
            The sets whose union should be taken.
            Any duplicates are ignored.

        Examples
        --------
        >>> reals, complexnrs = odl.RealNumbers(), odl.ComplexNumbers()
        >>> union = odl.SetUnion(reals, complexnrs)
        """
        for set_ in sets:
            if not isinstance(set_, Set):
                raise TypeError('{!r} is not a Set instance.'.format(set_))

        self.__sets = tuple(unique(sets))

    @property
    def sets(self):
        """The sets of this union as a tuple."""
        return self.__sets

    def __contains__(self, other):
        """Return ``other in self``.

        Returns
        -------
        contains : bool
            ``True`` if ``other`` is a member of any subset,
            ``False`` otherwise.

        Examples
        --------
        >>> reals, complexnrs = odl.RealNumbers(), odl.ComplexNumbers()
        >>> union = odl.SetUnion(reals, complexnrs)
        >>> 2 + 1j in union
        True
        >>> [1, 2] in union
        False
        """
        return any(other in set for set in self.sets)

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if ``other`` is a `SetUnion` instance, and
            has the same subsets as this set, ``False`` otherwise.
        """
        return (type(self) == type(other) and
                all(set_ in other for set_ in self) and
                all(set_ in self for set_ in other))

    def __hash__(self):
        """Return ``hash(self)``."""
        # Use `set` to allow permutations
        return hash((type(self), set(self.sets)))

    def element(self, inp=None):
        """Create a new element.

        First tries calling the first set, then the second, etc.

        For more specific control, use ``set[i].element()`` to pick which
        subset to use.
        """
        for set in self.sets:
            try:
                return set.element(inp)
            except NotImplementedError:
                pass
        raise NotImplementedError('`element` not implemented for any of the '
                                  'subsets')

    def __len__(self):
        """Return ``len(self)``."""
        return len(self.sets)

    def __getitem__(self, indcs):
        """Return ``self[indcs]``.

        Examples
        --------
        >>> reals, complexnrs = odl.RealNumbers(), odl.ComplexNumbers()
        >>> union = odl.SetUnion(reals, complexnrs)
        >>> union[0]
        RealNumbers()
        >>> union[:]
        SetUnion(RealNumbers(), ComplexNumbers())
        """
        if isinstance(indcs, slice):
            return SetUnion(*self.sets[indcs])
        else:
            return self.sets[indcs]

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> reals, complexnrs = odl.RealNumbers(), odl.ComplexNumbers()
        >>> odl.SetUnion(reals, complexnrs)
        SetUnion(RealNumbers(), ComplexNumbers())
        """
        sets_str = ', '.join(repr(set_) for set_ in self.sets)
        return '{}({})'.format(self.__class__.__name__, sets_str)


class SetIntersection(Set):

    """The intersection of several subsets.

    The elements of this set are elements of all the subsets.

    This is a *lazy* intersection, i.e. there is no intelligence and the set is
    literally stored as the intersection of its subsets.
    """

    def __init__(self, *sets):
        """Initialize a new instance.

        Parameters
        ----------
        set1, ..., setN : `Set`
            The sets whose intersection should be taken.
            Any duplicates are ignored.

        Examples
        --------
        >>> reals, complexnrs = odl.RealNumbers(), odl.ComplexNumbers()
        >>> union = odl.SetIntersection(reals, complexnrs)
        """
        for set_ in sets:
            if not isinstance(set_, Set):
                raise TypeError('{!r} is not a Set instance.'.format(set_))

        self.__sets = tuple(unique(sets))

    @property
    def sets(self):
        """The sets of this intersection as a tuple."""
        return self.__sets

    def __contains__(self, other):
        """Return ``other in self``.

        Returns
        -------
        contains : bool
            ``True`` if ``other`` is a member of all subsets,
            ``False`` otherwise.

        Examples
        --------
        >>> reals, complexnrs = odl.RealNumbers(), odl.ComplexNumbers()
        >>> intersection = odl.SetIntersection(reals, complexnrs)
        >>> 1.0 in intersection
        True
        >>> 1.0j in intersection
        False
        """
        return all(other in set for set in self.sets)

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if ``other`` is a `SetUnion` instance, and
            has the same subsets as this set, ``False`` otherwise.
        """
        return (type(self) == type(other) and
                all(set_ in other for set_ in self) and
                all(set_ in self for set_ in other))

    def __hash__(self):
        """Return ``hash(self)``."""
        # Use `set` to allow permutations
        return hash((type(self), set(self.sets)))

    def __len__(self):
        """Return ``len(self)``."""
        return len(self.sets)

    def __getitem__(self, indcs):
        """Return ``self[indcs]``.

        Examples
        --------
        >>> reals, complexnrs = odl.RealNumbers(), odl.ComplexNumbers()
        >>> intersection = odl.SetIntersection(reals, complexnrs)
        >>> intersection[0]
        RealNumbers()
        >>> intersection[:]
        SetIntersection(RealNumbers(), ComplexNumbers())
        """
        if isinstance(indcs, slice):
            return SetIntersection(*self.sets[indcs])
        else:
            return self.sets[indcs]

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> reals, complexnrs = odl.RealNumbers(), odl.ComplexNumbers()
        >>> odl.SetIntersection(reals, complexnrs)
        SetIntersection(RealNumbers(), ComplexNumbers())
        """
        sets_str = ', '.join(repr(set_) for set_ in self.sets)
        return '{}({})'.format(self.__class__.__name__, sets_str)


class FiniteSet(Set):

    """A set given by a finite number of elements."""

    def __init__(self, *elements):
        """Initialize a new instance.

        Parameters
        ----------
        element1, ..., elementN : `Set`
            The elements in the set. Any duplicates are ignored.

        Examples
        --------
        >>> set = odl.FiniteSet(1, 'string')
        """
        self.__elements = tuple(unique(elements))

    @property
    def elements(self):
        """The elements as a tuple."""
        return self.__elements

    def __contains__(self, other):
        """Return ``other in self``.

        Returns
        -------
        contains : bool
            ``True`` if ``other`` is an element in `elements`,
            ``False`` otherwise.

        Examples
        --------
        >>> set = odl.FiniteSet(1, 'string')
        >>> 1 in set
        True
        >>> 2 in set
        False
        >>> 'string' in set
        True
        """
        return other in self.elements

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if ``other`` is a `SetUnion` instance, and
            has the same subsets as this set, ``False`` otherwise.
        """
        # Need to loop since order could be different
        return (type(self) == type(other) and
                all(el in other for el in self) and
                all(el in self for el in other))

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((type(self), set(self.elements)))

    def element(self, inp=None):
        """Create a new element.

        For more specific control, use set[i].element() to pick which subset to
        use.
        """
        if inp is None:
            return self.elements[0]
        elif inp in self.elements:
            return inp
        else:
            raise ValueError('cannot convert inp {} to element in {}'
                             ''.format(inp, self))

    def __getitem__(self, indcs):
        """Return ``self[indcs]``.

        Examples
        --------
        >>> set = odl.FiniteSet(1, 2, 3, 'string')
        >>> set[:3]
        FiniteSet(1, 2, 3)
        >>> set[3]
        'string'
        """
        if isinstance(indcs, slice):
            return FiniteSet(*self.elements[indcs])
        else:
            return self.elements[indcs]

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> odl.FiniteSet(1, 'string')
        FiniteSet(1, 'string')
        """
        elements_str = ', '.join(repr(el) for el in self.elements)
        return '{}({})'.format(self.__class__.__name__, elements_str)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
