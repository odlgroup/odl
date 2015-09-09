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

"""Basic abstract and concrete sets.

List of classes
===============

+--------------------+---------+----------------------------------------+
|Class name          |Direct   |Description                             |
|                    |ancestors|                                        |
+====================+=========+========================================+
|``Set``             |`object` |**Abstract** base class for sets        |
+--------------------+---------+----------------------------------------+
|``EmptySet``        |``Set``  |Empty set, contains only `None`         |
+--------------------+---------+----------------------------------------+
|``UniversalSet``    |``Set``  |Contains everything                     |
+--------------------+---------+----------------------------------------+
|``Integers``        |``Set``  |Set of (signed) integers                |
+--------------------+---------+----------------------------------------+
|``RealNumbers``     |``Set``  |Set of real numbers                     |
+--------------------+---------+----------------------------------------+
|``ComplexNumbers``  |``Set``  |Set of complex numbers                  |
+--------------------+---------+----------------------------------------+
|``CartesianProduct``|``Set``  |Set of tuples with the i-th entry being |
|                    |         |an element of the i-th factor (set)     |
+--------------------+---------+----------------------------------------+
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from __future__ import unicode_literals
from builtins import int, object, str, zip
from future.utils import with_metaclass
from future import standard_library
standard_library.install_aliases()

# External imports
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real, Complex
import numpy as np

# ODL imports


class Set(with_metaclass(ABCMeta, object)):

    """An abstract set.

    Abstract Methods
    ----------------
    Each subclass of `Set` must implement two methods: one to check if
    an object is contained in the set and one to test if two sets are
    equal.

    `contains(self, other)`
    ~~~~~~~~~~~~~~~~~~~~~~~
    Test if `other` is a member of this set.

    **Parameters:**
        other : `object`
            The object to be tested for membership.

    **Returns:**
        equals : `boolean`
            `True` if `other` is a member of this set, `False`
            otherwise.


    `equals(self, other)`
    ~~~~~~~~~~~~~~~~~~~~~
    Test if `other` is the same set as this set, i.e. both sets are
    of the same type and contain the same elements.

    **Parameters:**
        other : `object`
            The object to be tested for equality.

    **Returns:**
        equals : `boolean`
            `True` if both sets are of the same type and contain the
            same elements, `False` otherwise.

    optional: `element(inp=None)`
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Create an element of this set, either from scratch or from an
    input parameter.

    **Parameters:**
        inp : `object`, optional
            The object from which to create the new element.

    **Returns:**
        element : member of this set
            If `inp == None`, return an arbitrary element.
            Otherwise, return the element created from `inp`.

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

    @abstractmethod
    def contains(self, other):
        """Test if `other` is a member of this set."""

    @abstractmethod
    def equals(self, other):
        """Test if `other` is the same set as this set."""

    def element(self, inp=None):
        """Return an element from `inp` or from scratch."""
        raise NotImplementedError("'element' method not implemented")

    # Default implemenations
    def __eq__(self, other):
        """s.__eq__(other) <==> s == other."""
        return self.equals(other)

    def __ne__(self, other):
        """s.__ne__(other) <==> s != other."""
        return not self.equals(other)

    def __contains__(self, other):
        """s.__contains__(other) <==> other in s."""
        return self.contains(other)


class EmptySet(Set):

    """The empty set.

    `None` is considered as "no element", i.e.
    `None in EmptySet() is True`
    """

    def contains(self, other):
        """Test if `other` is `None`."""
        return other is None

    def equals(self, other):
        """Test if `other` is an `EmptySet` instance."""
        return type(other) == type(self)

    def element(self, inp=None):
        """Return `None`."""
        return None

    def __str__(self):
        """s.__str__() <==> str(s)."""
        return "EmptySet"

    def __repr__(self):
        """s.__repr__() <==> repr(s)."""
        return "EmptySet()"


class UniversalSet(Set):
    """The set of all sets.

    Forget about set theory for a moment :-).
    """

    def contains(self, other):
        """Return `True`."""
        return True

    def equals(self, other):
        """Test if `other` is a `UniversalSet` instance."""
        return type(other) == type(self)

    def element(self, inp=None):
        """Return `inp` in any case."""
        return inp

    def __str__(self):
        """s.__str__() <==> str(s)."""
        return "UniversalSet"

    def __repr__(self):
        """s.__repr__() <==> repr(s)."""
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
        """The `length` attribute."""
        return self._length

    def contains(self, other):
        """Test if `other` is a string of at max `length` characters."""
        return isinstance(other, str) and len(other) <= self.length

    def equals(self, other):
        """Test if `other` is a `Strings` instance of equal length."""
        return type(other) == type(self) and other.length == self.length

    def element(self, inp=None):
        """Return a string from `inp` or from scratch."""
        if inp is not None:
            return str(inp)[:self.length]
        else:
            return ''

    def __str__(self):
        """s.__str__() <==> str(s)."""
        return 'Strings({})'.format(self.length)

    def __repr__(self):
        """s.__repr__() <==> repr(s)."""
        return 'Strings({})'.format(self.length)


class Integers(Set):

    """The set of integers."""
    def equals(self, other):
        """Tests if `other` is an `Integers` instance."""
        return type(other) == type(self)

    def contains(self, other):
        """Test if `other` is an integer."""
        return isinstance(other, Integral)

    def element(self, inp=None):
        """Return an integer from `inp` or from scratch."""
        if inp is not None:
            return int(inp)
        else:
            return 0

    def __str__(self):
        """s.__str__() <==> str(s)."""
        return "Integers"

    def __repr__(self):
        """s.__repr__() <==> repr(s)."""
        return "Integers()"


class RealNumbers(Set):
    """The set of real numbers."""

    def contains(self, other):
        """Test if `other` is a real number."""
        return isinstance(other, Real)

    def equals(self, other):
        """Test if `other` is a `RealNumbers` instance."""
        return type(other) == type(self)

    def element(self, inp=None):
        """Return a real number from `inp` or from scratch."""
        if inp is not None:
            return float(inp)
        else:
            return 0.0

    def __str__(self):
        """s.__str__() <==> str(s)."""
        return "RealNumbers"

    def __repr__(self):
        """s.__repr__() <==> repr(s)."""
        return "RealNumbers()"


class ComplexNumbers(Set):

    """The set of complex numbers."""

    def contains(self, other):
        """Test if `other` is a complex number."""
        return isinstance(other, Complex)

    def equals(self, other):
        """Test if `other` is a `ComplexNumbers` instance."""
        return type(other) == type(self)

    def element(self, inp=None):
        """Return a complex number from `inp` or from scratch."""
        if inp is not None:
            return complex(inp)
        else:
            return complex(0.0, 0.0)

    def __str__(self):
        """s.__str__() <==> str(s)."""
        return "ComplexNumbers"

    def __repr__(self):
        """s.__repr__() <==> repr(s)."""
        return "ComplexNumbers()"


class CartesianProduct(Set):

    """The Cartesian product of `n` sets.

    The elements of this set are `n`-tuples where the i-th entry
    is an element of the i-th set.
    """

    def __init__(self, *sets):
        """Initialize a new instance."""
        if not all(isinstance(set_, Set) for set_ in sets):
            wrong = [set_ for set_ in sets
                     if not isinstance(set_, Set)]
            raise TypeError('{} not Set instance(s)'.format(wrong))

        self._sets = sets

    @property
    def sets(self):
        """The factors (sets) as a tuple."""
        return self._sets

    def contains(self, other):
        """Test if `other` is contained in this set.

        Returns
        -------
        contains : `boolean`
            `True` if `other` has the same length as this Cartesian
            product and each entry is contained in the set with
            corresponding index, `False` otherwise.
        """
        try:
            iter(other)
        except TypeError:
            return False
        return (len(other) == len(self) and
                all(p in set_ for set_, p in zip(self.sets, other)))

    def equals(self, other):
        """Test if `other` is contained in this set.

        Returns
        -------
        equals : `boolean`
            `True` if `other` is a `CartesianProduct` instance, has
            the same length as this Cartesian product and all sets
            with the same index are equal, `False` otherwise.
        """
        return (type(other) == type(self) and
                len(other) == len(self) and
                all(so == ss for so, ss in zip(other.sets, self.sets)))

    def element(self, inp=None):
        """Create a `CartesianProduct` element.

        Parameters
        ----------
        inp : iterable, optional
            Collection of input values for the `element()` methods
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
        """s.__len__() <==> len(s)."""
        return len(self.sets)

    def __getitem__(self, indcs):
        """s.__getitem__(indcs) <==> s[indcs].

        Examples
        --------
        >>> emp, univ = EmptySet(), UniversalSet()
        >>> prod = CartesianProduct(emp, univ, univ, emp, emp)
        >>> prod[2]
        UniversalSet()
        >>> prod[2:4]
        CartesianProduct(UniversalSet(), EmptySet())
        """
        try:
            return self.sets[int(indcs)]  # single index
        except TypeError:
            index_arr = np.arange(len(self))[indcs]
            set_tpl = tuple(self.sets[i] for i in index_arr)
            return CartesianProduct(*set_tpl)

    def __str__(self):
        """s.__str__() <==> str(s)."""
        return ' x '.join(str(set_) for set_ in self.sets)

    def __repr__(self):
        """s.__repr__() <==> repr(s)."""
        sets_str = ', '.join(repr(set_) for set_ in self.sets)
        return 'CartesianProduct({})'.format(sets_str)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
