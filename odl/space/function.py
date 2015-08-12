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

"""
Support for functionspaces, such as L2.
"""

# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from future import standard_library
standard_library.install_aliases()
from builtins import super

# ODL imports
from odl.operator.operator import Operator
from odl.space.set import RealNumbers, ComplexNumbers, Set
from odl.space.space import Algebra
from odl.utility.utility import errfmt


class FunctionSet(Set):

    """A general set of functions with common domain and range.

    Attributes
    ----------

    +----------+-----------+------------------------------------------+
    |Name      |Type       |Description                               |
    +==========+===========+==========================================+
    |`domain`  |`Set`      |The domain of all functions in this set   |
    +----------+-----------+------------------------------------------+
    |`range`   |`Set`      |The range of all functions in this set    |
    +----------+-----------+------------------------------------------+

    Methods
    -------

    +-----------------+--------------------+--------------------------+
    |Signature        |Return type         |Description               |
    +=================+====================+==========================+
    |`element(func)`  |`FunctionSet.Vector`|Create an element in this |
    |                 |                    |`FunctionSet`.            |
    +-----------------+--------------------+--------------------------+
    |`equals(other)`  |`boolean`           |Test if `other` is equal  |
    |                 |                    |to this `FunctionSet`.    |
    +-----------------+--------------------+--------------------------+
    |`contains(other)`|`boolean`           |Test if `other` is        |
    |                 |                    |contained in this         |
    |                 |                    |`FunctionSet`.            |
    +-----------------+--------------------+--------------------------+

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

    def __init__(self, dom, ran):
        """Initialize a new `FunctionSet` instance.

        Parameters
        ----------
        dom : `Set`
            The domain of the functions.
        ran : `Set`
            The range of the functions.
        """
        if not isinstance(dom, Set):
            raise TypeError(errfmt('''
            `dom` {} not a `Set` instance.'''.format(dom)))

        if not isinstance(ran, Set):
            raise TypeError(errfmt('''
            `ran` {} not a `Set` instance.'''.format(dom)))

        self._domain = dom
        self._range = ran

    @property
    def domain(self):
        """Return `domain` attribute."""
        return self._domain

    @property
    def range(self):
        """Return `range` attribute."""
        return self._range

    def element(self, func):
        """Create a `FunctionSet` element.

        Parameters
        ----------
        `func` : callable
            The actual instruction executed when evaluating
            this element

        Returns
        -------
        `element` : `FunctionSet.Vector`
            The new element created from `func`
        """
        if isinstance(func, self.Vector):  # no double wrapping
            return self.element(func.function)
        else:
            return self.Vector(self, func)

    def equals(self, other):
        """Test if `other` is equal to this set.

        Paramters
        ---------
        other : `object`
            The object to test for equality.

        Returns
        -------
        equals : `boolean`
            `True` if `other` is a `FunctionSet` with same `domain`
            and `range`, `False` otherwise.
        """
        return (isinstance(other, FunctionSet) and
                self.domain == other.domain and
                self.range == other.range)

    def contains(self, other):
        """Test if `other` is contained in this set.

        Paramters
        ---------
        other : `object`
            The object to test for membership.

        Returns
        -------
        equals : `boolean`
            `True` if `other` is a `FunctionSet.Vector` whose `space`
            attribute equals this space, `False` otherwise.
        """
        return (isinstance(other, FunctionSet.Vector) and
                self == other.space)

    def __repr__(self):
        """`s.__repr__() <==> repr(s)`."""
        return 'FunctionSet({!r}, {!r})'.format(self.domain, self.range)

    def __str__(self):
        """`s.__str__() <==> str(s)`."""
        return 'FunctionSet({}, {})'.format(self.domain, self.range)

    class Vector(Operator):

        """Representation of a `FunctionSet` element."""

        def __init__(self, func_set, function):
            """Initialize a new `FunctionSet.Vector`.

            Parameters
            ----------
            func_set : `FunctionSet`
                The set of functions this element lives in
            function : callable
                The actual instruction executed when evaluating
                this element
            """
            if not isinstance(func_set, FunctionSet):
                raise TypeError(errfmt('''
                `func_set` {} not a `FunctionSet` instance.
                '''.format(func_set)))

            if not callable(function):
                raise TypeError(errfmt('''
                `function` {} is not callable.'''.format(function)))

            self._space = func_set
            self._function = function

        @property
        def space(self):
            """Return `space` attribute."""
            return self._space

        @property
        def function(self):
            """Return `function` attribute."""
            return self._function

        @property
        def domain(self):
            """The function domain (abstract in `Operator`)."""
            return self.space.domain

        @property
        def range(self):
            """The function range (abstract in `Operator`)."""
            return self.space.range

        def _call(self, inp):
            """The raw `call` method for out-of-place evaluation.

            Parameters
            ----------
            inp : `domain` element
                The point at which to evaluate the function

            Returns
            -------
            outp : `range` element
                The function value at the point
            """
            return self.function(inp)

        def equals(self, other):
            """Test `other` for equality."""
            return (isinstance(other, FunctionSet.Vector) and
                    self.space == other.space and
                    self.function == other.function)

        def __eq__(self, other):
            """`vec.__eq__(other) <==> vec == other`"""
            return self.equals(other)

        def __ne__(self, other):
            """`vec.__ne__(other) <==> vec != other`"""
            return not self.equals(other)


class FunctionSpace(FunctionSet, Algebra):

    """A vector space of functions."""

    def __init__(self, dom, field):
        """Initialize a new `FunctionSpace` instance.

        Parameters
        ----------
        dom : `Set`
            The domain of the functions.
        field : `RealNumbers` or `ComplexNumbers` instance
            The range of the functions.
        """
        if not isinstance(dom, Set):
            raise TypeError(errfmt('''
            `dom` {} not a `Set` instance.'''.format(dom)))

        if not (isinstance(field, RealNumbers) or
                isinstance(field, ComplexNumbers)):
            raise TypeError(errfmt('''
            `field` {} not a `RealNumbers` or `ComplexNumbers` instance.
            '''.format(field)))

        super().__init__(dom, field)
        self._field = field

    @property
    def field(self):
        """Return `field` attribute."""
        return self._field

    def element(self, func=None):
        """Create an element from `func` or from scratch.

        Parameters
        ----------
        `func` : callable, optional
            The actual instruction executed when evaluating
            this element.

        Returns
        -------
        `element` : `FunctionSpace.Vector`
            The new element created from `func`
        """
        if func is None:
            return self.zero()
        else:
            return self.Vector(self, func)

    def _lincomb(self, z, a, x, b, y):
        """Linear combination of `x` and `y`.

        Set z = a*x + b*y in the sense of pointwise arithmetics.

        Parameters
        ----------
        z : `FunctionSpace.Vector`
            The Vector that the result is written to.
        a : element of `field`
            Scalar to multiply `x` with.
        x : `FunctionSpace.Vector`
            The first of the summands
        b : element of `field`
            Scalar to multiply `y` with.
        y : `FunctionSpace.Vector`
            The second of the summands

        Returns
        -------
        None

        Note
        ----
        The additions and multiplications are implemented via a simple
        Python function, so the resulting function is probably slow.
        """
        # Store to allow aliasing
        x_old = x._function
        y_old = y._function

        def lincomb_function(arg):
            """The function calculating the linear combination."""
            return a * x_old(arg) + b * y_old(arg)

        z._function = lincomb_function

    def zero(self):
        """The function mapping everything to zero.

        Since `lincomb` is slow, we implement this function directly.
        """
        def zero_(*_):
            """The zero function."""
            return self.field.element(0.0)
        return self.element(zero_)

    def equals(self, other):
        """Test if `other` is equal to this space.

        Paramters
        ---------
        other : `object`
            The object to test for equality.

        Returns
        -------
        equals : `boolean`
            `True` if `other` is a `FunctionSpace` with same `domain`
            and `range`, `False` otherwise.
        """
        # TODO: this more strict equality notion is not in line with the
        # current hierarchy of Cartesian spaces. Make a choice!
        return (isinstance(other, FunctionSpace) and
                self.domain == other.domain and
                self.range == other.range)

    def _multiply(x, y):
        """Raw pointwise multiplication of two functions.

        Parameters
        ----------
        x : `FunctionSpace.Vector`
            First factor
        y : `FunctionSpace.Vector`
            Second factor, used to store the result

        Returns
        -------
        `None`

        Note
        ----
        The multiplication is implemented with a simple Python
        function, so the resulting function object is probably slow.
        """
        x_old = x.function
        y_old = y.function

        def product(arg):
            """The actual product function."""
            return x_old(arg) * y_old(arg)
        y._function = product

    class Vector(FunctionSet.Vector, Algebra.Vector):

        """Representation of a `FunctionSpace` element."""

        def __init__(self, space, function):
            """Initialize a new `FunctionSet.Vector`.

            Parameters
            ----------
            space : `FunctionSpace`
                The space of functions this element lives in
            function : callable
                The actual instruction executed when evaluating
                this element
            """
            if not isinstance(space, FunctionSpace):
                raise TypeError(errfmt('''
                `space` {} not a `FunctionSpace` instance.'''.format(space)))

            if not callable(function):
                raise TypeError(errfmt('''
                `function` {} is not callable.'''.format(function)))

            super().__init__(space, function)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
