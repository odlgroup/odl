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

"""Abstract linear vector spaces.

The classes in this module represent abstract mathematical concepts
of vector spaces. They cannot be used directly but are rather intended
to be subclassed by concrete space implementations. The spaces
provide default implementations of the most important vector space
operations. See the documentation of the respective classes for more
details.

The concept of linear vector spaces in ODL is largely inspired by
the `Rice Vector Library
<http://www.trip.caam.rice.edu/software/rvl/rvl/doc/html/>`_ (RVL).

The abstract `LinearSpace` class is intended for quick prototyping.
It has a number of abstract methods which must be overridden by a
subclass. On the other hand, it provides automatic error checking
and numerous attributes and methods for convenience.

In the following, the abstract methods are explained in detail.

Abstract methods
================

`element(inp=None)`
-------------------
This public method is the factory for the inner
`LinearSpace.Vector` class. It creates a new element of the space,
either from scratch or from an existing data container. In the
simplest possible case, it just delegates the construction to the
`Vector` class.

If no data is provided, the new element is **merely allocated, not
initialized**, thus it can contain *any* value.

**Parameters:**
    inp : object, optional
        A container for values for the element initialization

**Returns:**
    element : `LinearSpace.Vector`
        The new vector

`_lincomb(z, a, x, b, y)`
-------------------------
This private method is the raw implementation (i.e. without error
checking) of the linear combination `z <-- a * x + b * y`.
`_lincomb` and its public counterpart `lincomb` are used to cover
a range of convenience functions, see below.

**Parameters:**
    z : `LinearSpace.Vector`
        Element to which the result of the computation is written
    a,b : scalars, must be members of the space's `field`
        Multiplicative scalar factors for input vector `x` or `y`,
        respectively
    x,y : `LinearSpace.Vector`
        Input vectors

**Returns:** `None`

**Requirements:**
 * Aliasing of `x`, `y` and `z` **must** be allowed.
 * The input vectors `x` and `y` **must not** be modified.
 * The initial state of the output vector `z` **must not**
   influence the result.

`field`
-------
The public attribute determining the type of scalars which
underlie the space. Can be instances of either `RealNumbers` or
`ComplexNumbers` (see `odl.set.sets`).

Should be implemented as a `@property` to make it immutable.

`__eq__(other)`
---------------
`LinearSpace` inherits this abstract method from `Set`. Its
purpose is to check two `LinearSpace` instances for equality.

**Parameters:**
    other : object
        The object to compare to

**Returns:**
    equals : bool
        `True` if `other` is the same `LinearSpace`, `False`
        otherwise

Optional methods
================

`_dist(x, y)`
-------------
A raw (not type-checking) private method measuring the distance
between two vectors `x` and `y`.

A space with a distance is called a **metric space**.

**Parameters:**
    x,y : `LinearSpace.Vector`
        Vectors whose mutual distance to calculate

**Returns:**
    distance : float
        The distance between `x` and `y`, measured in the space's
        metric

**Requirements:**
    * `_dist(x, y) == _dist(y, x)`
    * `_dist(x, y) <= _dist(x, z) + _dist(z, y)`
    * `_dist(x, y) >= 0`
    * `_dist(x, y) == 0` (approx.) if and only if `x == y` (approx.)

`_norm(x)`
----------
A raw (not type-checking) private method measuring the length of a
space element `x`.

A space with a norm is called a **normed space**.

**Parameters:**
    x : `LinearSpace.Vector`
        The vector to measure

**Returns:**
    norm : float
        The length of `x` as measured in the space's norm

**Requirements:**
 * `_norm(s * x) = |s| * _norm(x)` for any scalar `s`
 * `_norm(x + y) <= _norm(x) + _norm(y)`
 * `_norm(x) >= 0`
 * `_norm(x) == 0` (approx.) if and only if `x == 0` (approx.)

Note
----
A normed space is automatically a metric space with the distance
function `_dist(x, y) = _norm(x - y)`.


`_inner(x, y)`
--------------
A raw (not type-checking) private method calculating the inner
product of two space elements `x` and `y`.

**Parameters:**
    x,y : `LinearSpace.Vector`
        Vectors whose inner product to calculate

**Returns:**
    inner : float or complex
        The inner product of `x` and `y`. If `field` is the real
        numbers, `inner` is a float, otherwise complex.

**Requirements:**
 * `_inner(x, y) == _inner(y, x)^*` with '*' = complex conjugation
 * `_inner(s * x, y) == s * _inner(x, y)` for `s` scalar
 * `_inner(x + z, y) == _inner(x, y) + _inner(z, y)`
 * `_inner(x, x) == 0` (approx.) if and only if `x == 0` (approx.)

Note
----
A Hilbert space is automatically a normed space with the norm function
`_norm(x) = sqrt(_inner(x, x))`, and in consequence also a metric space
with the distance function `_dist(x, y) = _norm(x - y)`.


`_multiply(z, x, y)`
--------------------
A raw (not type-checking) private method multiplying two vectors
`x` and `y` element-wise and storing the result in `z`.

**Parameters:**
    z : `LinearSpace.Vector`
        Vector to store the result
    x,y : `LinearSpace.Vector`
        Vectors whose element-wise product to calculate

**Returns:** `None`

**Requirements:**
 * `_multiply(z, x, y) <==> _multiply(z, y, x)`
 * `_multiply(z, s * x, y) <==> _multiply(z, x, y); z *= s  <==>
    _multiply(z, x, s * y)` for any scalar `s`
 * There is a space element `one` with
   `z` after `_multiply(z, one, x)` or `_multiply(z, x, one)`
   equals `x`.

Note
----
The above conditions on the multiplication constitute a
*unital commutative algebra* in the mathematical sense.

Notes
-----
See Wikipedia's mathematical overview articles
`Vector space
<https://en.wikipedia.org/wiki/Vector_space>`_,

`Algebra
<https://en.wikipedia.org/wiki/Associative_algebra>`_.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from builtins import object, str
from future import standard_library
standard_library.install_aliases()
from future.utils import with_metaclass

# External module imports
from abc import ABCMeta, abstractmethod, abstractproperty
import math as m

# ODL imports
from odl.sets.set import Set, UniversalSet


__all__ = ('LinearSpace',)


class LinearSpace(Set):

    """Abstract linear vector space.

    Its elements are represented as instances of the inner
    `LinearSpace.Vector` class.
    """

    @abstractmethod
    def element(self, inp=None):
        """Create an element from `inp` or from scratch.

        If called without `inp` argument, an arbitrary element in the
        space is generated without guarantee of its state.

        Parameters
        ----------
        inp : `object`, optional
            The input data from which to create the element

        Returns
        -------
        element : `LinearSpace.Vector`
            A vector in this space
        """

    @abstractmethod
    def _lincomb(self, z, a, x, b, y):
        """Calculate z = a*x + b*y.

        This method is intended to be private, public callers should
        resort to lincomb which is type-checked.
        """

    def _dist(self, x, y):
        """Calculate the distance between x and y.

        This method is intended to be private, public callers should
        resort to `dist` which is type-checked.
        """
        # default implementation
        return self.norm(x-y)

    def _norm(self, x):
        """Calculate the norm of x.

        This method is intended to be private, public callers should
        resort to `norm` which is type-checked.
        """
        # default implementation
        return m.sqrt(self.inner(x, x).real)

    def _inner(self, x, y):
        """Calculate the inner product of x and y.

        This method is intended to be private, public callers should
        resort to `inner` which is type-checked.
        """
        # No default implementation possible
        raise NotImplementedError('inner product not implemented in space {!r}'
                                  ''.format(self))

    def _multiply(self, z, x, y):
        """Calculate the pointwise multiplication z = x * y.

        This method is intended to be private, public callers should
        resort to `multiply` which is type-checked.
        """
        # No default implementation possible
        raise NotImplementedError('multiplication not implemented in space '
                                  '{!r}'.format(self))

    @abstractproperty
    def field(self):
        """The field of this vector space."""

    # Default methods
    def zero(self):
        """A zero vector in this space.

        The zero vector is defined as the additive unit of a space.

        Parameters
        ----------
        None

        Returns
        -------
        v : Vector
            The zero vector of this space
        """
        # Default implementation using lincomb
        tmp = self.element()
        self._lincomb(tmp, 0, tmp, 0, tmp)
        return tmp

    def __contains__(self, other):
        """`s.__contains__(other) <==> other in s`.

        Returns
        -------
        contains : `bool`
            `True` if `other` is a `LinearSpace.Vector` instance and
            `other.space` is equal to this space, `False` otherwise.

        Notes
        -----
        This is the strict default where spaces must be equal.
        Subclasses may choose to implement a less strict check.
        """
        return isinstance(other, LinearSpace.Vector) and other.space == self

    # Error checking variant of methods
    def lincomb(self, z, a, x, b=None, y=None):
        """Linear combination of vectors.

        Calculates

        z = a * x

        or, if b and y are given,

        z = a*x + b*y

        with error checking of types.

        Parameters
        ----------
        z : Vector
            The Vector that the result should be written to.
        a : Scalar in the field of this space
            Scalar to multiply `x` with.
        x : Vector
            The first of the summands
        b : Scalar, optional
            Scalar to multiply `y` with.
        y : Vector, optional
            The second of the summands

        Returns
        -------
        None

        Notes
        -----
        The vectors `z`, `x` and `y` may be aligned, thus a call

        space.lincomb(x, 2, x, 3.14, x)

        is (mathematically) equivalent to

        x = x * (1 + 2 + 3.14)
        """
        if z not in self:
            raise TypeError('output vector {!r} not in space {!r}.'
                            ''.format(z, self))

        if a not in self.field:
            raise TypeError('first scalar {!r} not in the field {!r} of the '
                            'space {!r}.'.format(a, self.field, self))

        if x not in self:
            raise TypeError('first input vector {!r} not in space {!r}.'
                            ''.format(x, self))

        if b is None:  # Single argument
            if y is not None:
                raise ValueError('second input vector provided but no '
                                 'second scalar.')

            # Call method
            return self._lincomb(z, a, x, 0, x)
        else:  # Two arguments
            if b not in self.field:
                raise TypeError('second scalar {!r} not in the field {!r} of '
                                'the space {!r}.'.format(b, self.field, self))

            if y not in self:
                raise TypeError('second input vector {!r} not in space {!r}.'
                                ''.format(x, self))

            # Call method
            return self._lincomb(z, a, x, b, y)

    def dist(self, x, y):
        """Calculate the distance between two vectors.

        Parameters
        ----------
        x : `LinearSpace.Vector`
            The first element

        y : LinearSpace.Vector
            The second element

        Returns
        -------
        dist : RealNumber
               Distance between vectors
        """
        if x not in self:
            raise TypeError('first vector {!r} not in space {!r}'
                            ''.format(x, self))
        if y not in self:
            raise TypeError('second vector {!r} not in space {!r}'
                            ''.format(y, self))

        return float(self._dist(x, y))

    def norm(self, x):
        """Calculate the norm of a vector."""
        if x not in self:
            raise TypeError('vector {!r} not in space {!r}'.format(x, self))

        return float(self._norm(x))

    def inner(self, x, y):
        """Calculate the inner product of the vectors x and y."""
        if x not in self:
            raise TypeError('first vector {!r} not in space {!r}'
                            ''.format(x, self))
        if y not in self:
            raise TypeError('second vector {!r} not in space {!r}'
                            ''.format(y, self))

        return self.field.element(self._inner(x, y))

    def multiply(self, z, x, y):
        """Calculate the pointwise product of x and y, and assign to z."""
        if x not in self:
            raise TypeError('first vector {!r} not in space {!r}'
                            ''.format(x, self))
        if y not in self:
            raise TypeError('second vector {!r} not in space {!r}'
                            ''.format(y, self))

        self._multiply(z, x, y)

    class Vector(with_metaclass(ABCMeta, object)):

        """Abstract `LinearSpace` element.

        Not intended for creation of vectors, use the space's
        `element()` method instead.
        """

        def __init__(self, space):
            """Default initializer of vectors.

            All deriving classes must call this method to set space.
            """
            if not isinstance(space, LinearSpace):
                raise TypeError('space {!r} is not a `LinearSpace` instance'
                                ''.format(space))
            self._space = space

        @property
        def space(self):
            """Space to which this vector belongs."""
            return self._space

        # Convenience functions
        def assign(self, other):
            """Assign the values of other to this vector."""
            self.space.lincomb(self, 1, other)

        def copy(self):
            """Create an identical (deep) copy of this vector."""
            result = self.space.element()
            result.assign(self)
            return result

        def lincomb(self, a, x, b=None, y=None):
            """Assign a linear combination to this vector.

            Implemented as `space.lincomb(self, a, x, b, y)`.
            """
            self.space.lincomb(self, a, x, b, y)

        def set_zero(self):
            """Set this vector to the zero vector."""
            self.space.lincomb(self, 0, self, 0, self)

        # Convenience operators
        def __iadd__(self, other):
            """Implementation of 'self += other'."""
            self.space.lincomb(self, 1, self, 1, other)
            return self

        def __isub__(self, other):
            """Implementation of 'self -= other'."""
            self.space.lincomb(self, 1, self, -1, other)
            return self

        def __imul__(self, other):
            """Implementation of 'self *= other'."""
            if other in self.space:
                self.space.multiply(self, other, self)
            else:
                self.space.lincomb(self, other, self)
            return self

        def __itruediv__(self, other):
            """Implementation of 'self /= other' (true division)."""
            return self.__imul__(1.0 / other)

        def __idiv__(self, other):
            """Implementation of 'self /= other'."""
            return self.__itruediv__(other)

        def __add__(self, other):
            """Implementation of 'self + other'."""
            tmp = self.space.element()
            self.space.lincomb(tmp, 1, self, 1, other)
            return tmp

        def __sub__(self, other):
            """Implementation of 'self - other'."""
            tmp = self.space.element()
            self.space.lincomb(tmp, 1, self, -1, other)
            return tmp

        def __mul__(self, other):
            """Implementation of 'self * other'."""
            tmp = self.space.element()
            if other in self.space:
                self.space.multiply(tmp, other, self)
            else:
                self.space.lincomb(tmp, other, self)
            return tmp

        def __ipow__(self, n):
            """Take the n:th power of self, only defined for integer n"""
            if n == 1:
                return self
            elif n % 2 == 0:
                self.space.multiply(self, self, self)
                return self.__ipow__(n//2)
            else:
                tmp = self.copy()
                for i in range(n):
                    self.space.multiply(tmp, tmp, self)
                return tmp

        def __pow__(self, n):
            """Take the n:th power of self, only defined for integer n"""
            tmp = self.copy()
            for i in range(n):
                self.space.multiply(tmp, tmp, self)
            return tmp

        def __rmul__(self, other):
            """Implementation of 'other * self'."""
            return self.__mul__(other)

        def __truediv__(self, other):
            """Implementation of 'self / other' (true division)."""
            return self.__mul__(1.0 / other)

        def __div__(self, other):
            """Implementation of 'self / scalar'."""
            return self.__truediv__(other)

        def __neg__(self):
            """Implementation of '-self'."""
            tmp = self.space.element()
            self.space.lincomb(tmp, -1.0, self)
            return tmp

        def __pos__(self):
            """Implementation of '+self'."""
            return self.copy()

        # Metric space method
        def __eq__(self, other):
            """`vec.__eq__(other) <==> vec == other`.

            Two vectors are equal if their distance is 0

            Parameters
            ----------
            other : LinearSpace.Vector
                Vector in this space.

            Returns
            -------
            equals : bool
                True if the vectors are equal, else false.

            Note
            ----
            Equality is very sensitive to numerical errors, thus any
            operations on a vector should be expected to break equality
            testing.

            Examples
            --------
            >>> from odl import Rn
            >>> import numpy as np
            >>> rn = Rn(1, norm=np.linalg.norm)
            >>> x = rn.element([0.1])
            >>> x == x
            True
            >>> y = rn.element([0.1])
            >>> x == y
            True
            >>> z = rn.element([0.3])
            >>> x+x+x == z
            False
            """
            if (not isinstance(other, LinearSpace.Vector) or
                    other.space != self.space):
                # Cannot use (if other not in self.space) since this is not
                # reflexive.
                return False
            elif other is self:
                # Optimization for the most common case
                return True
            else:
                return self.space.dist(self, other) == 0

        def __ne__(self, other):
            return not self.__eq__(other)

        def __str__(self):
            """Implementation of str()."""
            return str(self.space) + ".Vector"

        # TODO: DECIDE ON THESE + DOCUMENT
        def norm(self):
            return self.space.norm(self)

        def dist(self, other):
            return self.space.dist(self, other)

        def inner(self, other):
            return self.space.inner(self, other)

        def multiply(self, x, y):
            return self.space.multiply(self, x, y)


class UniversalSpace(LinearSpace):

    """A dummy linear space class mostly raising `NotImplementedError`."""

    def element(self, inp=None):
        """Dummy element creation method, raises `NotImplementedError`."""
        raise NotImplementedError

    def _lincomb(self, z, a, x, b, y):
        """Dummy linear combination, raises `NotImplementedError`."""
        raise NotImplementedError

    def _dist(self, x, y):
        """Dummy distance method, raises `NotImplementedError`."""
        raise NotImplementedError

    def _norm(self, x):
        """Dummy norm method, raises `NotImplementedError`."""
        raise NotImplementedError

    def _inner(self, x, y):
        """Dummy inner product method, raises `NotImplementedError`."""
        raise NotImplementedError

    def _multiply(self, z, x, y):
        """Dummy multiplication method, raises `NotImplementedError`."""
        raise NotImplementedError

    @property
    def field(self):
        """Dummy field `UniversalSet`."""
        return UniversalSet()

    def __eq__(self, other):
        """`s.__eq__(other) <==> s == other`.

        Dummy check, `True` for any `LinearSpace`.
        """
        return isinstance(other, LinearSpace)

    def __contains__(self, other):
        """`s.__contains__(other) <==> other in s`.

        Dummy membership check, `True` for any `LinearSpace.Vector`.
        """
        return True


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
