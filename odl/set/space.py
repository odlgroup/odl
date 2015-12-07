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
to be sub-classed by concrete space implementations. The spaces
provide default implementations of the most important vector space
operations. See the documentation of the respective classes for more
details.

The abstract `LinearSpace` class is intended for quick prototyping.
It has a number of abstract methods which must be overridden by a
subclass. On the other hand, it provides automatic error checking
and numerous attributes and methods for convenience.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from builtins import object, range, str
from future import standard_library
standard_library.install_aliases()

# External module imports
from abc import abstractmethod
import math as m

# ODL imports
from odl.set.sets import Set, UniversalSet


__all__ = ('LinearSpace', 'LinearSpaceVector', 'UniversalSpace')


class LinearSpace(Set):
    """Abstract linear vector space.

    Its elements are represented as instances of the inner
    `LinearSpaceVector` class.
    """

    def __init__(self, field):
        """Initialize a LinearSpace.

        This method should be called by all inheriting methods so that the
        field property of the space is set properly.
        """
        self._field = field

    @property
    def field(self):
        """The field of this vector space."""
        return self._field

    @abstractmethod
    def element(self, inp=None, **kwargs):
        """Create a `LinearSpaceVector` from ``inp`` or from scratch.

        If called without ``inp`` argument, an arbitrary element in the
        space is generated without guarantee of its state.

        Parameters
        ----------
        inp : `object`, optional
            The input data from which to create the element
        **args : `dict`, optional
            Optional further arguments.

        Returns
        -------
        element : `LinearSpaceVector`
            A vector in this space
        """

    @abstractmethod
    def _lincomb(self, a, x1, b, x2, out):
        """Calculate ``out = a*x1 + b*x2``.

        This method is intended to be private, public callers should
        resort to `lincomb` which is type-checked.
        """

    def _dist(self, x1, x2):
        """Calculate the distance between x1 and x2.

        This method is intended to be private, public callers should
        resort to `dist` which is type-checked.
        """
        # default implementation
        return self.norm(x1 - x2)

    def _norm(self, x):
        """Calculate the norm of x.

        This method is intended to be private, public callers should
        resort to `norm` which is type-checked.
        """
        # default implementation
        return m.sqrt(self.inner(x, x).real)

    def _inner(self, x1, x2):
        """Calculate the inner product of x1 and x2.

        This method is intended to be private, public callers should
        resort to `inner` which is type-checked.
        """
        # No default implementation possible
        raise NotImplementedError('inner product not implemented in space {!r}'
                                  ''.format(self))

    def _multiply(self, x1, x2, out):
        """Calculate the pointwise multiplication out = x1 * x2.

        This method is intended to be private, public callers should
        resort to `multiply` which is type-checked.
        """
        # No default implementation possible
        raise NotImplementedError('multiplication not implemented in space '
                                  '{!r}'.format(self))

    def one(self):
        """A one vector in this space.

        The one vector is defined as the multiplicative unit of a space.

        Returns
        -------
        v : `LinearSpaceVector`
            The one vector of this space
        """
        raise NotImplementedError('This space has no one')

    # Default methods
    def zero(self):
        """A zero vector in this space.

        The zero vector is defined as the additive unit of a space.

        Returns
        -------
        v : `LinearSpaceVector`
            The zero vector of this space
        """
        # Default implementation using lincomb
        tmp = self.element()
        self._lincomb(tmp, 0, tmp, 0, tmp)
        return tmp

    def __contains__(self, other):
        """Return ``other in self``.

        Returns
        -------
        contains : `bool`
            `True` if ``other`` is a `LinearSpaceVector` instance and
            ``other.space`` is equal to this space, `False` otherwise.

        Notes
        -----
        This is the strict default where spaces must be equal.
        Subclasses may choose to implement a less strict check.
        """
        return getattr(other, 'space', None) == self

    # Error checking variant of methods
    def lincomb(self, a, x1, b=None, x2=None, out=None):
        """Linear combination of vectors.

        Calculates

        ``out = a * x1``

        or, if b and y are given,

        ``out = a*x1 + b*x2``

        with error checking of types.

        Parameters
        ----------
        a : Scalar in the field of this space
            Scalar to multiply ``x1`` with.
        x1 : `LinearSpaceVector`
            The first of the summands
        b : Scalar, optional
            Scalar to multiply ``x2`` with.
        x2 : `LinearSpaceVector`, optional
            The second of the summands
        out : `LinearSpaceVector`, optional
            The Vector that the result should be written to.

        Returns
        -------
        out : `LinearSpaceVector`

        Notes
        -----
        The vectors ``out``, ``x1`` and ``x2`` may be aligned, thus a call

        ``space.lincomb(x, 2, x, 3.14, out=x)``

        is (mathematically) equivalent to

        ``x = x * (1 + 2 + 3.14)``
        """
        if out is None:
            out = self.element()
        elif out not in self:
            raise TypeError('output vector {!r} not in space {!r}.'
                            ''.format(out, self))

        if a not in self.field:
            raise TypeError('first scalar {!r} not in the field {!r} of the '
                            'space {!r}.'.format(a, self.field, self))

        if x1 not in self:
            raise TypeError('first input vector {!r} not in space {!r}.'
                            ''.format(x1, self))

        if b is None:  # Single argument
            if x2 is not None:
                raise ValueError('second input vector provided but no '
                                 'second scalar.')

            # Call method
            self._lincomb(a, x1, 0, x1, out)
            return out
        else:  # Two arguments
            if b not in self.field:
                raise TypeError('second scalar {!r} not in the field {!r} of '
                                'the space {!r}.'.format(b, self.field, self))

            if x2 not in self:
                raise TypeError('second input vector {!r} not in space {!r}.'
                                ''.format(x2, self))

            # Call method
            self._lincomb(a, x1, b, x2, out)
            return out

    def dist(self, x1, x2):
        """Calculate the distance between two vectors.

        Parameters
        ----------
        x1 : `LinearSpaceVector`
            The first element

        x2 : `LinearSpaceVector`
            The second element

        Returns
        -------
        dist : `float`
               Distance between vectors
        """
        if x1 not in self:
            raise TypeError('first vector {!r} not in space {!r}'
                            ''.format(x1, self))
        if x2 not in self:
            raise TypeError('second vector {!r} not in space {!r}'
                            ''.format(x2, self))

        return float(self._dist(x1, x2))

    def norm(self, x):
        """Calculate the norm of a vector.

        Parameters
        ----------
        x : `LinearSpaceVector`
            The vector

        Returns
        -------
        out : `float`
            Norm of the vector
        """
        if x not in self:
            raise TypeError('vector {!r} not in space {!r}'.format(x, self))

        return float(self._norm(x))

    def inner(self, x1, x2):
        """Calculate the inner product of ``x1`` and ``x2``.

        Parameters
        ----------
        x1 : `LinearSpaceVector`
            The first vector

        x2 : `LinearSpaceVector`
            The second vector

        Returns
        -------
        out : `LinearSpace.field` element
            Product of the vectors, same as ``out`` if given.
        """
        if x1 not in self:
            raise TypeError('first vector {!r} not in space {!r}'
                            ''.format(x1, self))
        if x2 not in self:
            raise TypeError('second vector {!r} not in space {!r}'
                            ''.format(x2, self))

        return self.field.element(self._inner(x1, x2))

    def multiply(self, x1, x2, out=None):
        """Calculate the pointwise product of ``x1`` and ``x2``.

        Parameters
        ----------
        x1 : `LinearSpaceVector`
            The first multiplicand

        x2 : `LinearSpaceVector`
            The second multiplicand

        out : `LinearSpaceVector`, optional
            Vector to write the product to.
            default: `LinearSpace.element`

        Returns
        -------
        out : `LinearSpaceVector`
            Product of the vectors, same as ``out`` if given.
        """
        if out is None:
            out = self.element()
        elif out not in self:
            raise TypeError('out {!r} not in space {!r}'
                            ''.format(out, self))

        if x1 not in self:
            raise TypeError('x1 {!r} not in space {!r}'
                            ''.format(x1, self))
        if x2 not in self:
            raise TypeError('x2 {!r} not in space {!r}'
                            ''.format(x2, self))

        self._multiply(x1, x2, out)
        return out

    def divide(self, x1, x2, out=None):
        """Calculate the pointwise division of ``x1`` and ``x2``

        Parameters
        ----------
        x1 : `LinearSpaceVector`
            The dividend

        x2 : `LinearSpaceVector`
            The divisor

        out : `LinearSpaceVector`, optional
            Vector to write the ratio to.
            default: `LinearSpace.element`

        Returns
        -------
        out : `LinearSpaceVector`
            Ratio of the vectors, same as ``out`` if given.
        """
        if out is None:
            out = self.element()
        elif out not in self:
            raise TypeError('out {!r} not in space {!r}'
                            ''.format(out, self))

        if x1 not in self:
            raise TypeError('x1 {!r} not in space {!r}'
                            ''.format(x1, self))
        if x2 not in self:
            raise TypeError('x2 {!r} not in space {!r}'
                            ''.format(x2, self))

        self._divide(x1, x2, out)
        return out

    @property
    def element_type(self):
        """ `LinearSpaceVector` """
        return LinearSpaceVector


class LinearSpaceVector(object):
    """Abstract `LinearSpace` element.

    Not intended for creation of vectors, use the space's
    `LinearSpace.element` method instead.
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
        """Space to which this vector belongs.

        `LinearSpace`
        """
        return self._space

    # Convenience functions
    def assign(self, other):
        """Assign the values of ``other`` to self."""
        return self.space.lincomb(1, other, out=self)

    def copy(self):
        """Create an identical (deep) copy of self."""
        result = self.space.element()
        result.assign(self)
        return result

    def lincomb(self, a, x1, b=None, x2=None):
        """Assign a linear combination to this vector.

        Implemented as ``space.lincomb(a, x1, b, x2, out=self)``.

        `LinearSpace.lincomb`
        """
        return self.space.lincomb(a, x1, b, x2, out=self)

    def set_zero(self):
        """Set this vector to zero.

        `LinearSpace.zero`
        """
        return self.space.lincomb(0, self, 0, self, out=self)

    # Convenience operators
    def __iadd__(self, other):
        """Implement ``self += other``."""
        if other in self.space:
            return self.space.lincomb(1, self, 1, other, out=self)
        else:
            return NotImplemented

    def __add__(self, other):
        """Return ``self + other``."""
        # Instead of using __iadd__ we duplicate code here for performance
        if other in self.space:
            tmp = self.space.element()
            return self.space.lincomb(1, self, 1, other, out=tmp)
        else:
            return NotImplemented

    def __isub__(self, other):
        """Implement ``self -= other``."""
        if other in self.space:
            return self.space.lincomb(1, self, -1, other, out=self)
        else:
            return NotImplemented

    def __sub__(self, other):
        """Return ``self - other``."""
        # Instead of using __isub__ we duplicate code here for performance
        if other in self.space:
            tmp = self.space.element()
            return self.space.lincomb(1, self, -1, other, out=tmp)
        else:
            return NotImplemented

    def __imul__(self, other):
        """Implement ``self *= other``."""
        if other in self.space.field:
            return self.space.lincomb(other, self, out=self)
        elif other in self.space:
            return self.space.multiply(other, self, out=self)
        else:
            return NotImplemented

    def __mul__(self, other):
        """Return ``self * other``."""
        # Instead of using __imul__ we duplicate code here for performance
        if other in self.space.field:
            tmp = self.space.element()
            return self.space.lincomb(other, self, out=tmp)
        elif other in self.space:
            tmp = self.space.element()
            return self.space.multiply(other, self, out=tmp)
        else:
            return NotImplemented

    def __rmul__(self, other):
        """Return ``other * self``."""
        return self.__mul__(other)

    def __itruediv__(self, other):
        """Implement ``self /= other`` (true division)."""
        if other in self.space.field:
            return self.space.lincomb(1.0 / other, self, out=self)
        elif other in self.space:
            return self.space.divide(self, other, out=self)
        else:
            return NotImplemented

    __idiv__ = __itruediv__

    def __truediv__(self, other):
        """Return ``self / other``."""
        if other in self.space.field:
            tmp = self.space.element()
            return self.space.lincomb(1.0 / other, self, out=tmp)
        elif other in self.space:
            tmp = self.space.element()
            return self.space.divide(self, other, out=tmp)
        else:
            return NotImplemented

    __div__ = __truediv__

    def __ipow__(self, n):
        """``n``-th power in-place.

        This is only defined for integer ``n``."""
        if n == 1:
            return self
        elif n % 2 == 0:
            self.space.multiply(self, self, out=self)
            return self.__ipow__(n // 2)
        else:
            tmp = self.copy()
            for i in range(n - 1):
                self.space.multiply(tmp, self, out=tmp)
            return tmp

    def __pow__(self, n):
        """``n``-th power.

        This is only defined for integer ``n``."""
        tmp = self.copy()
        tmp **= n
        return tmp

    def __neg__(self):
        """Implementation of ``-self``."""
        return (-1) * self

    def __pos__(self):
        """Implementation of ``+self``."""
        return self.copy()

    # Metric space method
    def __eq__(self, other):
        """Return ``self == other``.

        Two vectors are equal if their distance is 0

        Parameters
        ----------
        other : `LinearSpaceVector`
            Vector in this space.

        Returns
        -------
        equals : `bool`
            True if the vectors are equal, else false.

        Notes
        -----
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
        if (not isinstance(other, LinearSpaceVector) or
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
        """Return ``self != other``."""
        return not self.__eq__(other)

    def __str__(self):
        """Return ``str(self)``.

        This is a default implementation, only returning the space.
        """
        return str(self.space) + "Vector"

    def __repr__(self):
        """Return ``repr(self)``.

        This is a default implementation, only returning the space.
        """
        return repr(self.space) + "Vector"

    def __copy__(self):
        """Copy of vector

        `LinearSpaceVector.copy`
        """
        return self.copy()

    def __deepcopy__(self, memo):
        """Copy of vector

        `LinearSpaceVector.copy`
        """
        return self.copy()

    # TODO: DOCUMENT
    def norm(self):
        """Norm of vector

        `LinearSpace.norm`
        """
        return self.space.norm(self)

    def dist(self, other):
        """Distance to ``other``.

        `LinearSpace.dist`
        """
        return self.space.dist(self, other)

    def inner(self, other):
        """Inner product with ``other``.

        `LinearSpace.inner`
        """
        return self.space.inner(self, other)

    def multiply(self, x, y):
        """Multiply by ``other`` inplace.

        `LinearSpace.multiply`
        """
        return self.space.multiply(x, y, out=self)

    @property
    def T(self):
        """The transpose of a vector, the functional given by (. , self)

        Returns
        -------
        transpose : `InnerProductOperator`

        Notes
        -----
        This function is only defined in inner product spaces.

        In a complex space, this takes the conjugate transpose of
        the vector.

        Examples
        --------
        >>> from odl import Rn
        >>> import numpy as np
        >>> rn = Rn(3)
        >>> x = rn.element([1, 2, 3])
        >>> y = rn.element([2, 1, 3])
        >>> x.T(y)
        13.0
        """
        from odl.operator.default_ops import InnerProductOperator
        return InnerProductOperator(self.copy())


class UniversalSpace(LinearSpace):
    """A dummy linear space class mostly raising `NotImplementedError`."""

    def __init__(self):
        """Initialize a universal space"""
        LinearSpace.__init__(self, field=UniversalSet())

    def element(self, inp=None):
        """Dummy element creation method.

        raises `NotImplementedError`.
        """
        raise NotImplementedError

    def _lincomb(self, a, x1, b, x2, out):
        """Dummy linear combination.

        raises `NotImplementedError`.
        """
        raise NotImplementedError

    def _dist(self, x1, x2):
        """Dummy distance method.

        raises `NotImplementedError`.
        """
        raise NotImplementedError

    def _norm(self, x):
        """Dummy norm method.

        raises `NotImplementedError`.
        """
        raise NotImplementedError

    def _inner(self, x1, x2):
        """Dummy inner product method.

        raises `NotImplementedError`.
        """
        raise NotImplementedError

    def _multiply(self, x1, x2, out):
        """Dummy multiplication method.

        raises `NotImplementedError`."""
        raise NotImplementedError

    def _divide(self, x1, x2, out):
        """Dummy division method.

        raises `NotImplementedError`.
        """
        raise NotImplementedError

    def __eq__(self, other):
        """Return ``self == other``.

        Dummy check, `True` for any `LinearSpace`.
        """
        return isinstance(other, LinearSpace)

    def __contains__(self, other):
        """Return ``other in self``.

        Dummy membership check, `True` for any `LinearSpaceVector`.
        """
        return isinstance(other, LinearSpaceVector)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
