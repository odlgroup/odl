﻿# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Abstract linear vector spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from builtins import object, range

import numpy as np

from odl.set.sets import Field, Set, UniversalSet


__all__ = ('LinearSpace', 'UniversalSpace')


class LinearSpace(Set):
    """Abstract linear vector space.

    Its elements are represented as instances of the
    `LinearSpaceElement` class.
    """

    def __init__(self, field):
        """Initialize a new instance.

        This method should be called by all inheriting methods so that
        the `field` property of the space is properly set.

        Parameters
        ----------
        field : `Field`
            Scalar field of numbers for this space.
        """
        if field is None or isinstance(field, Field):
            self.__field = field
        else:
            raise TypeError('`field` must be a `Field` instance, got {!r}'
                            ''.format(field))

    @property
    def field(self):
        """Scalar field of numbers for this vector space."""
        return self.__field

    def element(self, inp=None, **kwargs):
        """Create a `LinearSpaceElement` from ``inp`` or from scratch.

        If called without ``inp`` argument, an arbitrary element of the
        space is generated without guarantee of its state.

        If ``inp in self``, this has return ``inp`` or a view of ``inp``,
        otherwise, a copy may or may not occur.

        Parameters
        ----------
        inp : optional
            Input data from which to create the element.
        kwargs :
            Optional further arguments.

        Returns
        -------
        element : `LinearSpaceElement`
            A new element of this space.
        """
        raise NotImplementedError('abstract method')

    @property
    def examples(self):
        """Example elements `zero` and `one` (if available)."""
        # All spaces should yield the zero element
        yield ('Zero', self.zero())

        try:
            yield ('One', self.one())
        except NotImplementedError:
            pass

    def _lincomb(self, a, x1, b, x2, out):
        """Implement ``out[:] = a * x1 + b * x2``.

        This method is intended to be private. Public callers should
        resort to `lincomb` which is type-checked.
        """
        raise NotImplementedError('abstract method')

    def _dist(self, x1, x2):
        """Return the distance between ``x1`` and ``x2``.

        This method is intended to be private. Public callers should
        resort to `dist` which is type-checked.
        """
        return self.norm(x1 - x2)

    def _norm(self, x):
        """Return the norm of ``x``.

        This method is intended to be private. Public callers should
        resort to `norm` which is type-checked.
        """
        return float(np.sqrt(self.inner(x, x).real))

    def _inner(self, x1, x2):
        """Return the inner product of ``x1`` and ``x2``.

        This method is intended to be private. Public callers should
        resort to `inner` which is type-checked.
        """
        raise LinearSpaceNotImplementedError(
            'inner product not implemented in space {!r}'.format(self))

    def _multiply(self, x1, x2, out):
        """Implement the pointwise multiplication ``out[:] = x1 * x2``.

        This method is intended to be private. Public callers should
        resort to `multiply` which is type-checked.
        """
        raise LinearSpaceNotImplementedError(
            'multiplication not implemented in space {!r}'.format(self))

    def one(self):
        """Return the one (multiplicative unit) element of this space."""
        raise LinearSpaceNotImplementedError(
            '`one` element not implemented in space {!r}'.format(self))

    # Default methods
    def zero(self):
        """Return the zero (additive unit) element of this space."""
        tmp = self.element()
        self.lincomb(0, tmp, 0, tmp, tmp)
        return tmp

    def __contains__(self, other):
        """Return ``other in self``.

        Returns
        -------
        contains : bool
            ``True`` if ``other`` is a `LinearSpaceElement` instance and
            ``other.space`` is equal to this space, ``False`` otherwise.

        Notes
        -----
        This is the strict default where spaces must be equal.
        Subclasses may choose to implement a less strict check.
        """
        return getattr(other, 'space', None) == self

    # Error checking variant of methods
    def lincomb(self, a, x1, b=None, x2=None, out=None):
        """Implement ``out[:] = a * x1 + b * x2``.

        This function implements

            ``out[:] = a * x1``

        or, if ``b`` and ``y`` are given,

            ``out = a * x1 + b * x2``.

        Parameters
        ----------
        a : `field` element
            Scalar to multiply ``x1`` with.
        x1 : `LinearSpaceElement`
            First space element in the linear combination.
        b : `field` element, optional
            Scalar to multiply ``x2`` with. Required if ``x2`` is
            provided.
        x2 : `LinearSpaceElement`, optional
            Second space element in the linear combination.
        out : `LinearSpaceElement`, optional
            Element to which the result is written.

        Returns
        -------
        out : `LinearSpaceElement`
            Result of the linear combination. If ``out`` was provided,
            the returned object is a reference to it.

        Notes
        -----
        The elements ``out``, ``x1`` and ``x2`` may be aligned, thus a call

            ``space.lincomb(x, 2, x, 3.14, out=x)``

        is (mathematically) equivalent to

            ``x = x * (1 + 2 + 3.14)``.
        """
        if self.field is None:
            raise TypeError('`lincomb` cannot be used with `field=None`')

        if out is None:
            out = self.element()
        elif out not in self:
            raise LinearSpaceTypeError('`out` {!r} is not an element of {!r}'
                                       ''.format(out, self))
        if a not in self.field:
            raise LinearSpaceTypeError('`a` {!r} not an element of the field '
                                       '{!r} of {!r}'
                                       ''.format(a, self.field, self))
        if x1 not in self:
            raise LinearSpaceTypeError('`x1` {!r} is not an element of {!r}'
                                       ''.format(x1, self))

        if b is None:  # Single element
            if x2 is not None:
                raise ValueError('`x2` provided but not `b`')
            self._lincomb(a, x1, 0, x1, out)
            return out

        else:  # Two elements
            if b not in self.field:
                raise LinearSpaceTypeError('`b` {!r} not an element of the '
                                           'field {!r} of {!r}'
                                           ''.format(b, self.field, self))
            if x2 not in self:
                raise LinearSpaceTypeError('`x2` {!r} is not an element of '
                                           '{!r}'.format(x2, self))

            self._lincomb(a, x1, b, x2, out)

        return out

    def dist(self, x1, x2):
        """Return the distance between ``x1`` and ``x2``.

        Parameters
        ----------
        x1, x2 : `LinearSpaceElement`
            Elements whose distance to compute.

        Returns
        -------
        dist : float
            Distance between ``x1`` and ``x2``.
        """
        if x1 not in self:
            raise LinearSpaceTypeError('`x1` {!r} is not an element of '
                                       '{!r}'.format(x1, self))
        if x2 not in self:
            raise LinearSpaceTypeError('`x2` {!r} is not an element of '
                                       '{!r}'.format(x2, self))
        return float(self._dist(x1, x2))

    def norm(self, x):
        """Return the norm of ``x``.

        Parameters
        ----------
        x : `LinearSpaceElement`
            Element whose norm to compute.

        Returns
        -------
        norm : float
            Norm of ``x``.
        """
        if x not in self:
            raise LinearSpaceTypeError('`x` {!r} is not an element of '
                                       '{!r}'.format(x, self))
        return float(self._norm(x))

    def inner(self, x1, x2):
        """Return the inner product of ``x1`` and ``x2``.

        Parameters
        ----------
        x1, x2 : `LinearSpaceElement`
            Elements whose inner product to compute.

        Returns
        -------
        inner : `LinearSpace.field` element
            Inner product of ``x1`` and ``x2``.
        """
        if self.field is None:
            raise TypeError('`inner` cannot be used with `field=None`')
        if x1 not in self:
            raise LinearSpaceTypeError('`x1` {!r} is not an element of '
                                       '{!r}'.format(x1, self))
        if x2 not in self:
            raise LinearSpaceTypeError('`x2` {!r} is not an element of '
                                       '{!r}'.format(x2, self))
        return self.field.element(self._inner(x1, x2))

    def multiply(self, x1, x2, out=None):
        """Return the pointwise product of ``x1`` and ``x2``.

        Parameters
        ----------
        x1, x2 : `LinearSpaceElement`
            Multiplicands in the product.
        out : `LinearSpaceElement`, optional
            Element to which the result is written.

        Returns
        -------
        out : `LinearSpaceElement`
            Product of the elements. If ``out`` was provided, the
            returned object is a reference to it.
        """
        if out is None:
            out = self.element()

        if out not in self:
            raise LinearSpaceTypeError('`out` {!r} is not an element of '
                                       '{!r}'.format(out, self))
        if x1 not in self:
            raise LinearSpaceTypeError('`x1` {!r} is not an element of '
                                       '{!r}'.format(x1, self))
        if x2 not in self:
            raise LinearSpaceTypeError('`x2` {!r} is not an element of '
                                       '{!r}'.format(x2, self))
        self._multiply(x1, x2, out)
        return out

    def divide(self, x1, x2, out=None):
        """Return the pointwise quotient of ``x1`` and ``x2``

        Parameters
        ----------
        x1 : `LinearSpaceElement`
            Dividend in the quotient.
        x2 : `LinearSpaceElement`
            Divisor in the quotient.
        out : `LinearSpaceElement`, optional
            Element to which the result is written.

        Returns
        -------
        out : `LinearSpaceElement`
            Quotient of the elements. If ``out`` was provided, the
            returned object is a reference to it.
        """
        if out is None:
            out = self.element()

        if out not in self:
            raise LinearSpaceTypeError('`out` {!r} is not an element of '
                                       '{!r}'.format(out, self))
        if x1 not in self:
            raise LinearSpaceTypeError('`x1` {!r} is not an element of '
                                       '{!r}'.format(x1, self))
        if x2 not in self:
            raise LinearSpaceTypeError('`x2` {!r} is not an element of '
                                       '{!r}'.format(x2, self))
        self._divide(x1, x2, out)
        return out

    @property
    def element_type(self):
        """Type of elements of this space (`LinearSpaceElement`)."""
        return LinearSpaceElement

    def __pow__(self, shape):
        """Return ``self ** shape``.

        Notes
        -----
        This can be overridden by subclasses in order to give better memory
        coherence or otherwise a better interface.

        Examples
        --------
        Create simple power space:

        >>> r2 = odl.rn(2)
        >>> r2 ** 4
        ProductSpace(rn(2), 4)

        Multiple powers work as expected:

        >>> r2 ** (4, 2)
        ProductSpace(ProductSpace(rn(2), 4), 2)
        """
        from odl.space import ProductSpace

        try:
            shape = (int(shape),)
        except TypeError:
            shape = tuple(shape)

        pspace = self
        for n in shape:
            pspace = ProductSpace(pspace, n)

        return pspace

    def __mul__(self, other):
        """Return ``self * other``.

        Notes
        -----
        This can be overridden by subclasses in order to give better memory
        coherence or otherwise a better interface.

        Examples
        --------
        Create simple product space:

        >>> r2 = odl.rn(2)
        >>> r3 = odl.rn(3)
        >>> r2 * r3
        ProductSpace(rn(2), rn(3))
        """
        from odl.space import ProductSpace

        if not isinstance(other, LinearSpace):
            raise TypeError('Can only multiply with `LinearSpace`, got {!r}'
                            ''.format(other))

        return ProductSpace(self, other)

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


class LinearSpaceElement(object):

    """Abstract class for `LinearSpace` elements.

    Do not use this class directly -- to create an element of a vector
    space, call the space's `LinearSpace.element` method instead.
    """

    def __init__(self, space):
        """Initialize a new instance.

        All deriving classes must call this method to set the `space`
        property.
        """
        self.__space = space

    @property
    def space(self):
        """Space to which this element belongs."""
        return self.__space

    # Convenience functions
    def assign(self, other):
        """Assign the values of ``other`` to ``self``."""
        return self.space.lincomb(1, other, out=self)

    def copy(self):
        """Create an identical (deep) copy of self."""
        result = self.space.element()
        result.assign(self)
        return result

    def lincomb(self, a, x1, b=None, x2=None):
        """Implement ``self[:] = a * x1 + b * x2``.

        Parameters
        ----------
        a : element of ``space.field``
            Scalar to multiply ``x1`` with.
        x1 : `LinearSpaceElement`
            First space element in the linear combination.
        b : element of ``space.field``, optional
            Scalar to multiply ``x2`` with. Required if ``x2`` is
            provided.
        x2 : `LinearSpaceElement`, optional
            Second space element in the linear combination.

        See Also
        --------
        LinearSpace.lincomb
        """
        return self.space.lincomb(a, x1, b, x2, out=self)

    def set_zero(self):
        """Set this element to zero.

        See Also
        --------
        LinearSpace.zero
        """
        return self.space.lincomb(0, self, 0, self, out=self)

    # Convenience methods
    def __iadd__(self, other):
        """Implement ``self += other``."""
        if self.space.field is None:
            return NotImplemented
        elif other in self.space:
            return self.space.lincomb(1, self, 1, other, out=self)
        elif isinstance(other, LinearSpaceElement):
            # We do not `return NotImplemented` here since we don't want a
            # fallback for in-place. Otherwise python attempts
            # `self = self + other` which does not modify self.
            raise TypeError('cannot add {!r} and {!r} in-place'
                            ''.format(self, other))
        elif other in self.space.field:
            one = getattr(self.space, 'one', None)
            if one is None:
                raise TypeError('cannot add {!r} and {!r} in-place'
                                ''.format(self, other))
            else:
                # other --> other * space.one()
                return self.space.lincomb(1, self, other, one(), out=self)
        else:
            try:
                other = self.space.element(other)
            except (TypeError, ValueError):
                raise TypeError('cannot add {!r} and {!r} in-place'
                                ''.format(self, other))
            else:
                return self.__iadd__(other)

    def __add__(self, other):
        """Return ``self + other``."""
        # Instead of using __iadd__ we duplicate code here for performance
        if getattr(other, '__array_priority__', 0) > self.__array_priority__:
            return other.__radd__(self)
        elif self.space.field is None:
            return NotImplemented
        elif other in self.space:
            tmp = self.space.element()
            return self.space.lincomb(1, self, 1, other, out=tmp)
        elif isinstance(other, LinearSpaceElement):
            return NotImplemented
        elif other in self.space.field:
            one = getattr(self.space, 'one', None)
            if one is None:
                return NotImplemented
            else:
                tmp = one()
                return self.space.lincomb(1, self, other, tmp, out=tmp)
        else:
            try:
                other = self.space.element(other)
            except (TypeError, ValueError):
                return NotImplemented
            else:
                return self.__add__(other)

    def __radd__(self, other):
        """Return ``other + self``."""
        if getattr(other, '__array_priority__', 0) > self.__array_priority__:
            return other.__add__(self)
        else:
            return self.__add__(other)

    def __isub__(self, other):
        """Implement ``self -= other``."""
        if self.space.field is None:
            return NotImplemented
        elif other in self.space:
            return self.space.lincomb(1, self, -1, other, out=self)
        elif isinstance(other, LinearSpaceElement):
            # We do not `return NotImplemented` here since we don't want a
            # fallback for in-place. Otherwise python attempts
            # `self = self - other` which does not modify self.
            raise TypeError('cannot subtract {!r} and {!r} in-place'
                            ''.format(self, other))
        elif self.space.field is None:
            return NotImplemented
        elif other in self.space.field:
            one = getattr(self.space, 'one', None)
            if one is None:
                raise TypeError('cannot subtract {!r} and {!r} in-place'
                                ''.format(self, other))
            else:
                return self.space.lincomb(1, self, -other, one(), out=self)
        else:
            try:
                other = self.space.element(other)
            except (TypeError, ValueError):
                raise TypeError('cannot subtract {!r} and {!r} in-place'
                                ''.format(self, other))
            else:
                return self.__isub__(other)

    def __sub__(self, other):
        """Return ``self - other``."""
        # Instead of using __isub__ we duplicate code here for performance
        if getattr(other, '__array_priority__', 0) > self.__array_priority__:
            return other.__rsub__(self)
        elif self.space.field is None:
            return NotImplemented
        elif other in self.space:
            tmp = self.space.element()
            return self.space.lincomb(1, self, -1, other, out=tmp)
        elif isinstance(other, LinearSpaceElement):
            return NotImplemented
        elif other in self.space.field:
            one = getattr(self.space, 'one', None)
            if one is None:
                return NotImplemented
            else:
                tmp = one()
                return self.space.lincomb(1, self, -other, tmp, out=tmp)
        else:
            try:
                other = self.space.element(other)
            except (TypeError, ValueError):
                return NotImplemented
            else:
                return self.__sub__(other)

    def __rsub__(self, other):
        """Return ``other - self``."""
        if getattr(other, '__array_priority__', 0) > self.__array_priority__:
            return other.__sub__(self)
        elif self.space.field is None:
            return NotImplemented
        elif other in self.space:
            tmp = self.space.element()
            return self.space.lincomb(1, other, -1, self, out=tmp)
        elif isinstance(other, LinearSpaceElement):
            return NotImplemented
        elif other in self.space.field:
            one = getattr(self.space, 'one', None)
            if one is None:
                return NotImplemented
            else:
                # other --> other * space.one()
                tmp = one()
                self.space.lincomb(other, tmp, out=tmp)
                return self.space.lincomb(1, tmp, -1, self, out=tmp)
        else:
            try:
                other = self.space.element(other)
            except (TypeError, ValueError):
                return NotImplemented
            else:
                return self.__rsub__(other)

    def __imul__(self, other):
        """Implement ``self *= other``."""
        if self.space.field is None:
            return NotImplemented
        elif other in self.space.field:
            return self.space.lincomb(other, self, out=self)
        elif other in self.space:
            return self.space.multiply(other, self, out=self)
        elif isinstance(other, LinearSpaceElement):
            # We do not `return NotImplemented` here since we don't want a
            # fallback for in-place. Otherwise python attempts
            # `self = self * other` which does not modify self.
            raise TypeError('cannot multiply {!r} and {!r} in-place'
                            ''.format(self, other))
        else:
            try:
                other = self.space.element(other)
            except (TypeError, ValueError):
                raise TypeError('cannot multiply {!r} and {!r} in-place'
                                ''.format(self, other))
            else:
                return self.__imul__(other)

    def __mul__(self, other):
        """Return ``self * other``."""
        # Instead of using __imul__ we duplicate code here for performance
        if getattr(other, '__array_priority__', 0) > self.__array_priority__:
            return other.__rmul__(self)
        elif self.space.field is None:
            return NotImplemented
        elif other in self.space.field:
            tmp = self.space.element()
            result = self.space.lincomb(other, self, out=tmp)
            return result
        elif other in self.space:
            tmp = self.space.element()
            return self.space.multiply(other, self, out=tmp)
        elif isinstance(other, LinearSpaceElement):
            return NotImplemented
        else:
            try:
                other = self.space.element(other)
            except (TypeError, ValueError):
                return NotImplemented
            else:
                return self.__mul__(other)

    def __rmul__(self, other):
        """Return ``other * self``."""
        if getattr(other, '__array_priority__', 0) > self.__array_priority__:
            return other.__mul__(self)
        else:
            return self.__mul__(other)

    def __itruediv__(self, other):
        """Implement ``self /= other``."""
        if self.space.field is None:
            return NotImplemented
        if other in self.space.field:
            return self.space.lincomb(1.0 / other, self, out=self)
        elif other in self.space:
            return self.space.divide(self, other, out=self)
        elif isinstance(other, LinearSpaceElement):
            # We do not `return NotImplemented` here since we don't want a
            # fallback for in-place. Otherwise python attempts
            # `self = self / other` which does not modify self.
            raise TypeError('cannot divide {!r} and {!r} in-place'
                            ''.format(self, other))
        else:
            try:
                other = self.space.element(other)
            except (TypeError, ValueError):
                raise TypeError('cannot divide {!r} and {!r} in-place'
                                ''.format(self, other))
            else:
                return self.__itruediv__(other)

    __idiv__ = __itruediv__

    def __truediv__(self, other):
        """Return ``self / other``."""
        if getattr(other, '__array_priority__', 0) > self.__array_priority__:
            return other.__rtruediv__(self)
        elif self.space.field is None:
            return NotImplemented
        elif other in self.space.field:
            tmp = self.space.element()
            return self.space.lincomb(1.0 / other, self, out=tmp)
        elif other in self.space:
            tmp = self.space.element()
            return self.space.divide(self, other, out=tmp)
        elif isinstance(other, LinearSpaceElement):
            return NotImplemented
        else:
            try:
                other = self.space.element(other)
            except (TypeError, ValueError):
                return NotImplemented
            else:
                return self.__truediv__(other)

    __div__ = __truediv__

    def __rtruediv__(self, other):
        """Return ``other / self``."""
        if getattr(other, '__array_priority__', 0) > self.__array_priority__:
            return other.__truediv__(self)
        elif self.space.field is None:
            return NotImplemented
        elif other in self.space.field:
            one = getattr(self.space, 'one', None)
            if one is None:
                return NotImplemented
            else:
                # other --> other * space.one()
                tmp = one()
                self.space.lincomb(other, tmp, out=tmp)
                return self.space.divide(tmp, self, out=tmp)
        elif other in self.space:
            tmp = self.space.element()
            return self.space.divide(other, self, out=tmp)
        elif isinstance(other, LinearSpaceElement):
            return NotImplemented
        else:
            try:
                other = self.space.element(other)
            except (TypeError, ValueError):
                return NotImplemented
            else:
                return self.__rtruediv__(other)

    __rdiv__ = __rtruediv__

    def __ipow__(self, p):
        """Implement ``self ** p``.

        This is only defined for integer ``p``."""
        if self.space.field is None:
            return NotImplemented
        p, p_in = int(p), p
        if p != p_in:
            raise ValueError('expected integer `p`, got {}'.format(p_in))
        if p < 0:
            self **= -p
            self.space.divide(self.space.one(), self, out=self)
            return self
        elif p == 0:
            self.assign(self.space.one())
            return self
        elif p == 1:
            return self
        elif p % 2 == 0:
            self *= self
            self **= p // 2
            return self
        else:
            tmp = self.copy()
            for _ in range(p - 2):
                tmp *= self
            self *= tmp
            return self

    def __pow__(self, p):
        """Return ``self ** p``."""
        if self.space.field is None:
            return NotImplemented
        tmp = self.copy()
        tmp.__ipow__(p)
        return tmp

    def __neg__(self):
        """Return ``-self``."""
        if self.space.field is None:
            return NotImplemented
        return (-1) * self

    def __pos__(self):
        """Return ``+self``."""
        return self.copy()

    def __cmp__(self, other):
        """Comparsion not implemented."""
        # Stops python 2 from allowing comparsion of arbitrary objects
        raise TypeError('unorderable types: {}, {}'
                        ''.format(self.__class__.__name__, type(other)))

    # Metric space method
    def __eq__(self, other):
        """Return ``self == other``.

        Two elements are equal if their distance is zero.

        Parameters
        ----------
        other : `LinearSpaceElement`
            Element of this space.

        Returns
        -------
        equals : bool
            ``True`` if the elements are equal ``False`` otherwise.

        See Also
        --------
        LinearSpace.dist

        Notes
        -----
        Equality is very sensitive to numerical errors, thus any
        arithmetic operations should be expected to break equality.

        Examples
        --------
        >>> rn = odl.rn(1, norm=np.linalg.norm)
        >>> x = rn.element([0.1])
        >>> x == x
        True
        >>> y = rn.element([0.1])
        >>> x == y
        True
        >>> z = rn.element([0.3])
        >>> x + x + x == z
        False
        """
        if other is self:
            # Optimization for a common case
            return True
        elif (not isinstance(other, LinearSpaceElement) or
              other.space != self.space):
            # Cannot use (if other not in self.space) since this is not
            # reflexive.
            return False
        else:
            return self.space.dist(self, other) == 0

    def __ne__(self, other):
        """Return ``self != other``."""
        return not self.__eq__(other)

    # Disable hash since vectors are mutable
    __hash__ = None

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)

    def __copy__(self):
        """Return a copy of this element.

        See Also
        --------
        LinearSpace.copy
        """
        return self.copy()

    def __deepcopy__(self, memo):
        """Return a deep copy of this element.

        See Also
        --------
        LinearSpace.copy
        """
        return self.copy()

    def norm(self):
        """Return the norm of this element.

        See Also
        --------
        LinearSpace.norm
        """
        return self.space.norm(self)

    def dist(self, other):
        """Return the distance of ``self`` to ``other``.

        See Also
        --------
        LinearSpace.dist
        """
        return self.space.dist(self, other)

    def inner(self, other):
        """Return the inner product of ``self`` and ``other``.

        See Also
        --------
        LinearSpace.inner
        """
        return self.space.inner(self, other)

    def multiply(self, other, out=None):
        """Return ``out = self * other``.

        If ``out`` is provided, the result is written to it.

        See Also
        --------
        LinearSpace.multiply
        """
        return self.space.multiply(self, other, out=out)

    def divide(self, other, out=None):
        """Return ``out = self / other``.

        If ``out`` is provided, the result is written to it.

        See Also
        --------
        LinearSpace.divide
        """
        return self.space.divide(self, other, out=out)

    @property
    def T(self):
        """This element's transpose, i.e. the functional ``<. , self>``.

        Returns
        -------
        transpose : `InnerProductOperator`

        Notes
        -----
        This function is only defined in inner product spaces.

        In a complex space, the conjugate transpose of is taken instead
        of the transpose only.

        Examples
        --------
        >>> rn = odl.rn(3)
        >>> x = rn.element([1, 2, 3])
        >>> y = rn.element([2, 1, 3])
        >>> x.T(y)
        13.0
        """
        from odl.operator import InnerProductOperator
        return InnerProductOperator(self.copy())

    # Give an `Element` a higher priority than any NumPy array type. This
    # forces the usage of `__op__` of `Element` if the other operand
    # is a NumPy object (applies also to scalars!).
    __array_priority__ = 1000000.0


class UniversalSpace(LinearSpace):

    """A dummy linear space class.

    Mostly raising `LinearSpaceNotImplementedError`.
    """

    def __init__(self):
        """Initialize a new instance."""
        super(UniversalSpace, self).__init__(field=UniversalSet())

    def element(self, inp=None):
        """Dummy element creation method.

        raises `LinearSpaceNotImplementedError`.
        """
        raise LinearSpaceNotImplementedError

    def _lincomb(self, a, x1, b, x2, out):
        """Dummy linear combination.

        raises `LinearSpaceNotImplementedError`.
        """
        raise LinearSpaceNotImplementedError

    def _dist(self, x1, x2):
        """Dummy distance method.

        raises `LinearSpaceNotImplementedError`.
        """
        raise LinearSpaceNotImplementedError

    def _norm(self, x):
        """Dummy norm method.

        raises `LinearSpaceNotImplementedError`.
        """
        raise LinearSpaceNotImplementedError

    def _inner(self, x1, x2):
        """Dummy inner product method.

        raises `LinearSpaceNotImplementedError`.
        """
        raise LinearSpaceNotImplementedError

    def _multiply(self, x1, x2, out):
        """Dummy multiplication method.

        raises `LinearSpaceNotImplementedError`."""
        raise LinearSpaceNotImplementedError

    def _divide(self, x1, x2, out):
        """Dummy division method.

        raises `LinearSpaceNotImplementedError`.
        """
        raise LinearSpaceNotImplementedError

    def __eq__(self, other):
        """Return ``self == other``.

        Dummy check, ``True`` for any `LinearSpace`.
        """
        return isinstance(other, LinearSpace)

    def __contains__(self, other):
        """Return ``other in self``.

        Dummy membership check, ``True`` for any `LinearSpaceElement`.
        """
        return isinstance(other, LinearSpaceElement)


class LinearSpaceTypeError(TypeError):
    """Exception for type errors in `LinearSpace`'s.

    This exception is raised when the wrong type of element is fed to
    `LinearSpace.lincomb` and related functions.
    """


class LinearSpaceNotImplementedError(NotImplementedError):
    """Exception for unimplemented functionality in `LinearSpace`'s.

    This exception is raised when a method is called in `LinearSpace`
    that has not been defined in a specific space.
    """


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
