# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Abstract linear vector spaces."""

from __future__ import print_function, division, absolute_import
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
        field : `Field` or None
            Scalar field of numbers for this space.
        """
        if field is None or isinstance(field, Field):
            self.__field = field
        else:
            raise TypeError('`field` must be a `Field` instance or `None`, '
                            'got {!r}'.format(field))

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

        or, if ``b`` and ``x2`` are given,

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

            ``space.lincomb(2, x, 3.14, x, out=x)``

        is (mathematically) equivalent to

            ``x = x * (2 + 3.14)``.
        """
        if out is None:
            out = self.element()
        elif out not in self:
            raise LinearSpaceTypeError('`out` {!r} is not an element of {!r}'
                                       ''.format(out, self))
        if self.field is not None and a not in self.field:
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
            if self.field is not None and b not in self.field:
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
        if x1 not in self:
            raise LinearSpaceTypeError('`x1` {!r} is not an element of '
                                       '{!r}'.format(x1, self))
        if x2 not in self:
            raise LinearSpaceTypeError('`x2` {!r} is not an element of '
                                       '{!r}'.format(x2, self))
        inner = self._inner(x1, x2)
        if self.field is None:
            return inner
        else:
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
        elif out not in self:
            raise LinearSpaceTypeError(
                '`out` {!r} is not an element of {!r}'.format(out, self)
            )

        field = () if self.field is None else self.field
        if not (x1 in field or x1 in self):
            raise LinearSpaceTypeError(
                '`x1` {!r} is not an element of {!r} or its field'
                ''.format(out, self)
            )
        if not (x2 in field or x2 in self):
            raise LinearSpaceTypeError(
                '`x2` {!r} is not an element of {!r} or its field'
                ''.format(out, self)
            )

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
        elif out not in self:
            raise LinearSpaceTypeError(
                '`out` {!r} is not an element of {!r}'.format(out, self)
            )

        field = () if self.field is None else self.field
        if not (x1 in field or x1 in self):
            raise LinearSpaceTypeError(
                '`x1` {!r} is not an element of {!r} or its field'
                ''.format(out, self)
            )
        if not (x2 in field or x2 in self):
            raise LinearSpaceTypeError(
                '`x2` {!r} is not an element of {!r} or its field'
                ''.format(out, self)
            )

        self._divide(x1, x2, out)
        return out

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

        Multiple powers work "outside-in":

        >>> r2 ** (4, 2)
        ProductSpace(ProductSpace(rn(2), 2), 4)
        """
        from odl.space import ProductSpace

        try:
            shape = (int(shape),)
        except TypeError:
            shape = tuple(shape)

        pspace = self
        for n in reversed(shape):
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
        """Return ``other in self``."""
        return NotImplementedError('abstract method')


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
    from odl.util.testutils import run_doctests
    run_doctests()
