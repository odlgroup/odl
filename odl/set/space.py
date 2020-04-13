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

from odl.set.sets import Field, Set


__all__ = ('LinearSpace',)


class LinearSpace(Set):

    """Abstract linear vector space."""

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
        raise NotImplementedError(
            'inner product not implemented in space {!r}'.format(self))

    def _multiply(self, x1, x2, out):
        """Implement the pointwise multiplication ``out[:] = x1 * x2``.

        This method is intended to be private. Public callers should
        resort to `multiply` which is type-checked.
        """
        raise NotImplementedError(
            'multiplication not implemented in space {!r}'.format(self))

    def one(self):
        """Return the one (multiplicative unit) element of this space."""
        raise NotImplementedError(
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

        Raises
        ------
        TypeError
            If ``out`` is given but not an element of this space.

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
            raise TypeError(
                '`out` {!r} is not an element of {!r}'.format(out, self)
            )

        if self.field is not None:
            a = self.field.element(a)

        x1 = self.element(x1)

        if b is None:
            if x2 is not None:
                raise ValueError('`x2` provided but not `b`')
            self._lincomb(a, x1, 0, x1, out)
            return out

        if self.field is not None:
            b = self.field.element(b)

        x2 = self.element(x2)
        self._lincomb(a, x1, b, x2, out)
        return out

    def assign(self, out, x):
        """Assign ``x`` to ``out``."""
        self.lincomb(1, x, out=out)

    def copy(self, x):
        """Return a copy of ``x``.

        This default implementation is intended to work with any space that
        supports `lincomb`. Subclasses may choose to implement an optimized
        variant.
        """
        out = self.element()
        self.assign(out, x)
        return out

    def set_zero(self, out):
        """Set ``out`` to zero.

        This default implementation should be overridden for spaces where
        elements can have nonsensical entries, where multiplication with
        0 does not yield the desired result.
        """
        self.lincomb(0, out, out=out)

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
        x1 = self.element(x1)
        x2 = self.element(x2)
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
        x = self.element(x)
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
        x1 = self.element(x1)
        x2 = self.element(x2)
        inner = self._inner(x1, x2)
        if self.field is None:
            return inner
        else:
            return self.field.element(inner)

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
            raise TypeError(
                '`out` {!r} is not an element of {!r}'.format(out, self)
            )

        if np.isscalar(x1):
            if self.field is not None:
                x1 = self.field.element(x1)
        else:
            x1 = self.element(x1)

        if np.isscalar(x2):
            if self.field is not None:
                x2 = self.field.element(x2)
        else:
            x2 = self.element(x2)

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
            raise TypeError(
                '`out` {!r} is not an element of {!r}'.format(out, self)
            )

        if np.isscalar(x1):
            if self.field is not None:
                x1 = self.field.element(x1)
        else:
            x1 = self.element(x1)

        if np.isscalar(x2):
            if self.field is not None:
                x2 = self.field.element(x2)
        else:
            x2 = self.element(x2)

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
            raise TypeError(
                '`other` must be a `LinearSpace`, got {!r}'.format(other)
            )

        return ProductSpace(self, other)

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
