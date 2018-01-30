# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Default operators defined on any (reasonable) space."""

from __future__ import absolute_import, division, print_function

from copy import copy

import numpy as np

from odl.operator.operator import Operator
from odl.set import Field, LinearSpace, RealNumbers
from odl.space import ProductSpace
from odl.util import (
    REPR_PRECISION, npy_printoptions, repr_string, signature_string_parts)

__all__ = ('ScalingOperator', 'ZeroOperator', 'IdentityOperator',
           'LinCombOperator', 'MultiplyOperator', 'PowerOperator',
           'InnerProductOperator', 'NormOperator', 'DistOperator',
           'ConstantOperator', 'RealPart', 'ImagPart', 'ComplexEmbedding',
           'ComplexModulus')


class ScalingOperator(Operator):

    """Operator of multiplication with a scalar.

    Implements::

        ScalingOperator(s)(x) == s * x
    """

    def __init__(self, domain, scalar, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace` or `Field`
            Set of elements on which this operator acts.
        scalar : ``domain.field`` element
            Fixed scaling factor of this operator.
        range : `LinearSpace` or `Field`, optional
            Set to which this operator maps.
            Default: ``domain``.

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> vec = r3.element([1, 2, 3])
        >>> out = r3.element()
        >>> op = odl.ScalingOperator(r3, 2.0)
        >>> op(vec, out)  # In-place, Returns out
        rn(3).element([ 2.,  4.,  6.])
        >>> out
        rn(3).element([ 2.,  4.,  6.])
        >>> op(vec)  # Out-of-place
        rn(3).element([ 2.,  4.,  6.])
        """
        if not isinstance(domain, (LinearSpace, Field)):
            raise TypeError('`domain` {!r} not a `LinearSpace` or `Field` '
                            'instance'.format(domain))

        if range is None:
            range = domain
        else:
            if not isinstance(range, (LinearSpace, Field)):
                raise TypeError('`range` {!r} not a `LinearSpace` or `Field` '
                                'instance'.format(range))

        super(ScalingOperator, self).__init__(domain, range, linear=True)
        self.__scalar = domain.field.element(scalar)

    @property
    def scalar(self):
        """Fixed scaling factor of this operator."""
        return self.__scalar

    def _call(self, x, out=None):
        """Scale ``x`` and write to ``out`` if given."""
        if out is None:
            out = self.scalar * x
        else:
            out.lincomb(self.scalar, x)
        return out

    @property
    def inverse(self):
        """Return the inverse operator.

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> op = odl.ScalingOperator(r3, 2.0)
        >>> inv = op.inverse
        >>> inv(op(x)) == x
        True
        >>> op(inv(x)) == x
        True
        """
        if self.scalar == 0.0:
            raise ZeroDivisionError('scaling operator not invertible for '
                                    'scalar==0')
        return ScalingOperator(self.range, 1.0 / self.scalar, self.domain)

    @property
    def adjoint(self):
        """Adjoint, given as scaling with the conjugate of the scalar.

        Returns
        -------
        adjoint : `ScalingOperator`
            ``self`` if `scalar` is real, else `scalar` is conjugated.
        """
        if complex(self.scalar).imag == 0.0:
            return ScalingOperator(self.range, self.scalar, self.domain)
        else:
            return ScalingOperator(self.range, self.scalar.conjugate(),
                                   self.domain)

    def norm(self, estimate=False, **kwargs):
        """Return the operator norm of this operator.

        Parameters
        ----------
        estimate, kwargs : bool
            Ignored. Present to conform with base-class interface.

        Returns
        -------
        norm : float
            The operator norm, absolute value of `scalar`.

        Examples
        --------
        >>> spc = odl.rn(3)
        >>> scaling = odl.ScalingOperator(spc, 3.0)
        >>> scaling.norm(estimate=True)
        3.0
        """
        return np.abs(self.scalar)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> op = odl.ScalingOperator(r3, 2.0)
        >>> op
        ScalingOperator(rn(3), scalar=2.0)
        """
        posargs = [self.domain]
        optargs = [('scalar', self.scalar, None),
                   ('range', self.range, self.domain)]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts)


class IdentityOperator(ScalingOperator):

    """Operator mapping each element to itself.

    Implements::

        IdentityOperator()(x) == x
    """

    def __init__(self, domain, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace` or `Field`
            Set of elements on which this operator acts.
        range : `LinearSpace` or `Field`, optional
            Set to which this operator maps.
            Default: ``domain``.
        """
        super(IdentityOperator, self).__init__(domain, 1, range)

    @property
    def adjoint(self):
        """Adjoint of the identity operator."""
        return IdentityOperator(self.range, self.domain)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> op = odl.IdentityOperator(r3)
        >>> op
        IdentityOperator(rn(3))
        """
        posargs = [self.domain]
        optargs = [('range', self.range, self.domain)]
        inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts)


class LinCombOperator(Operator):

    """Operator mapping two space elements to a linear combination.

    Implements::

        LinCombOperator(a, b)([x, y]) == a * x + b * y
    """

    def __init__(self, space, a, b):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            Space of elements which the operator is acting on.
        a, b : ``space.field`` elements
            Scalars to multiply ``x[0]`` and ``x[1]`` with, respectively.

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> r3xr3 = odl.ProductSpace(r3, r3)
        >>> xy = r3xr3.element([[1, 2, 3], [1, 2, 3]])
        >>> z = r3.element()
        >>> op = odl.LinCombOperator(r3, 1.0, 1.0)
        >>> op(xy, out=z)  # Returns z
        rn(3).element([ 2.,  4.,  6.])
        >>> z
        rn(3).element([ 2.,  4.,  6.])
        """
        domain = ProductSpace(space, space)
        super(LinCombOperator, self).__init__(domain, space, linear=True)
        self.a = a
        self.b = b

    def _call(self, x, out=None):
        """Linearly combine ``x`` and write to ``out`` if given."""
        if out is None:
            out = self.range.element()
        out.lincomb(self.a, x[0], self.b, x[1])
        return out

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r})'.format(self.__class__.__name__,
                                             self.range, self.a, self.b)

    def __str__(self):
        """Return ``str(self)``."""
        return "{}*x + {}*y".format(self.a, self.b)


class MultiplyOperator(Operator):

    """Operator multiplying by a fixed space or field element.

    Implements::

        MultiplyOperator(y)(x) == x * y

    Here, ``y`` is a `LinearSpaceElement` or `Field` element and
    ``x`` is a `LinearSpaceElement`.
    Hence, this operator can be defined either on a `LinearSpace` or on
    a `Field`. In the first case it is the pointwise multiplication,
    in the second the scalar multiplication.
    """

    def __init__(self, multiplicand, domain=None, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        multiplicand : `LinearSpaceElement` or ``domain`` `element-like`
            Vector or scalar with which should be multiplied. If ``domain``
            is provided, this parameter can be an `element-like` object
            for ``domain``. Otherwise it must be a `LinearSpaceElement`.
        domain : `LinearSpace` or `Field`, optional
            Set to which the operator can be applied. Mandatory if
            ``multiplicand`` is not a `LinearSpaceElement`.
            Default: ``multiplicand.space``.
        range : `LinearSpace` or `Field`, optional
            Set to which the operator maps.
            Default: ``domain`` if given, otherwise ``multiplicand.space``.

        Examples
        --------
        If a `LinearSpaceElement` is provided, domain and range are
        inferred:

        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> op = odl.MultiplyOperator(x)
        >>> op([2, 4, 6])
        rn(3).element([  2.,   8.,  18.])
        >>> out = r3.element()
        >>> op(x, out)
        rn(3).element([ 1.,  4.,  9.])

        For a scalar or `element-like` multiplicand, ``domain`` (and
        ``range``) should be given:

        >>> op = odl.MultiplyOperator(x, domain=r3.field, range=r3)
        >>> op(3)
        rn(3).element([ 3.,  6.,  9.])
        >>> out = r3.element()
        >>> op(3, out)
        rn(3).element([ 3.,  6.,  9.])
        """
        if domain is None:
            domain = multiplicand.space

        if range is None:
            range = domain

        super(MultiplyOperator, self).__init__(domain, range, linear=True)

        self.__multiplicand = multiplicand
        self.__domain_is_field = isinstance(self.domain, Field)
        self.__range_is_field = isinstance(self.range, Field)

    @property
    def multiplicand(self):
        """Value to multiply by."""
        return self.__multiplicand

    def _call(self, x, out=None):
        """Multiply ``x`` and write to ``out`` if given."""
        if out is None:
            return x * self.multiplicand
        elif not self.__range_is_field:
            if self.__domain_is_field:
                out.lincomb(x, self.multiplicand)
            else:
                out.assign(self.multiplicand * x)
        else:
            raise ValueError('can only use `out` with `LinearSpace` range')

    @property
    def adjoint(self):
        """Adjoint of this operator.

        Returns
        -------
        adjoint : `InnerProductOperator` or `MultiplyOperator`
            If the domain of this operator is the scalar field of a
            `LinearSpace` the adjoint is the inner product with ``y``,
            else it is the multiplication with ``y``.

        Examples
        --------
        Multiply by a space element:

        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> op = odl.MultiplyOperator(x)
        >>> op.adjoint(x)
        rn(3).element([ 1.,  4.,  9.])

        Multiply scalars with a fixed vector:

        >>> op = odl.MultiplyOperator(x, domain=r3.field, range=r3)
        >>> op.adjoint(x)
        14.0

        Multiply vectors with a fixed scalar:

        >>> op = odl.MultiplyOperator(3.0, domain=r3, range=r3)
        >>> op.adjoint(x)
        rn(3).element([ 3.,  6.,  9.])
        """
        if self.__domain_is_field:
            return InnerProductOperator(self.multiplicand, self.range,
                                        self.domain)
        else:
            # TODO: complex case
            return MultiplyOperator(self.multiplicand, self.range, self.domain)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> op = odl.MultiplyOperator(x)
        >>> op
        MultiplyOperator(rn(3).element([ 1.,  2.,  3.]))
        """
        posargs = [self.multiplicand]
        optargs = [('domain', self.domain,
                    getattr(self.multiplicand, 'space', None)),
                   ('range', self.range, self.domain)]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts)


class PowerOperator(Operator):

    """Operator taking a fixed power of a space or field element.

    Implements::

        PowerOperator(p)(x) == x ** p

    Here, ``x`` is a `LinearSpaceElement` or `Field` element and ``p`` is
    a number. Hence, this operator can be defined either on a
    `LinearSpace` or on a `Field`.
    """

    def __init__(self, domain, exponent, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace` or `Field`
            Set of elements on which the operator can be applied.
        exponent : float
            Exponent parameter of the power function applied to an element.
        range : `LinearSpace` or `Field`, optional
            Set to which this operator maps.
            Default: ``domain``.

        Examples
        --------
        Usage on a space of vectors:

        >>> op = odl.PowerOperator(odl.rn(3), exponent=2)
        >>> op([1, 2, 3])
        rn(3).element([ 1.,  4.,  9.])

        For a scalar space:

        >>> op = odl.PowerOperator(odl.RealNumbers(), exponent=2)
        >>> op(2.0)
        4.0
        """
        if range is None:
            range = domain
        super(PowerOperator, self).__init__(
            domain, range, linear=(exponent == 1))
        self.__exponent = float(exponent)
        self.__range_is_field = isinstance(range, Field)

    @property
    def exponent(self):
        """Power of the input element to take."""
        return self.__exponent

    def _call(self, x, out=None):
        """Take the power of ``x`` and write to ``out`` if given."""
        if out is None:
            return x ** self.exponent
        elif self.__range_is_field:
            raise ValueError('cannot use `out` with field')
        else:
            out.assign(x)
            out **= self.exponent

    def derivative(self, point):
        """Derivative of this operator.

            ``PowerOperator(p).derivative(y)(x) == p * y ** (p - 1) * x``

        Parameters
        ----------
        point : `domain` element
            The point in which to take the derivative

        Returns
        -------
        derivative : `Operator`
            The derivative in ``point``

        Examples
        --------
        Use on vector spaces:

        >>> op = odl.PowerOperator(odl.rn(3), exponent=2)
        >>> dop = op.derivative(op.domain.element([1, 2, 3]))
        >>> dop([1, 1, 1])
        rn(3).element([ 2.,  4.,  6.])

        Use with scalars:

        >>> op = odl.PowerOperator(odl.RealNumbers(), exponent=2)
        >>> dop = op.derivative(2.0)
        >>> dop(2.0)
        8.0
        """
        return (self.exponent *
                MultiplyOperator(point ** (self.exponent - 1),
                                 domain=self.domain, range=self.range)
                )

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> op = odl.PowerOperator(odl.rn(3), exponent=2)
        >>> op
        PowerOperator(rn(3), exponent=2.0)
        """
        posargs = [self.domain]
        optargs = [('exponent', self.exponent, None),
                   ('range', self.range, self.domain)]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts)


class InnerProductOperator(Operator):
    """Operator taking the inner product with a fixed space element.

    Implements::

        InnerProductOperator(y)(x) <==> y.inner(x)

    This is only applicable in inner product spaces.

    See Also
    --------
    DistOperator : Distance to a fixed space element.
    NormOperator : Vector space norm as operator.
    """

    def __init__(self, vector, domain=None, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        vector : `LinearSpaceElement` or ``domain`` `element-like`
            The element with which the inner product is taken. If ``domain``
            is given, this can be an `element-like` object for ``domain``,
            otherwise it must be a `LinearSpaceElement`.
        domain : `LinearSpace` or `Field`, optional
            Set of elements on which the operator can be applied. Optional
            if ``vector`` is a `LinearSpaceElement`, in which case
            ``vector.space`` is taken as default. Otherwise this parameter
            is mandatory.
        range : `LinearSpace` or `Field`, optional
            Set to which this operator maps.
            Default: ``domain.field``.

        Examples
        --------
        By default, ``domain`` and ``range`` are inferred from the
        given ``vector``:

        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> op = odl.InnerProductOperator(x)
        >>> op.range
        RealNumbers()
        >>> op([1, 2, 3])
        14.0

        With an explicit domain, we do not need a `LinearSpaceElement`
        as ``vector``:

        >>> op = odl.InnerProductOperator([1, 2, 3], domain=r3)
        >>> op([1, 2, 3])
        14.0

        We can also specify an explicit range, which should be able to hold
        a single scalar:

        >>> r1 = odl.rn(1)
        >>> op = odl.InnerProductOperator([1, 2, 3], domain=r3, range=r1)
        >>> op([1, 2, 3])
        rn(1).element([ 14.])
        >>> r = odl.rn(())
        >>> op = odl.InnerProductOperator([1, 2, 3], domain=r3, range=r)
        >>> op([1, 2, 3])
        rn(()).element(14.0)
        """
        if domain is None:
            domain = vector.space
        if range is None:
            range = domain.field
        super(InnerProductOperator, self).__init__(domain, range, linear=True)
        self.__vector = self.domain.element(vector)

    @property
    def vector(self):
        """Element to take the inner product with."""
        return self.__vector

    def _call(self, x):
        """Return the inner product with ``x``."""
        return x.inner(self.vector)

    @property
    def adjoint(self):
        """Adjoint of this operator.

        Returns
        -------
        adjoint : `MultiplyOperator`
            The operator of multiplication with `vector`.

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> op = odl.InnerProductOperator(x)
        >>> op.adjoint(2.0)
        rn(3).element([ 2.,  4.,  6.])
        """
        return MultiplyOperator(self.vector, domain=self.range,
                                range=self.domain)

    @property
    def T(self):
        """Fixed vector of this operator.

        Returns
        -------
        vector : `LinearSpaceElement`
            The fixed space element used in this inner product operator.

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> x.T
        InnerProductOperator(rn(3).element([ 1.,  2.,  3.]))
        >>> x.T.T
        rn(3).element([ 1.,  2.,  3.])
        """
        return self.vector

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> op = odl.InnerProductOperator(x)
        >>> op
        InnerProductOperator(rn(3).element([ 1.,  2.,  3.]))
        """
        posargs = [self.vector]
        optargs = [('domain', self.domain,
                    getattr(self.vector, 'space', None)),
                   ('range', self.range, self.domain.field)]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts)


class NormOperator(Operator):

    """Vector space norm as an operator.

    Implements::

        NormOperator()(x) <==> x.norm()

    This is only applicable in normed spaces, i.e., spaces implementing
    a `LinearSpace.norm` method.

    See Also
    --------
    InnerProductOperator : Inner product with a fixed space element.
    DistOperator : Distance to a fixed space element.
    """

    def __init__(self, domain, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace`
            Set of elements on which the operator can be applied. Needs
            to implement ``space.norm``.
        range : `LinearSpace` or `Field`, optional
            Set to which this operator maps.
            Default: ``RealNumbers``.

        Examples
        --------
        >>> r2 = odl.rn(2)
        >>> op = odl.NormOperator(r2)
        >>> op([3, 4])
        5.0
        """
        if range is None:
            range = RealNumbers()
        super(NormOperator, self).__init__(domain, range, linear=False)

    def _call(self, x):
        """Return the norm of ``x``."""
        return x.norm()

    def derivative(self, point):
        r"""Derivative of this operator in ``point``.

            ``NormOperator().derivative(y)(x) == (y / y.norm()).inner(x)``

        This is only applicable in inner product spaces.

        Parameters
        ----------
        point : `domain` `element-like`
            Point in which to take the derivative.

        Returns
        -------
        derivative : `InnerProductOperator`

        Raises
        ------
        ValueError
            If ``point.norm() == 0``, in which case the derivative is not well
            defined in the Frechet sense.

        Notes
        -----
        The derivative cannot be written in a general sense except in Hilbert
        spaces, in which case it is given by

        .. math::
            (D \|\cdot\|)(y)(x) = \langle y / \|y\|, x \rangle

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> op = odl.NormOperator(r3)
        >>> derivative = op.derivative([1, 0, 0])
        >>> derivative([1, 0, 0])
        1.0
        """
        point = self.domain.element(point)
        norm = point.norm()
        if norm == 0:
            raise ValueError('not differentiable in 0')

        return InnerProductOperator(point / norm, self.domain, self.range)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> r2 = odl.rn(2)
        >>> op = odl.NormOperator(r2)
        >>> op
        NormOperator(rn(2))
        """
        posargs = [self.domain]
        optargs = [('range', self.range, RealNumbers())]
        inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts)


class DistOperator(Operator):

    """Operator taking the distance to a fixed space element.

    Implements::

        DistOperator(y)(x) == y.dist(x)

    This is only applicable in metric spaces, i.e., spaces implementing
    a `LinearSpace.dist` method.

    See Also
    --------
    InnerProductOperator : Inner product with fixed space element.
    NormOperator : Vector space norm as an operator.
    """

    def __init__(self, vector, domain=None, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        vector : `LinearSpaceElement` or ``domain`` `element-like`
            Point to which to calculate the distance. If ``domain`` is
            given, this can be `element-like` for ``domain``, otherwise
            it must be a `LinearSpaceElement`.
        domain : `LinearSpace`, optional
            Set of elements on which the operator can be applied. Needs
            to implement ``space.dist``. Optional if ``vector`` is a
            `LinearSpaceElement`, in which case ``vector.space`` is taken
            as default. Otherwise this parameter is mandatory.
        range : `LinearSpace` or `Field`, optional
            Set to which this operator maps.
            Default: ``RealNumbers``.

        Examples
        --------
        >>> r2 = odl.rn(2)
        >>> x = r2.element([1, 1])
        >>> op = odl.DistOperator(x)
        >>> op([4, 5])
        5.0
        """
        if domain is None:
            domain = vector.space
        if range is None:
            range = RealNumbers()
        super(DistOperator, self).__init__(domain, range, linear=False)
        self.__vector = self.domain.element(vector)

    @property
    def vector(self):
        """Element to which to take the distance."""
        return self.__vector

    def _call(self, x):
        """Return the distance from ``self.vector`` to ``x``."""
        return self.vector.dist(x)

    def derivative(self, point):
        r"""The derivative operator.

            ``DistOperator(y).derivative(z)(x) ==
            ((y - z) / y.dist(z)).inner(x)``

        This is only applicable in inner product spaces.

        Parameters
        ----------
        x : `domain` `element-like`
            Point in which to take the derivative.

        Returns
        -------
        derivative : `InnerProductOperator`

        Raises
        ------
        ValueError
            If ``point == self.vector``, in which case the derivative is not
            well defined in the Frechet sense.

        Notes
        -----
        The derivative cannot be written in a general sense except in Hilbert
        spaces, in which case it is given by

        .. math::
            (D d(\cdot, y))(z)(x) = \langle (y-z) / d(y, z), x \rangle

        Examples
        --------
        >>> r2 = odl.rn(2)
        >>> x = r2.element([1, 1])
        >>> op = odl.DistOperator(x)
        >>> derivative = op.derivative([2, 1])
        >>> derivative([1, 0])
        1.0
        """
        point = self.domain.element(point)
        diff = point - self.vector
        dist = self.vector.dist(point)
        if dist == 0:
            raise ValueError('not differentiable at the reference vector {!r}'
                             ''.format(self.vector))

        return InnerProductOperator(diff / dist, self.domain, self.range)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> r2 = odl.rn(2)
        >>> x = r2.element([1, 1])
        >>> op = odl.DistOperator(x)
        >>> op
        DistOperator(rn(2).element([ 1.,  1.]))
        """
        posargs = [self.vector]
        optargs = [('domain', self.domain,
                    getattr(self.vector, 'space', None)),
                   ('range', self.range, RealNumbers())]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts)


class ConstantOperator(Operator):

    """Operator that always returns the same value.

    Implements::

        ConstantOperator(y)(x) == y
    """

    def __init__(self, constant, domain=None, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        constant : `LinearSpaceElement` or ``range`` `element-like`
            Constant vector that should be returned. If ``domain`` is
            given, this can be `element-like` for ``domain``, otherwise
            it must be a `LinearSpaceElement`.
        domain : `LinearSpace`, optional
            Set of elements on which the operator can be applied.
            Default: ``range`` if provided, otherwise ``constant.space``.
        range : `LinearSpace` or `Field`, optional
            Set to which this operator maps. Optional if ``constant`` is a
            `LinearSpaceElement`, in which case ``constant.space`` is taken
            as default. Otherwise this parameter is mandatory.

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> op = odl.ConstantOperator(x)
        >>> op(x, out=r3.element())
        rn(3).element([ 1.,  2.,  3.])
        """
        if range is None:
            range = constant.space
        if domain is None:
            domain = range

        super(ConstantOperator, self).__init__(domain, range, linear=False)
        self.__constant = self.range.element(constant)

    @property
    def constant(self):
        """Constant space element returned by this operator."""
        return self.__constant

    def _call(self, x, out=None):
        """Return the constant vector or assign it to ``out``."""
        if out is None:
            return self.range.element(copy(self.constant))
        else:
            out.assign(self.constant)

    def derivative(self, point):
        """Derivative of this operator, always zero.

        Returns
        -------
        derivative : `ZeroOperator`

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> op = odl.ConstantOperator(x)
        >>> deriv = op.derivative([1, 1, 1])
        >>> deriv([2, 2, 2])
        rn(3).element([ 0.,  0.,  0.])
        """
        return ZeroOperator(self.domain, self.range)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> op = odl.ConstantOperator(x)
        >>> op
        ConstantOperator(rn(3).element([ 1.,  2.,  3.]))
        """
        posargs = [self.constant]
        optargs = [('domain', self.domain, self.range),
                   ('range', self.range,
                    getattr(self.constant, 'space', None))]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts)


class ZeroOperator(Operator):

    """Operator mapping each element to the zero element.

    Implements::

        ZeroOperator(space)(x) == space.zero()
    """

    def __init__(self, domain, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace`, optional
            Set of elements on which the operator can be applied.
        range : `LinearSpace` or `Field`, optional
            Set to which this operator maps.
            Default: ``domain``

        Examples
        --------
        >>> op = odl.ZeroOperator(odl.rn(3))
        >>> op([1, 2, 3])
        rn(3).element([ 0.,  0.,  0.])

        Also works with domain != range:

        >>> op = odl.ZeroOperator(odl.rn(3), odl.cn(4))
        >>> op([1, 2, 3])
        cn(4).element([ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j])
        """
        if range is None:
            range = domain

        super(ZeroOperator, self).__init__(domain, range, linear=True)

    def _call(self, x, out=None):
        """Return the zero vector or assign it to ``out``."""
        if self.domain == self.range:
            if out is None:
                out = 0 * x
            else:
                out.lincomb(0, x)
        else:
            result = self.range.zero()
            if out is None:
                out = result
            else:
                out.assign(result)
        return out

    @property
    def adjoint(self):
        """Adjoint of the operator.

        If ``self.domain == self.range`` the zero operator is self-adjoint,
        otherwise it is the `ZeroOperator` from `range` to `domain`.
        """
        return ZeroOperator(self.range, self.domain)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> op = odl.ZeroOperator(odl.rn(3))
        >>> op
        ZeroOperator(rn(3))
        """
        posargs = [self.domain]
        optargs = [('range', self.range, self.domain)]
        inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts)


class RealPart(Operator):

    """Operator that extracts the real part of a vector.

    Implements::

        RealPart(x) == x.real
    """

    def __init__(self, domain, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `TensorSpace` or `Field`
            Space in which the real part should be taken. Needs to
            implement ``domain.real_space`` and ``domain.real``.
        range : `TensorSpace` or `Field`, optional
            Set to which this operator maps.
            Default: ``domain.real_space``

        Examples
        --------
        Take the real part of complex vector:

        >>> c3 = odl.cn(3)
        >>> op = odl.RealPart(c3)
        >>> op([1 + 2j, 2, 3 - 1j])
        rn(3).element([ 1.,  2.,  3.])

        The operator is the identity on real spaces:

        >>> r3 = odl.rn(3)
        >>> op = odl.RealPart(r3)
        >>> op([1, 2, 3])
        rn(3).element([ 1.,  2.,  3.])

        The operator also works on other `TensorSpace` spaces such as
        `DiscreteLp` spaces:

        >>> r3 = odl.uniform_discr(0, 1, 3, dtype=complex)
        >>> op = odl.RealPart(r3)
        >>> op([1, 2, 3])
        uniform_discr(0.0, 1.0, 3).element([ 1.,  2.,  3.])
        """
        if range is None:
            range = domain.real_space
        linear = (domain == range)
        super(RealPart, self).__init__(domain, range, linear=linear)

    def _call(self, x):
        """Return ``self(x)``."""
        return x.real

    @property
    def inverse(self):
        """Return the (pseudo-)inverse.

        Examples
        --------
        The inverse is its own inverse if its domain is real:

        >>> r3 = odl.rn(3)
        >>> op = odl.RealPart(r3)
        >>> op.inverse(op([1, 2, 3]))
        rn(3).element([ 1.,  2.,  3.])

        This is not a true inverse, only a pseudoinverse, the complex part
        will by necessity be lost.

        >>> c3 = odl.cn(3)
        >>> op = odl.RealPart(c3)
        >>> op.inverse(op([1 + 2j, 2, 3 - 1j]))
        cn(3).element([ 1.+0.j,  2.+0.j,  3.+0.j])
        """
        if self.is_linear:
            return self
        else:
            return ComplexEmbedding(self.range, self.domain, scalar=1)

    @property
    def adjoint(self):
        r"""Return the (left) adjoint.

        Notes
        -----
        Due to technicalities of operators from a complex space into a real
        space, this does not satisfy the usual adjoint equation:

        .. math::
            \langle Ax, y \rangle = \langle x, A^*y \rangle

        Instead it is an adjoint in a weaker sense as follows:

        .. math::
            \langle AA^*x, y \rangle = \langle A^*x, A^*y \rangle

        Examples
        --------
        The adjoint satisfies the adjoint equation for real spaces:

        >>> r3 = odl.rn(3)
        >>> op = odl.RealPart(r3)
        >>> x = op.domain.element([1, 2, 3])
        >>> y = op.range.element([3, 2, 1])
        >>> x.inner(op.adjoint(y)) == op(x).inner(y)
        True

        If the domain is complex, it only satisfies the weaker definition:

        >>> c3 = odl.cn(3)
        >>> op = odl.RealPart(c3)
        >>> x = op.range.element([1, 2, 3])
        >>> y = op.range.element([3, 2, 1])
        >>> AtAxy = op(op.adjoint(x)).inner(y)
        >>> AtxAty = op.adjoint(x).inner(op.adjoint(y))
        >>> AtAxy == AtxAty
        True
        """
        if self.is_linear:
            return RealPart(self.range, self.domain)
        else:
            return ComplexEmbedding(self.range, self.domain, scalar=1)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> c3 = odl.cn(3)
        >>> op = odl.RealPart(c3)
        >>> op
        RealPart(cn(3))
        """
        posargs = [self.domain]
        optargs = [('range', self.range, self.domain.real_space)]
        inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts)


class ImagPart(Operator):

    """Operator that extracts the imaginary part of a vector.

    Implements::

        ImagPart(x) == x.imag
    """

    def __init__(self, domain, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `TensorSpace` or `Field`
            Space in which the imaginary part should be taken. Needs to
            implement ``domain.real_space`` and ``domain.imag``.
        range : `TensorSpace` or `Field`, optional
            Set to which this operator maps.
            Default: ``domain.real_space``

        Examples
        --------
        Take the imaginary part of complex vector:

        >>> c3 = odl.cn(3)
        >>> op = odl.ImagPart(c3)
        >>> op([1 + 2j, 2, 3 - 1j])
        rn(3).element([ 2.,  0., -1.])

        The operator is the zero operator on real spaces:

        >>> r3 = odl.rn(3)
        >>> op = odl.ImagPart(r3)
        >>> op([1, 2, 3])
        rn(3).element([ 0.,  0.,  0.])
        """
        if range is None:
            range = domain.real_space
        linear = (domain == range)
        super(ImagPart, self).__init__(domain, range, linear=linear)

    def _call(self, x):
        """Return ``self(x)``."""
        return x.imag

    @property
    def inverse(self):
        """Return the pseudoinverse.

        Examples
        --------
        The inverse is the zero operator if the domain is real:

        >>> r3 = odl.rn(3)
        >>> op = odl.ImagPart(r3)
        >>> op.inverse(op([1, 2, 3]))
        rn(3).element([ 0.,  0.,  0.])

        This is not a true inverse, only a pseudoinverse, the real part
        will by necessity be lost.

        >>> c3 = odl.cn(3)
        >>> op = odl.ImagPart(c3)
        >>> op.inverse(op([1 + 2j, 2, 3 - 1j]))
        cn(3).element([ 0.+2.j,  0.+0.j, -0.-1.j])
        """
        if self.is_linear:
            return ZeroOperator(self.range, self.domain)
        else:
            return ComplexEmbedding(self.range, self.domain, scalar=1j)

    @property
    def adjoint(self):
        r"""Return the (left) adjoint.

        Notes
        -----
        Due to technicalities of operators from a complex space into a real
        space, this does not satisfy the usual adjoint equation:

        .. math::
            \langle Ax, y \rangle = \langle x, A^*y \rangle

        Instead it is an adjoint in a weaker sense as follows:

        .. math::
            \langle AA^*x, y \rangle = \langle A^*x, A^*y \rangle

        Examples
        --------
        The adjoint satisfies the adjoint equation for real spaces:

        >>> r3 = odl.rn(3)
        >>> op = odl.ImagPart(r3)
        >>> x = op.domain.element([1, 2, 3])
        >>> y = op.range.element([3, 2, 1])
        >>> x.inner(op.adjoint(y)) == op(x).inner(y)
        True

        If the domain is complex, it only satisfies the weaker definition:

        >>> c3 = odl.cn(3)
        >>> op = odl.ImagPart(c3)
        >>> x = op.range.element([1, 2, 3])
        >>> y = op.range.element([3, 2, 1])
        >>> AtAxy = op(op.adjoint(x)).inner(y)
        >>> AtxAty = op.adjoint(x).inner(op.adjoint(y))
        >>> AtAxy == AtxAty
        True
        """
        if self.is_linear:
            return ZeroOperator(self.range, self.domain)
        else:
            return ComplexEmbedding(self.range, self.domain, scalar=1j)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> c3 = odl.cn(3)
        >>> op = odl.ImagPart(c3)
        >>> op
        ImagPart(cn(3))
        """
        posargs = [self.domain]
        optargs = [('range', self.range, self.domain.real_space)]
        inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts)


class ComplexEmbedding(Operator):

    """Operator that embeds a vector into a complex space.

    Implements::

        ComplexEmbedding(space)(x) <==> space.complex_space.element(x)
    """

    def __init__(self, domain, range=None, scalar=1.0):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `TensorSpace` or `Field`
            Space that should be embedded into its complex counterpart.
            Needs to implement ``domain.complex_space``.
        range : `TensorSpace` or `Field`, optional
            Set to which this operator maps.
            Default: ``domain.complex_space``
        scalar : ``space.complex_space.field`` element, optional
            Scalar to be multiplied with incoming vectors in order
            to get the complex vector.

        Examples
        --------
        Embed real vector into complex space:

        >>> r3 = odl.rn(3)
        >>> op = odl.ComplexEmbedding(r3)
        >>> op([1, 2, 3])
        cn(3).element([ 1.+0.j,  2.+0.j,  3.+0.j])

        Embed real vector as imaginary part into complex space:

        >>> op = odl.ComplexEmbedding(r3, scalar=1j)
        >>> op([1, 2, 3])
        cn(3).element([ 0.+1.j,  0.+2.j,  0.+3.j])

        On complex spaces the operator is the same as simple multiplication by
        scalar:

        >>> c3 = odl.cn(3)
        >>> op = odl.ComplexEmbedding(c3, scalar=1 + 2j)
        >>> op([1 + 1j, 2 + 2j, 3 + 3j])
        cn(3).element([-1.+3.j, -2.+6.j, -3.+9.j])
        """
        if range is None:
            range = domain.complex_space
        super(ComplexEmbedding, self).__init__(domain, range, linear=True)
        self.scalar = self.range.field.element(scalar)

    def _call(self, x, out):
        """Return ``self(x)``."""
        if self.domain.is_real:
            # Real domain, multiply separately
            out.real = self.scalar.real * x
            out.imag = self.scalar.imag * x
        else:
            # Complex domain
            out.lincomb(self.scalar, x)

    @property
    def inverse(self):
        """Return the (left) inverse.

        If the domain is a real space, this is not a true inverse,
        only a (left) inverse.

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> op = odl.ComplexEmbedding(r3, scalar=1)
        >>> op.inverse(op([1, 2, 4]))
        rn(3).element([ 1.,  2.,  4.])
        """
        if self.domain.is_real:
            # Real domain
            # Optimizations for simple cases.
            if self.scalar.real == self.scalar:
                return (1 / self.scalar.real) * RealPart(self.range,
                                                         self.domain)
            elif 1j * self.scalar.imag == self.scalar:
                return (1 / self.scalar.imag) * ImagPart(self.range,
                                                         self.domain)
            else:
                # General case
                inv_scalar = (1 / self.scalar).conjugate()
                return ((inv_scalar.real) * RealPart(self.range, self.domain) +
                        (inv_scalar.imag) * ImagPart(self.range, self.domain))
        else:
            # Complex domain
            return ComplexEmbedding(self.range, self.domain,
                                    self.scalar.conjugate())

    @property
    def adjoint(self):
        r"""Return the (right) adjoint.

        Notes
        -----
        Due to technicalities of operators from a real space into a complex
        space, this does not satisfy the usual adjoint equation:

        .. math::
            \langle Ax, y \rangle = \langle x, A^*y \rangle

        Instead it is an adjoint in a weaker sense as follows:

        .. math::
            \langle A^*Ax, y \rangle = \langle Ax, Ay \rangle

        Examples
        --------
        The adjoint satisfies the adjoint equation for complex spaces

        >>> c3 = odl.cn(3)
        >>> op = ComplexEmbedding(c3, scalar=1j)
        >>> x = c3.element([1 + 1j, 2 + 2j, 3 + 3j])
        >>> y = c3.element([3 + 1j, 2 + 2j, 3 + 1j])
        >>> Axy = op(x).inner(y)
        >>> xAty = x.inner(op.adjoint(y))
        >>> Axy == xAty
        True

        For real domains, it only satisfies the (right) adjoint equation

        >>> r3 = odl.rn(3)
        >>> op = ComplexEmbedding(r3, scalar=1j)
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([3, 2, 3])
        >>> AtAxy = op.adjoint(op(x)).inner(y)
        >>> AxAy = op(x).inner(op(y))
        >>> AtAxy == AxAy
        True
        """
        if self.domain.is_real:
            # Real domain
            # Optimizations for simple cases.
            if self.scalar.real == self.scalar:
                return self.scalar.real * ComplexEmbedding(self.range,
                                                           self.domain)
            elif 1j * self.scalar.imag == self.scalar:
                return self.scalar.imag * ImagPart(self.range, self.domain)
            else:
                # General case
                return (self.scalar.real * RealPart(self.range, self.domain) +
                        self.scalar.imag * ImagPart(self.range, self.domain))
        else:
            # Complex domain
            return ComplexEmbedding(self.range, self.domain,
                                    self.scalar.conjugate())

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> op = odl.ComplexEmbedding(r3)
        >>> op
        ComplexEmbedding(rn(3))
        >>> op = odl.ComplexEmbedding(r3, scalar=1j)
        >>> op
        ComplexEmbedding(rn(3), scalar=1j)
        """
        posargs = [self.domain]
        optargs = [('range', self.range, self.domain.complex_space),
                   ('scalar', self.scalar, 1.0)]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts)


class ComplexModulus(Operator):

    """Operator that computes the modulus (absolute value) of a vector."""

    def __init__(self, domain, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `TensorSpace` or `Field`
            Space in which the complex modulus should be taken. Needs to
            implement ``domain.real_space``.
        range : `TensorSpace` or `Field`, optional
            Set to which this operator maps.
            Default: ``domain.real_space``

        Examples
        --------
        Take the real part of a complex vector:

        >>> c2 = odl.cn(2)
        >>> op = odl.ComplexModulus(c2)
        >>> op([3 + 4j, 2])
        rn(2).element([ 5.,  2.])

        The operator is the absolute value on real spaces:

        >>> r2 = odl.rn(2)
        >>> op = odl.ComplexModulus(r2)
        >>> op([1, -2])
        rn(2).element([ 1.,  2.])

        The operator also works on other `TensorSpace`'s such as
        `DiscreteLp`:

        >>> r2 = odl.uniform_discr(0, 1, 2, dtype=complex)
        >>> op = odl.ComplexModulus(r2)
        >>> op([3 + 4j, 2])
        uniform_discr(0.0, 1.0, 2).element([ 5.,  2.])
        """
        if range is None:
            range = domain.real_space
        linear = (domain == range)
        super(ComplexModulus, self).__init__(domain, range, linear=linear)

    def _call(self, x):
        """Return ``self(x)``."""
        squared_mod = x.real ** 2 + x.imag ** 2
        if hasattr(squared_mod, 'ufuncs'):
            return squared_mod.ufuncs.sqrt()
        else:
            return np.sqrt(squared_mod)

    @property
    def inverse(self):
        """Return the (pseudo-)inverse.

        Examples
        --------
        The (pseudo-)inverse in the real case is the identity:

        >>> r2 = odl.rn(2)
        >>> op = odl.ComplexModulus(r2)
        >>> op.inverse(op([1, -2]))
        rn(2).element([ 1.,  2.])

        If the domain is complex, a pseudo-inverse is taken, assigning equal
        positive weights to the real and complex parts:

        >>> c2 = odl.cn(2)
        >>> op = odl.ComplexModulus(c2)
        >>> op.inverse(op([np.sqrt(2), 2 + 2j]))
        cn(2).element([ 1.+1.j,  2.+2.j])
        """
        if self.is_linear:
            return IdentityOperator(self.range, self.domain)
        else:
            return ComplexEmbedding(self.range, self.domain,
                                    scalar=(1 + 1j) / np.sqrt(2))

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> c2 = odl.cn(2)
        >>> op = odl.ComplexModulus(c2)
        >>> op
        ComplexModulus(cn(2))
        """
        posargs = [self.domain]
        optargs = [('range', self.range, self.domain.real_space)]
        inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
