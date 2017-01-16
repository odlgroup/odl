# Copyright 2014-2016 The ODL development group
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

"""Default functionals defined on any space similar to R^n or L^2."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np
import scipy
from numbers import Integral

from odl.solvers.functional.functional import Functional
from odl.space import ProductSpace
from odl.operator import (Operator, ConstantOperator, ZeroOperator,
                          ScalingOperator, DiagonalOperator, PointwiseNorm)
from odl.solvers.functional.functional import (
    Functional, FunctionalDefaultConvexConjugate)
from odl.solvers.nonsmooth.proximal_operators import (
    proximal_l1, proximal_cconj_l1, proximal_l2, proximal_cconj_l2,
    proximal_l2_squared, proximal_const_func, proximal_box_constraint,
    proximal_cconj, proximal_cconj_kl, proximal_cconj_kl_cross_entropy,
    combine_proximals)
from odl.util import conj_exponent


__all__ = ('LpNorm', 'L1Norm', 'L2Norm', 'L2NormSquared',
           'ZeroFunctional', 'ConstantFunctional', 'IndicatorLpUnitBall',
           'GroupL1Norm', 'IndicatorGroupL1UnitBall', 'IndicatorZero',
           'IndicatorBox', 'IndicatorNonnegativity', 'KullbackLeibler',
           'KullbackLeiblerCrossEntropy', 'SeparableSum',
           'QuadraticForm',
           'NuclearNorm', 'IndicatorNuclearNormUnitBall',
           'ScalingFunctional', 'IdentityFunctional',
           'MoreauEnvelope')


class LpNorm(Functional):

    """The functional corresponding to the Lp-norm.

    Notes
    -----
    If the functional is defined on an :math:`\mathbb{R}^n`-like space, the
    :math:`\| \cdot \|_p`-norm is defined as

    .. math::

        \| x \|_p = \\left(\\sum_{i=1}^n |x_i|^p \\right)^{1/p}.

    If the functional is defined on an :math:`L_2`-like space, the
    :math:`\| \cdot \|_p`-norm is defined as

    .. math::

        \| x \|_p = \\left(\\int_\Omega |x(t)|^p dt. \\right)^{1/p}
    """

    def __init__(self, space, exponent):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `FnBase`
            Domain of the functional.
        exponent : float
            Exponent for the norm (``p``).
        """
        self.exponent = float(exponent)
        super().__init__(space=space, linear=False, grad_lipschitz=np.nan)

    # TODO: update when integration operator is in place: issue #440
    def _call(self, x):
        """Return the Lp-norm of ``x``."""
        if self.exponent == 0:
            return self.domain.one().inner(np.not_equal(x, 0))
        elif self.exponent == 1:
            return x.ufuncs.absolute().inner(self.domain.one())
        elif self.exponent == 2:
            return np.sqrt(x.inner(x))
        elif np.isfinite(self.exponent):
            tmp = x.ufuncs.absolute()
            tmp.ufuncs.power(self.exponent, out=tmp)
            return np.power(tmp.inner(self.domain.one()), 1 / self.exponent)
        elif self.exponent == np.inf:
            return x.ufuncs.absolute().ufuncs.max()
        elif self.exponent == -np.inf:
            return x.ufuncs.absolute().ufuncs.min()
        else:
            raise RuntimeError('unknown exponent')

    @property
    def convex_conj(self):
        """The convex conjugate functional of the Lp-norm."""
        return IndicatorLpUnitBall(self.domain,
                                   exponent=conj_exponent(self.exponent))

    @property
    def proximal(self):
        """Return the proximal factory of the functional.

        See Also
        --------
        odl.solvers.nonsmooth.proximal_operators.proximal_l1 :
            proximal factory for the L1-norm.
        odl.solvers.nonsmooth.proximal_operators.proximal_l2 :
            proximal factory for the L2-norm.
        """
        if self.exponent == 1:
            return proximal_l1(space=self.domain)
        elif self.exponent == 2:
            return proximal_l2(space=self.domain)
        else:
            raise NotImplementedError('`proximal` only implemented for p=1 or '
                                      'p=2')

    @property
    def gradient(self):
        """Gradient operator of the functional.

        The functional is not differentiable in ``x=0``. However, when
        evaluating the gradient operator in this point it will return 0.
        """
        functional = self

        if self.exponent == 1:
            class L1Gradient(Operator):

                """The gradient operator of this functional."""

                def __init__(self):
                    """Initialize a new instance."""
                    super().__init__(functional.domain, functional.domain,
                                     linear=False)

                def _call(self, x):
                    """Apply the gradient operator to the given point."""
                    return x.ufuncs.sign()

            return L1Gradient()
        elif self.exponent == 2:
            class L2Gradient(Operator):

                """The gradient operator of this functional."""

                def __init__(self):
                    """Initialize a new instance."""
                    super().__init__(functional.domain, functional.domain,
                                     linear=False)

                def _call(self, x):
                    """Apply the gradient operator to the given point.

                    The gradient is not defined in 0.
                    """
                    norm_of_x = x.norm()
                    if norm_of_x == 0:
                        return self.domain.zero()
                    else:
                        return x / norm_of_x

            return L2Gradient()
        else:
            raise NotImplementedError('`gradient` only implemented for p=1 or '
                                      'p=2')

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self.domain,
                                       self.exponent)


class GroupL1Norm(Functional):

    """The functional corresponding to the mixed L1--Lp norm on `ProductSpace`.

    The L1-norm, ``|| ||x||_p ||_1``,  is defined as the integral/sum of
    ``||x||_p``, where  ``||x||_p`` is the pointwise p-norm.

    This is also known as the cross norm.

    Notes
    -----
    If the functional is defined on an :math:`\mathbb{R}^{n \\times m}`-like
    space, the group :math:`L_1`-norm, denoted
    :math:`\| \\cdot \|_{\\times, p}` is defined as

    .. math::

        \|F\|_{\\times, p} =
        \\sum_{i = 1}^n \\left(\\sum_{j=1}^m |F_{i,j}|^p\\right)^{1/p}

    If the functional is defined on an :math:`(\\mathcal{L}^p)^m`-like space,
    the group :math:`L_1`-norm is defined as

    .. math::

        \| F \|_{\\times, p} =
        \\int_{\Omega} \\left(\\sum_{j = 1}^m |F_j(x)|^p\\right)^{1/p}
        \mathrm{d}x.
    """

    def __init__(self, vfspace, exponent=None):
        """Initialize a new instance.

        Parameters
        ----------
        vfspace : `ProductSpace`
            Space of vector fields on which the operator acts.
            It has to be a product space of identical spaces, i.e. a
            power space.
        exponent : non-zero float, optional
            Exponent of the norm in each point. Values between
            0 and 1 are currently not supported due to numerical
            instability. Infinity gives the supremum norm.
            Default: ``vfspace.exponent``, usually 2.

        Examples
        --------
        >>> space = odl.rn(2)
        >>> pspace = odl.ProductSpace(space, 2)
        >>> op = GroupL1Norm(pspace)
        >>> op([[3, 3], [4, 4]])
        10.0

        Set exponent of inner (p) norm:

        >>> op2 = GroupL1Norm(pspace, exponent=1)
        >>> op2([[3, 3], [4, 4]])
        14.0
        """
        if not isinstance(vfspace, ProductSpace):
            raise TypeError('`space` must be a `ProductSpace`')
        if not vfspace.is_power_space:
            raise TypeError('`space.is_power_space` must be `True`')
        self.pointwise_norm = PointwiseNorm(vfspace, exponent)
        super().__init__(space=vfspace, linear=False, grad_lipschitz=np.nan)

    def _call(self, x):
        """Return the group L1-norm of ``x``."""
        # TODO: update when integration operator is in place: issue #440
        pointwise_norm = self.pointwise_norm(x)
        return pointwise_norm.inner(pointwise_norm.space.one())

    @property
    def gradient(self):
        """Gradient operator of the functional.

        The functional is not differentiable in ``x=0``. However, when
        evaluating the gradient operator in this point it will return 0.

         Notes
        -----
        The gradient is given by

        .. math::
            \\left[ \\nabla \| \|f\|_1 \|_1 \\right]_i =
            \\frac{f_i}{|f_i|}

        .. math::
            \\left[ \\nabla \| \|f\|_2 \|_1 \\right]_i =
            \\frac{f_i}{\|f\|_2}

        else:

        .. math::
            \\left[ \\nabla || ||f||_p ||_1 \\right]_i =
            \\frac{| f_i |^{p-2} f_i}{||f||_p^{p-1}}
        """
        functional = self

        class GroupL1Gradient(Operator):

            """The gradient operator of the `GroupL1Norm` functional."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)

            def _call(self, x):
                """Return ``self(x)``."""
                p = functional.pointwise_norm.exponent

                if functional.pointwise_norm.exponent == 1:
                    result = np.abs(x)
                    np.divide(x, result, out=result, where=result != 0)
                    return result
                elif functional.pointwise_norm.exponent == 2:
                    result = functional.pointwise_norm(x)
                    np.divide(x, result, out=result, where=result != 0)
                    return result
                else:
                    dividend = np.power(np.abs(x), p - 2) * x
                    divisor = np.power(functional.pointwise_norm(x), p - 1)
                    np.divide(dividend, divisor, out=divisor,
                              where=divisor != 0)
                    return divisor

        return GroupL1Gradient()

    @property
    def proximal(self):
        """Return the ``proximal factory`` of the functional.

        See Also
        --------
        proximal_l1 : `proximal factory` for the L1-norm.
        """
        if self.pointwise_norm.exponent == 1:
            return proximal_l1(space=self.domain)
        elif self.pointwise_norm.exponent == 2:
            return proximal_l1(space=self.domain, isotropic=True)
        else:
            raise NotImplementedError('`proximal` only implemented for p = 1 '
                                      'or 2')

    @property
    def convex_conj(self):
        """The convex conjugate functional of the group L1-norm."""
        conj_exp = conj_exponent(self.pointwise_norm.exponent)
        return IndicatorGroupL1UnitBall(self.domain, exponent=conj_exp)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, exponent={})'.format(self.__class__.__name__,
                                              self.domain,
                                              self.pointwise_norm.exponent)


class IndicatorGroupL1UnitBall(Functional):

    """The convex conjugate to the mixed L1--Lp norm on `ProductSpace`.

    See Also
    --------
    GroupL1Norm
    """

    def __init__(self, vfspace, exponent=None):
        """Initialize a new instance.

        Parameters
        ----------
        vfspace : `ProductSpace`
            Space of vector fields on which the operator acts.
            It has to be a product space of identical spaces, i.e. a
            power space.
        exponent : non-zero float, optional
            Exponent of the norm in each point. Values between
            0 and 1 are currently not supported due to numerical
            instability. Infinity gives the supremum norm.
            Default: ``vfspace.exponent``, usually 2.

        Examples
        --------
        >>> space = odl.rn(2)
        >>> pspace = odl.ProductSpace(space, 2)
        >>> op = IndicatorGroupL1UnitBall(pspace)
        >>> op([[0.1, 0.5], [0.2, 0.3]])
        0
        >>> op([[3, 3], [4, 4]])
        inf

        Set exponent of inner (p) norm:

        >>> op2 = IndicatorGroupL1UnitBall(pspace, exponent=1)
        """
        if not isinstance(vfspace, ProductSpace):
            raise TypeError('`space` must be a `ProductSpace`')
        if not vfspace.is_power_space:
            raise TypeError('`space.is_power_space` must be `True`')
        self.pointwise_norm = PointwiseNorm(vfspace, exponent)
        super().__init__(space=vfspace, linear=False, grad_lipschitz=np.nan)

    def _call(self, x):
        """Return ``self(x)``."""
        x_norm = self.pointwise_norm(x).ufuncs.max()

        if x_norm > 1:
            return np.inf
        else:
            return 0

    @property
    def proximal(self):
        """Return the `proximal factory` of the functional.

        See Also
        --------
        proximal_cconj_l1 : `proximal factory` for the L1-norms convex
                            conjugate.
        """
        if self.pointwise_norm.exponent == np.inf:
            return proximal_cconj_l1(space=self.domain)
        elif self.pointwise_norm.exponent == 2:
            return proximal_cconj_l1(space=self.domain, isotropic=True)
        else:
            raise NotImplementedError('`proximal` only implemented for p = 1 '
                                      'or 2')

    @property
    def convex_conj(self):
        """Convex conjugate functional of IndicatorLpUnitBall.

        Returns
        -------
        convex_conj : GroupL1Norm
            The convex conjugate is the the group L1-norm.
        """
        conj_exp = conj_exponent(self.pointwise_norm.exponent)
        return GroupL1Norm(self.domain, exponent=conj_exp)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, exponent={})'.format(self.__class__.__name__,
                                              self.domain,
                                              self.pointwise_norm.exponent)


class IndicatorLpUnitBall(Functional):

    """The indicator function on the unit ball in given the ``Lp`` norm.

    It does not implement `gradient` since it is not differentiable everywhere.

    Notes
    -----
    This functional is defined as

        .. math::

            f(x) = \\left\{ \\begin{array}{ll}
            0 & \\text{if } ||x||_{L_p} \\leq 1, \\\\
            \\infty & \\text{else,}
            \\end{array} \\right.

    where :math:`||x||_{L_p}` is the :math:`L_p`-norm, which for finite values
    of :math:`p` is defined as

        .. math::

            \| x \|_{L_p} = \\left( \\int_{\Omega} |x|^p dx \\right)^{1/p},

    and for :math:`p = \\infty` it is defined as

        .. math::

            ||x||_{\\infty} = \max_x (|x|).

    The functional also allows noninteger and nonpositive values of the
    exponent :math:`p`, however in this case :math:`\| x \|_{L_p}` is not a
    norm.
    """

    def __init__(self, space, exponent):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `FnBase`
            Domain of the functional.
        exponent : int or infinity
            Specifies wich norm to use.
        """
        super().__init__(space=space, linear=False)
        self.__norm = LpNorm(space, exponent)
        self.__exponent = float(exponent)

    @property
    def exponent(self):
        """Exponent corresponding to the norm."""
        return self.__exponent

    def _call(self, x):
        """Apply the functional to the given point."""
        x_norm = self.__norm(x)

        if x_norm > 1:
            return np.inf
        else:
            return 0

    @property
    def convex_conj(self):
        """The conjugate functional of IndicatorLpUnitBall.

        The convex conjugate functional of an ``Lp`` norm, ``p < infty`` is the
        indicator function on the unit ball defined by the corresponding dual
        norm ``q``, given by ``1/p + 1/q = 1`` and where ``q = infty`` if
        ``p = 1`` [Roc1970]_. By the Fenchel-Moreau theorem, the convex
        conjugate functional of indicator function on the unit ball in ``Lq``
        is the corresponding Lp-norm [BC2011]_.
        """
        if self.exponent == np.inf:
            return L1Norm(self.domain)
        elif self.exponent == 2:
            return L2Norm(self.domain)
        else:
            return LpNorm(self.domain, exponent=conj_exponent(self.exponent))

    @property
    def proximal(self):
        """Return the `proximal factory` of the functional.

        See Also
        --------
        odl.solvers.nonsmooth.proximal_operators.proximal_cconj_l1 :
            `proximal factory` for convex conjuagte of L1-norm.
        odl.solvers.nonsmooth.proximal_operators.proximal_cconj_l2 :
            `proximal factory` for convex conjuagte of L2-norm.
        """
        if self.exponent == np.inf:
            return proximal_cconj_l1(space=self.domain)
        elif self.exponent == 2:
            return proximal_cconj_l2(space=self.domain)
        else:
            raise NotImplementedError('`gradient` only implemented for p=2 or '
                                      'p=inf')

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r},{!r})'.format(self.__class__.__name__,
                                      self.domain, self.exponent)


class L1Norm(LpNorm):

    """The functional corresponding to L1-norm.

    The L1-norm, ``||x||_1``,  is defined as the integral/sum of ``|x|``.

    Notes
    -----
    If the functional is defined on an :math:`\mathbb{R}^n`-like space, the
    :math:`\| \cdot \|_1`-norm is defined as

    .. math::

        \| x \|_1 = \\sum_{i=1}^n |x_i|.

    If the functional is defined on an :math:`L_2`-like space, the
    :math:`\| \cdot \|_1`-norm is defined as

    .. math::

        \| x \|_1 = \\int_\Omega |x(t)| dt.
    """

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `FnBase`
            Domain of the functional.
        """
        super().__init__(space=space, exponent=1)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__,
                                 self.domain)


class L2Norm(LpNorm):

    """The functional corresponding to the L2-norm.

    The L2-norm, ``||x||_2``,  is defined as the square-root out of the
    integral/sum of ``x^2``.

    Notes
    -----
    If the functional is defined on an :math:`\mathbb{R}^n`-like space, the
    :math:`\| \cdot \|_2`-norm is defined as

    .. math::

        \| x \|_2 = \\sqrt{ \\sum_{i=1}^n |x_i|^2 }.

    If the functional is defined on an :math:`L_2`-like space, the
    :math:`\| \cdot \|_2`-norm is defined as

    .. math::

        \| x \|_2 = \\sqrt{ \\int_\Omega |x(t)|^2 dt. }
    """

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `FnBase`
            Domain of the functional.
        """
        super().__init__(space=space, exponent=2)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__,
                                 self.domain)


class L2NormSquared(Functional):

    """The functional corresponding to the squared L2-norm.

    The squared L2-norm, ``||x||_2^2``,  is defined as the integral/sum of
    ``x^2``.

    Notes
    -----
    If the functional is defined on an :math:`\mathbb{R}^n`-like space, the
    :math:`\| \cdot \|_2^2`-functional is defined as

    .. math::

        \| x \|_2^2 = \\sum_{i=1}^n |x_i|^2.

    If the functional is defined on an :math:`L_2`-like space, the
    :math:`\| \cdot \|_2^2`-functional is defined as

    .. math::

        \| x \|_2^2 = \\int_\Omega |x(t)|^2 dt.
    """

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `FnBase`
            Domain of the functional.
        """
        super().__init__(space=space, linear=False, grad_lipschitz=2)

    # TODO: update when integration operator is in place: issue #440
    def _call(self, x):
        """Return the squared L2-norm of ``x``."""
        return x.inner(x)

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        return ScalingOperator(self.domain, 2.0)

    @property
    def proximal(self):
        """Return the `proximal factory` of the functional.

        See Also
        --------
        odl.solvers.nonsmooth.proximal_operators.proximal_l2_squared :
            `proximal factory` for the squared L2-norm.
        """
        return proximal_l2_squared(space=self.domain)

    @property
    def convex_conj(self):
        """The convex conjugate functional of the squared L2-norm.

        Notes
        -----
        The conjugate functional of :math:`\| \\cdot \|_2^2` is
        :math:`\\frac{1}{4}\| \\cdot \|_2^2`
        """
        return (1.0 / 4) * L2NormSquared(self.domain)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.domain)


class ConstantFunctional(Functional):

    """The constant functional.

    This functional maps all elements in the domain to a given, constant value.
    """

    def __init__(self, space, constant):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            Domain of the functional.
        constant : element in ``domain.field``
            The constant value of the functional
        """
        super().__init__(space=space, linear=(constant == 0), grad_lipschitz=0)
        self.__constant = self.range.element(constant)

    @property
    def constant(self):
        """The constant value of the functional."""
        return self.__constant

    def _call(self, x):
        """Return a constant value."""
        return self.constant

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        return ZeroOperator(self.domain)

    @property
    def proximal(self):
        """Return the `proximal factory` of the functional."""
        return proximal_const_func(self.domain)

    @property
    def convex_conj(self):
        """Convex conjugate functional of the constant functional.

        Notes
        -----
        This functional is defined as

         .. math::

            f^*(x) = \\left\{ \\begin{array}{ll}
            -constant & \\text{if } x = 0, \\\\
            \\infty & \\text{else}
            \\end{array} \\right.
        """
        return IndicatorZero(self.domain, -self.constant)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self.domain, self.constant)


class ZeroFunctional(ConstantFunctional):

    """Functional that maps all elements in the domain to zero."""

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            Domain of the functional.
        """
        super().__init__(space=space, constant=0)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.domain)


class ScalingFunctional(Functional, ScalingOperator):

    """Functional that scales the input argument by a value.

    Since the range of a functional is always a field, the domain of this
    functional is also a field, i.e. real or complex numbers.
    """

    def __init__(self, field, scale):
        """Initialize a new instance.

        Parameters
        ----------
        field : `Field`
            Domain of the functional.
        scale : element in ``domain``
            The constant value to scale by.

        Examples
        --------
        >>> import odl
        >>> field = odl.RealNumbers()
        >>> func = ScalingFunctional(field, 3)
        >>> func(5)
        15.0
        """
        Functional.__init__(self, space=field, linear=True, grad_lipschitz=0)
        ScalingOperator.__init__(self, field, scale)

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        return ConstantFunctional(self.domain, self.scalar)


class IdentityFunctional(ScalingFunctional):

    """Functional that maps a scalar to itself.

    See Also
    --------
    IdentityOperator
    """

    def __init__(self, field):
        """Initialize a new instance.

        Parameters
        ----------
        field : `Field`
            Domain of the functional.
        """
        ScalingFunctional.__init__(self, field, 1.0)


class IndicatorBox(Functional):

    """Indicator on some box shaped domain.

    Notes
    -----
    The indicator :math:`F` with lower bound :math:`a` and upper bound
    :math:`b` is defined as:

    .. math::

        F(x) = \\begin{cases}
            0 & \\text{if } a \\leq x \\leq b \\text{ everywhere}, \\\\
            \\infty & \\text{else}
            \\end{cases}
    """

    def __init__(self, space, lower=None, upper=None):
        """Initialize an instance.

        Parameters
        ----------
        space : `LinearSpace`
            Domain of the functional.
        lower : ``space.field`` element or ``space`` `element-like`, optional
            The lower bound.
            Default: ``None``, interpreted as -infinity
        upper : ``space.field`` element or ``space`` `element-like`, optional
            The upper bound.
            Default: ``None``, interpreted as +infinity

        Examples
        --------
        >>> space = odl.rn(3)
        >>> func = IndicatorBox(space, 0, 2)
        >>> func([0, 1, 2])  # all points inside
        0
        >>> func([0, 1, 3])  # one point outside
        inf
        """
        Functional.__init__(self, space, linear=False)
        self.lower = lower
        self.upper = upper

    def _call(self, x):
        """Apply the functional to the given point."""
        # Compute the projection of x onto the box, if this is equal to x we
        # know x is inside the box.
        tmp = self.domain.element()
        if self.lower is not None and self.upper is None:
            x.ufuncs.maximum(self.lower, out=tmp)
        elif self.lower is None and self.upper is not None:
            x.ufuncs.minimum(self.upper, out=tmp)
        elif self.lower is not None and self.upper is not None:
            x.ufuncs.maximum(self.lower, out=tmp)
            tmp.ufuncs.minimum(self.upper, out=tmp)
        else:
            tmp.assign(x)

        return np.inf if x.dist(tmp) > 0 else 0

    @property
    def proximal(self):
        """Return the `proximal factory` of the functional."""
        return proximal_box_constraint(self.domain, self.lower, self.upper)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r})'.format(self.__class__.__name__,
                                             self.domain,
                                             self.lower, self.upper)


class IndicatorNonnegativity(IndicatorBox):

    """Indicator on the set of non-negative numbers.

    Notes
    -----
    The nonnegativity indicator :math:`F`  is defined as:

    .. math::

        F(x) = \\begin{cases}
            0 & \\text{if } 0 \\leq x \\text{ everywhere}, \\\\
            \\infty & \\text{else}
            \\end{cases}
    """

    def __init__(self, space):
        """Initialize an instance.

        Parameters
        ----------
        space : `LinearSpace`
            Domain of the functional.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> func = IndicatorNonnegativity(space)
        >>> func([0, 1, 2])  # all points positive
        0
        >>> func([0, 1, -3])  # one point negative
        inf
        """
        IndicatorBox.__init__(self, space, lower=0, upper=None)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.domain)


class IndicatorZero(Functional):

    """The indicator function of the singleton set {0}.

    The function has a constant value if the input is zero, otherwise infinity.
    """

    def __init__(self, space, constant=0):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            Domain of the functional.
        constant : element in ``domain.field``, optional
            The constant value of the functional

        Examples
        --------
        >>> space = odl.rn(3)
        >>> func = IndicatorZero(space)
        >>> func([0, 0, 0])
        0
        >>> func([0, 0, 1])
        inf

        >>> func = IndicatorZero(space, constant=2)
        >>> func([0, 0, 0])
        2
        """
        self.__constant = constant
        super().__init__(space, linear=False)

    @property
    def constant(self):
        """The constant value of the functional if ``x=0``."""
        return self.__constant

    def _call(self, x):
        """Apply the functional to the given point."""
        if x.norm() == 0:
            # In this case x is the zero-element.
            return self.constant
        else:
            return np.inf

    @property
    def convex_conj(self):
        """The convex conjugate functional.

        Notes
        -----
        By the Fenchel-Moreau theorem the convex conjugate is the constant
        functional [BC2011]_ with the constant value of -`constant`.
        """
        return ConstantFunctional(self.domain, -self.constant)

    @property
    def proximal(self):
        """Return the proximal factory of the functional.

        This is the zero operator.
        """
        def zero_proximal(sigma=1.0):
            """Proximal factory for zero operator.

            Parameters
            ----------
            sigma : positive float, optional
                Step size parameter.
            """
            return ZeroOperator(self.domain)

        return zero_proximal

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self.domain, self.constant)


class KullbackLeibler(Functional):

    """The Kullback-Leibler divergence functional.

    Notes
    -----
    The functional :math:`F` with prior :math:`g>0` is given by:

    .. math::
        F(x)
        =
        \\begin{cases}
            \\sum_{i} \left( x_i - g_i + g_i \log \left( \\frac{g_i}{ x_i }
            \\right) \\right) & \\text{if } x_i > 0 \\forall i
            \\\\
            +\\infty & \\text{else.}
        \\end{cases}

    KL based objectives are common in MLEM optimization problems and are often
    used as data-matching term when data noise governed by a multivariate
    Poisson probability distribution is significant.

    The functional is related to the Kullback-Leibler cross entropy functional
    `KullbackLeiblerCrossEntropy`. The KL cross entropy is the one
    diescribed in `this Wikipedia article
    <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_, and
    the functional :math:`F` is obtained by switching place of the prior and
    the varialbe in the KL cross entropy functional.

    For a theoretical exposition, see `Csiszar1991`_.

    See Also
    --------
    KullbackLeiblerConvexConj : the convex conjugate functional
    KullbackLeiblerCrossEntropy : related functional

    References
    ----------
    .. _Csiszar1991:  http://www.jstor.org/stable/2241918
    """

    def __init__(self, space, prior=None):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `FnBase`
            Domain of the functional.
        prior : ``space`` `element-like`, optional
            Data term, positive.
            Default: if None it is take as the one-element.
        """
        super().__init__(space=space, linear=False, grad_lipschitz=np.nan)

        if prior is not None and prior not in self.domain:
            raise ValueError('`prior` not in `domain`'
                             ''.format(prior, self.domain))

        self.__prior = prior

    @property
    def prior(self):
        """The prior in the Kullback-Leibler functional."""
        return self.__prior

    # TODO: update when integration operator is in place: issue #440
    def _call(self, x):
        """Return the KL-diveregnce in the point ``x``.

        If any components of ``x`` is non-positive, the value is positive
        infinity.
        """
        if self.prior is None:
            tmp = ((x - 1 - np.log(x)).inner(self.domain.one()))
        else:
            tmp = ((x - self.prior + self.prior * np.log(self.prior / x))
                   .inner(self.domain.one()))
        if np.isnan(tmp):
            # In this case, some element was less than or equal to zero
            return np.inf
        else:
            return tmp

    @property
    def gradient(self):
        """Gradient operator of the functional.

        The gradient is not defined in points where one or more components
        are non-positive.
        """
        functional = self

        class KLGradient(Operator):

            """The gradient operator of this functional."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)

            def _call(self, x):
                """Apply the gradient operator to the given point.
                The gradient is not defined in points where one or more
                components are non-positive.
                """
                if functional.prior is None:
                    return (-1.0) / x + 1
                else:
                    return (-functional.prior) / x + 1

        return KLGradient()

    @property
    def proximal(self):
        """Return the `proximal factory` of the functional.

        See Also
        --------
        odl.solvers.nonsmooth.proximal_operators.proximal_cconj_kl :
            `proximal factory` for convex conjugate of KL.
        odl.solvers.nonsmooth.proximal_operators.proximal_cconj :
            Proximal of the convex conjugate of a functional.
        """
        return proximal_cconj(proximal_cconj_kl(space=self.domain,
                                                g=self.prior))

    @property
    def convex_conj(self):
        """The convex conjugate functional of the KL-functional."""
        return KullbackLeiblerConvexConj(self.domain, self.prior)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self.domain, self.prior)


class KullbackLeiblerConvexConj(Functional):

    """The convex conjugate of Kullback-Leibler divergence functional.

    Notes
    -----
    The functional :math:`F^*` with prior :math:`g>0` is given by:

    .. math::
        F^*(x)
        =
        \\begin{cases}
            \\sum_{i} \left( -g_i \ln(1 - x_i) \\right)
            & \\text{if } x_i < 1 \\forall i
            \\\\
            +\\infty & \\text{else}
        \\end{cases}

    See Also
    --------
    KullbackLeibler : convex conjugate functional
    """

    def __init__(self, space, prior=None):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `FnBase`
            Domain of the functional.
        g : ``space`` `element-like`, optional
            Data term, positive.
            Default: if None it is take as the one-element.
        """
        super().__init__(space=space, linear=False, grad_lipschitz=np.nan)

        if prior is not None and prior not in self.domain:
            raise ValueError('`prior` not in `domain`'
                             ''.format(prior, self.domain))

        self.__prior = prior

    @property
    def prior(self):
        """The prior in convex conjugate Kullback-Leibler functional."""
        return self.__prior

    # TODO: update when integration operator is in place: issue #440
    def _call(self, x):
        """Return the value in the point ``x``.

        If any components of ``x`` is larger than or equal to 1, the value is
        positive infinity.
        """
        if self.prior is None:
            tmp = -1.0 * (np.log(1 - x)).inner(self.domain.one())
        else:
            tmp = (-self.prior * np.log(1 - x)).inner(self.domain.one())
        if np.isnan(tmp):
            # In this case, some element was larger than or equal to one
            return np.inf
        else:
            return tmp

    @property
    def gradient(self):
        """Gradient operator of the functional.

        The gradient is not defined in points where one or more components
        are larger than or equal to one.
        """
        functional = self

        class KLCCGradient(Operator):

            """The gradient operator of this functional."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)

            def _call(self, x):
                """Apply the gradient operator to the given point.

                The gradient is not defined in points where one or more
                components are larger than or equal to one.
                """
                if functional.prior is None:
                    return 1.0 / (1 - x)
                else:
                    return functional.prior / (1 - x)

        return KLCCGradient()

    @property
    def proximal(self):
        """Return the `proximal factory` of the functional.

        See Also
        --------
        odl.solvers.nonsmooth.proximal_operators.proximal_cconj_kl :
            `proximal factory` for convex conjugate of KL.
        odl.solvers.nonsmooth.proximal_operators.proximal_cconj :
            Proximal of the convex conjugate of a functional.
        """
        return proximal_cconj_kl(space=self.domain, g=self.prior)

    @property
    def convex_conj(self):
        """The convex conjugate functional of the conjugate KL-functional."""
        return KullbackLeibler(self.domain, self.prior)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self.domain, self.prior)


class KullbackLeiblerCrossEntropy(Functional):

    """The Kullback-Leibler Cross Entropy divergence functional.

    Notes
    -----
    The functional :math:`F` with prior :math:`g>0` is given by:

    .. math::
        F(x)
        =
        \\begin{cases}
            \\sum_{i} \left( g_i - x_i + x_i \log \left( \\frac{x_i}{g_i}
            \\right) \\right)
            & \\text{if } g_i > 0 \\forall i
            \\\\
            +\\infty & \\text{else}
        \\end{cases}

    For further information about the functional, see the
    `Wikipedia article on the Kullback Leibler divergence
    <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_,
    or read for example `this article
    <http://ieeexplore.ieee.org/document/1056144/?arnumber=1056144>`_.

    The KL cross entropy functional :math:`F`, described above, is related to
    another functional which is also know as KL divergence. This functional
    is often used as data discrepancy term in inverse problems, when data is
    corrupted with Poisson noise. It is obtained by changing place
    of the prior and the variable. See the See Also section.

    For a theoretical exposition, see `Csiszar1991`_.

    See Also
    --------
    KullbackLeibler : related functional
    KullbackLeiblerCrossEntropyConvexConj : the convex conjugate functional

    References
    ----------
    .. _Csiszar1991:  http://www.jstor.org/stable/2241918
    """

    def __init__(self, space, prior=None):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `FnBase`
            Domain of the functional.
        prior : ``space`` `element-like`, optional
            Data term, positive.
            Default: if None it is take as the one-element.
        """
        super().__init__(space=space, linear=False, grad_lipschitz=np.nan)

        if prior is not None and prior not in self.domain:
            raise ValueError('`prior` not in `domain`'
                             ''.format(prior, self.domain))

        self.__prior = prior

    @property
    def prior(self):
        """The prior in the Kullback-Leibler functional."""
        return self.__prior

    # TODO: update when integration operator is in place: issue #440
    def _call(self, x):
        """Return the KL-diveregnce in the point ``x``.

        If any components of ``x`` is non-positive, the value is positive
        infinity.
        """
        if self.prior is None:
            tmp = (1 - x + scipy.special.xlogy(x, x)).inner(self.domain.one())
        else:
            tmp = ((self.prior - x + scipy.special.xlogy(x, x / self.prior))
                   .inner(self.domain.one()))
        if np.isnan(tmp):
            # In this case, some element was less than or equal to zero
            return np.inf
        else:
            return tmp

    @property
    def gradient(self):
        """Gradient operator of the functional.

        The gradient is not defined in points where one or more components
        are less than or equal to 0.
        """
        functional = self

        class KLCrossEntropyGradient(Operator):

            """The gradient operator of this functional."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)

            def _call(self, x):
                """Apply the gradient operator to the given point.

                The gradient is not defined in for points with components less
                than or equal to zero.
                """
                if functional.prior is None:
                    tmp = np.log(x)
                else:
                    tmp = np.log(x / functional.prior)

                if np.all(np.isfinite(tmp)):
                    return tmp
                else:
                    # The derivative is not defined.
                    raise ValueError('The gradient of the Kullback-Leibler '
                                     'Cross Entropy functional is not defined '
                                     'for `x` with one or more components '
                                     'less than or equal to zero.'.format(x))

        return KLCrossEntropyGradient()

    @property
    def proximal(self):
        """Return the `proximal factory` of the functional.

        See Also
        --------
        odl.solvers.nonsmooth.proximal_operators.\
proximal_cconj_kl_cross_entropy :
            `proximal factory` for convex conjugate of the KL cross entropy.
        odl.solvers.nonsmooth.proximal_operators.proximal_cconj :
            Proximal of the convex conjugate of a functional.
        """
        return proximal_cconj(proximal_cconj_kl_cross_entropy(
            space=self.domain, g=self.prior))

    @property
    def convex_conj(self):
        """The convex conjugate functional of the KL-functional."""
        return KullbackLeiblerCrossEntropyConvexConj(self.domain, self.prior)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self.domain, self.prior)


class KullbackLeiblerCrossEntropyConvexConj(Functional):
    """The convex conjugate of Kullback-Leibler Cross Entorpy functional.

    Notes
    -----
    The functional :math:`F^*` with prior :math:`g>0` is given by

    .. math::
        F^*(x) = \\sum_i g_i \\left(e^{x_i} - 1\\right)

    See Also
    --------
    KullbackLeiblerCrossEntropy : convex conjugate functional
    """

    def __init__(self, space, prior=None):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `FnBase`
            Domain of the functional.
        g : ``space`` `element-like`, optional
            Data term, positive.
            Default: if None it is take as the one-element.
        """
        super().__init__(space=space, linear=False, grad_lipschitz=np.nan)

        if prior is not None and prior not in self.domain:
            raise ValueError('`prior` not in `domain`'
                             ''.format(prior, self.domain))

        self.__prior = prior

    @property
    def prior(self):
        """The prior in convex conjugate Kullback-Leibler Cross Entorpy."""
        return self.__prior

    # TODO: update when integration operator is in place: issue #440
    def _call(self, x):
        """Return the value in the point ``x``."""
        if self.prior is None:
            tmp = (np.exp(x) - 1).inner(self.domain.one())
        else:
            tmp = (self.prior * (np.exp(x) - 1)).inner(self.domain.one())
        return tmp

    # TODO: replace this when UFuncOperators is in place: PL #576
    @property
    def gradient(self):
        """Gradient operator of the functional."""
        functional = self

        class KLCrossEntCCGradient(Operator):

            """The gradient operator of this functional."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)

            def _call(self, x):
                """Apply the gradient operator to the given point."""
                if functional.prior is None:
                    return np.exp(x)
                else:
                    return functional.prior * np.exp(x)

        return KLCrossEntCCGradient()

    @property
    def proximal(self):
        """Return the `proximal factory` of the functional.

        See Also
        --------
        odl.solvers.nonsmooth.proximal_operators.\
proximal_cconj_kl_cross_entropy :
            `proximal factory` for convex conjugate of the KL cross entropy.
        """
        return proximal_cconj_kl_cross_entropy(space=self.domain, g=self.prior)

    @property
    def convex_conj(self):
        """The convex conjugate functional of the conjugate KL-functional."""
        return KullbackLeiblerCrossEntropy(self.domain, self.prior)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self.domain, self.prior)


class SeparableSum(Functional):

    """The functional corresponding to separable sum of functionals.

    The separable sum of functionals ``f_1, f_2, ..., f_n`` is given by::

        h(x_1, x_2, ..., x_n) = sum_i^n f_i(x_i)

    The separable sum is thus defined for any collection of functionals with
    the same range.

    Notes
    -----
    The separable sum of functionals :math:`f_1, f_2, ..., f_n` is given by

    .. math::
        h(x_1, x_2, ..., x_n) = \sum_{i=1}^n f_i(x_i)

    It has several useful features that also distribute. For example, the
    gradient is a `DiagonalOperator`:

    .. math::
        [\\nabla h](x_1, x_2, ..., x_n) =
        [\\nabla f_1(x_i), \\nabla f_2(x_i), ..., \\nabla f_n(x_i)]

    The convex conjugate is also a separable sum:

    .. math::
        [h^*](y_1, y_2, ..., y_n) = \sum_{i=1}^n f_i^*(y_i)

    And the proximal distributes:

    .. math::
        \mathrm{prox}_{\\sigma h}(x_1, x_2, ..., x_n) =
        [\mathrm{prox}_{\\sigma f_1}(x_1),
         \mathrm{prox}_{\\sigma f_2}(x_2),
         ...,
         \mathrm{prox}_{\\sigma f_n}(x_n)]

    """

    def __init__(self, *functionals):
        """Initialize a new instance.

        Parameters
        ----------
        functional1, ..., functionalN : `Functional`
            The functionals in the sum.
            Can also be given as ``space, n`` with ``n`` integer,
            in which case the functional is repeated ``n`` times.

        Examples
        --------
        Create functional ``f([x1, x2]) = ||x1||_1 + ||x2||_2``:

        >>> space = odl.rn(3)
        >>> l1 = odl.solvers.L1Norm(space)
        >>> l2 = odl.solvers.L2Norm(space)
        >>> f_sum = odl.solvers.SeparableSum(l1, l2)

        Create functional ``f([x1, ... ,xn]) = \sum_i ||xi||_1``:

        >>> f_sum = odl.solvers.SeparableSum(l1, 5)
        """
        # Make a power space if the second argument is an integer
        if (len(functionals) == 2 and
                isinstance(functionals[1], Integral)):
            functionals = [functionals[0]] * functionals[1]

        domains = [func.domain for func in functionals]
        domain = ProductSpace(*domains)
        linear = all(func.is_linear for func in functionals)

        self.__functionals = functionals

        super().__init__(space=domain, linear=linear)

    def _call(self, x):
        """Return the separable sum evaluated in ``x``."""
        return sum(fi(xi) for xi, fi in zip(x, self.functionals))

    @property
    def functionals(self):
        """The summands of the functional."""
        return self.__functionals

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        gradients = [func.gradient for func in self.functionals]
        return DiagonalOperator(*gradients)

    @property
    def proximal(self):
        """Return the `proximal factory` of the functional.

        The proximal operator separates over separable sums.

        Returns
        -------
        proximal : combine_proximals
        """
        proximals = [func.proximal for func in self.functionals]
        return combine_proximals(*proximals)

    @property
    def convex_conj(self):
        """The convex conjugate functional.

        Convex conjugate distributes over separable sums, so the result is
        simply the separable sum of the convex conjugates.
        """
        convex_conjs = [func.convex_conj for func in self.functionals]
        return SeparableSum(*convex_conjs)

    def __repr__(self):
        """Return ``repr(self)``."""
        func_repr = ', '.join(repr(func) for func in self.functionals)
        return '{}({})'.format(self.__class__.__name__, func_repr)


class QuadraticForm(Functional):

    """Functional for a general quadratic form ``x^T A x + b^T x + c``."""

    def __init__(self, operator=None, vector=None, constant=0):
        """Initialize a new instance.

        All parameters are optional, but at least one of ``op`` and ``vector``
        have to be provided in order to infer the space.

        The computed value is::

            x.inner(operator(x)) + vector.inner(x) + constant

        Parameters
        ----------
        operator : `Operator`, optional
            Operator for the quadratic part of the functional.
            ``None`` means that this part is ignored.
        vector : `Operator`, optional
            Vector for the linear part of the functional.
            ``None`` means that this part is ignored.
        constant : `Operator`, optional
            Constant offset of the functional.
        """
        if operator is None and vector is None:
            raise ValueError('need to provide at least one of `operator` and '
                             '`vector`')
        if operator is not None:
            domain = operator.domain
        elif vector is not None:
            domain = vector.space

        if (operator is not None and vector is not None and
                vector not in operator.domain):
            raise ValueError('domain of `operator` and space of `vector` need '
                             'to match')

        self.__operator = operator
        self.__vector = vector
        self.__constant = constant

        super().__init__(space=domain,
                         linear=(operator is None and constant == 0))

        if self.constant not in self.range:
            raise ValueError('`constant` must be an element in the range of '
                             'the functional')

    @property
    def operator(self):
        """Operator for the quadratic part of the functional."""
        return self.__operator

    @property
    def vector(self):
        """Vector for the linear part of the functional."""
        return self.__vector

    @property
    def constant(self):
        """Constant offset of the functional."""
        return self.__constant

    def _call(self, x):
        """Return ``self(x)``."""
        if self.operator is None:
            return self.vector.inner(x) + self.constant
        elif self.vector is None:
            return x.inner(self.operator(x)) + self.constant
        else:
            tmp = self.operator(x)
            tmp += self.vector
            return x.inner(tmp) + self.constant

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        if self.operator is None:
            return ConstantOperator(self.domain, self.vector)
        else:
            if not self.operator.is_linear:
                # TODO: Acutally works otherwise, but needs more work
                raise NotImplementedError('`operator` must be linear')

            # Figure out if operator is symmetric
            opadjoint = self.operator.adjoint
            if opadjoint == self.operator:
                gradient = 2 * self.operator
            else:
                gradient = self.operator + opadjoint

            # Return gradient
            if self.vector is None:
                return gradient
            else:
                return gradient + self.vector

    @property
    def convex_conj(self):
        """The convex conjugate functional of the quadratic form.

        Notes
        -----
        The convex conjugate of the quadratic form :math:`<x, Ax> + <b, x> + c`
        is given by

        .. math::
            (<x, Ax> + <b, x> + c)^* (x) =
            <(x - b), A^-1 (x - b)> - c =
            <x , A^-1 x> - <x, A^-* b> - <x, A^-1 b> + <b, A^-1 b> - c

        """
        if self.operator is None:
            # Everywhere infinite in this case
            raise ValueError('convex conjugate not defined without operator')

        if self.vector is None:
            # Handle trivial case separately
            return QuadraticForm(operator=self.operator.inverse,
                                 constant=-self.constant)
        else:
            # Compute the needed variables
            opinv = self.operator.inverse
            vector = -opinv.adjoint(self.vector) - opinv(self.vector)
            constant = self.vector.inner(opinv(self.vector)) - self.constant

            # Create new quadratic form
            return QuadraticForm(operator=opinv,
                                 vector=vector,
                                 constant=constant)


class NuclearNorm(Functional):

    """Nuclear norm for matrix valued functions.

    Notes
    -----
    For a matrix-valued function
    :math:`f : \\Omega \\rightarrow \\mathbb{R}^{n \\times m}`,
    the nuclear norm with parameters :math:`p` and :math:`q` is defined by

    .. math::
        \\left( \int_\Omega \|\sigma(f(x))\|_p^q d x \\right)^{1/q},

    where :math:`\sigma(f(x))` is the vector of singular values of the matrix
    :math:`f(x)` and :math:`\| \cdot \|_p` is the usual :math:`p`-norm on
    :math:`\mathbb{R}^{\min(n, m)}`.

    For a detailed description of its properties, e.g, its proximal, convex
    conjugate and more, see [Du+2016]_.

    References
    ----------
    J. Duran, M. Moeller, C. Sbert, and D. Cremers. Collaborative Total
    Variation: A General Framework for Vectorial TV Models, SIAM Journal of
    Imaging Sciences 9(1): 116--151, 2016.
    """

    def __init__(self, space, outer_exp=1, singular_vector_exp=2):
        """Initialize a new instance.

        Parameters
        ----------
        space : `ProductSpace` of `ProductSpace` of `FnBase`
            Domain of the functional.
        outer_exp : {1, 2, inf}, optional
            Exponent for the outer norm.
        singular_vector_exp : {1, 2, inf}, optional
            Exponent for the norm for the singular vectors.

        Examples
        --------
        Simple example, nuclear norm of matrix valued function with all ones
        in 3 points. The singular values are [2, 0], which has squared 2-norm
        2. Since there are 3 points, the expected total value is 6.

        >>> r3 = odl.rn(3)
        >>> space = odl.ProductSpace(odl.ProductSpace(r3, 2), 2)
        >>> norm = NuclearNorm(space)
        >>> norm(space.one())
        6.0
        """
        if (not isinstance(space, ProductSpace) or
                not isinstance(space[0], ProductSpace)):
            raise TypeError('`space` must be a `ProductSpace` of '
                            '`ProductSpace`s')
        if (not space.is_power_space or not space[0].is_power_space):
            raise TypeError('`space` must be of the form `FnBase^(nxm)`')

        super().__init__(space=space, linear=False, grad_lipschitz=np.nan)

        self.outernorm = LpNorm(self.domain[0][0], exponent=outer_exp)
        self.pwisenorm = PointwiseNorm(self.domain[0],
                                       exponent=singular_vector_exp)
        self.pshape = (self.domain.size, self.domain[0].size)

    # TODO: Remove when numpy 1.11 is required by ODL
    def _moveaxis(self, arr, source, dest):
        """Implementation of `numpy.moveaxis`.

        Needed since `numpy.moveaxis` requires numpy 1.11, which ODL doesn't
        have as a dependency.
        """
        try:
            source = list(source)
        except TypeError:
            source = [source]
        try:
            dest = list(dest)
        except TypeError:
            dest = [dest]

        source = [a + arr.ndim if a < 0 else a for a in source]
        dest = [a + arr.ndim if a < 0 else a for a in dest]

        order = [n for n in range(arr.ndim) if n not in source]

        for dest, src in sorted(zip(dest, source)):
            order.insert(dest, src)

        return arr.transpose(order)

    def _asarray(self, vec):
        """Convert ``x`` to an array.

        Here the indices are changed such that the "outer" indices come last
        in order to have the access order as `numpy.linalg.svd` needs it.

        This is the inverse of `_asvector`.
        """
        shape = self.domain[0][0].shape + self.pshape
        arr = np.empty(shape, dtype=self.domain.dtype)
        for i, xi in enumerate(vec):
            for j, xij in enumerate(xi):
                arr[..., i, j] = xij.asarray()

        return arr

    def _asvector(self, arr):
        """Convert ``vec`` to a `domain` element.

        This is the inverse of `_asarray`.
        """
        result = self._moveaxis(arr, [-2, -1], [0, 1])
        return self.domain.element(result)

    def _call(self, x):
        """Return ``self(x)``."""

        # Convert to array with most
        arr = self._asarray(x)
        svd_diag = np.linalg.svd(arr, compute_uv=False)

        # Rotate the axes so the svd-direction is first
        s_reordered = self._moveaxis(svd_diag, -1, 0)

        # Return nuclear norm
        return self.outernorm(self.pwisenorm(s_reordered))

    @property
    def proximal(self):
        """Return the proximal operator.

        Raises
        ------
        NotImplementedError
            if ``outer_exp`` is not 1 or ``singular_vector_exp`` is not 1, 2 or
            infinity
        """
        if self.outernorm.exponent != 1:
            raise NotImplementedError('`proximal` only implemented for '
                                      '`outer_exp==1`')
        if self.pwisenorm.exponent not in [1, 2, np.inf]:
            raise NotImplementedError('`proximal` only implemented for '
                                      '`singular_vector_exp` in [1, 2, inf]')

        def nddot(a, b):
            """Compute pointwise matrix product in the last indices."""
            return np.einsum('...ij,...jk->...ik', a, b)

        func = self

        # Add epsilon to fix rounding errors, i.e. make sure that when we
        # project on the unit ball, we actually end up slightly inside the unit
        # ball. Without, we may end up slightly outside.
        dtype = getattr(self.domain, 'dtype', float)
        eps = np.finfo(dtype).resolution * 10

        class NuclearNormProximal(Operator):
            """Proximal operator of `NuclearNorm`."""
            def __init__(self, sigma):
                self.sigma = float(sigma)
                Operator.__init__(self, func.domain, func.domain, linear=False)

            def _call(self, x):
                """Return ``self(x)``."""
                arr = func._asarray(x)

                # Compute SVD
                U, s, Vt = np.linalg.svd(arr, full_matrices=False)

                # transpose pointwise
                V = Vt.swapaxes(-1, -2)

                # Take pseudoinverse of s
                sinv = s.copy()
                sinv[sinv != 0] = 1 / sinv[sinv != 0]

                # Take pointwise proximal operator of s w.r.t. the norm
                # on the singular vectors
                if func.pwisenorm.exponent == 1:
                    abss = np.abs(s) - (self.sigma - eps)
                    sprox = np.sign(s) * np.maximum(abss, 0)
                elif func.pwisenorm.exponent == 2:
                    s_reordered = func._moveaxis(s, -1, 0)
                    snorm = func.pwisenorm(s_reordered).asarray()
                    snorm = np.maximum(self.sigma, snorm, out=snorm)
                    sprox = ((1 - eps) - self.sigma / snorm)[..., None] * s
                elif func.pwisenorm.exponent == np.inf:
                    snorm = np.sum(np.abs(s), axis=-1)
                    snorm = np.maximum(self.sigma, snorm, out=snorm)
                    sprox = ((1 - eps) - self.sigma / snorm)[..., None] * s
                else:
                    raise RuntimeError

                # Compute s matrix
                sproxsinv = (sprox * sinv)[..., :, None]

                # Compute the final result
                result = nddot(nddot(arr, V), sproxsinv * Vt)

                # Cast to vector and return. Note array and vector have
                # different shapes.
                return func._asvector(result)

            def __repr__(self):
                """Return ``repr(self)``."""
                return '{!r}.proximal({})'.format(func, self.sigma)

        return NuclearNormProximal

    @property
    def convex_conj(self):
        """Convex conjugate of the nuclear norm.

        The convex conjugate is the indicator function on the unit ball of
        the dual norm where the dual norm is obtained by taking the conjugate
        exponent of both the outer and singular vector exponents.
        """
        return IndicatorNuclearNormUnitBall(
            self.domain,
            conj_exponent(self.outernorm.exponent),
            conj_exponent(self.pwisenorm.exponent))

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {}, {})'.format(self.__class__.__name__,
                                         self.domain,
                                         self.outernorm.exponent,
                                         self.pwisenorm.exponent)


class IndicatorNuclearNormUnitBall(Functional):
    """Indicator on unit ball of nuclear norm for matrix valued functions.

    Notes
    -----
    For a matrix-valued function
    :math:`f : \\Omega \\rightarrow \\mathbb{R}^{n \\times m}`,
    the nuclear norm with parameters :math:`p` and :math:`q` is defined by

    .. math::
        \\left( \int_\Omega \|\sigma(f(x))\|_p^q d x \\right)^{1/q},

    where :math:`\sigma(f(x))` is the vector of singular values of the matrix
    :math:`f(x)` and :math:`\| \cdot \|_p` is the usual :math:`p`-norm on
    :math:`\mathbb{R}^{\min(n, m)}`.

    This function is defined as the indicator on the unit ball of the nuclear
    norm, that is, 0 if the nuclear norm is less than 1, and infinity else.

    For a detailed description of its properties, e.g, its proximal, convex
    conjugate and more, see [Du+2016]_.

    References
    ----------
    J. Duran, M. Moeller, C. Sbert, and D. Cremers. Collaborative Total
    Variation: A General Framework for Vectorial TV Models, SIAM Journal of
    Imaging Sciences 9(1): 116--151, 2016.
    """

    def __init__(self, space, outer_exp=1, singular_vector_exp=2):
        """Initialize a new instance.

        Parameters
        ----------
        space : `ProductSpace` of `ProductSpace` of `FnBase`
            Domain of the functional.
        outer_exp : {1, 2, inf}, optional
            Exponent for the outer norm.
        singular_vector_exp : {1, 2, inf}, optional
            Exponent for the norm for the singular vectors.

        Examples
        --------
        Simple example, nuclear norm of matrix valued function with all ones
        in 3 points. The singular values are [2, 0], which has squared 2-norm
        2. Since there are 3 points, the expected total value is 6.
        Since the nuclear norm is larger than 1, the indicator is infinity.

        >>> r3 = odl.rn(3)
        >>> space = odl.ProductSpace(odl.ProductSpace(r3, 2), 2)
        >>> norm = IndicatorNuclearNormUnitBall(space)
        >>> norm(space.one())
        inf
        """
        self.__norm = NuclearNorm(space, outer_exp, singular_vector_exp)
        super().__init__(space=space, linear=False, grad_lipschitz=np.nan)

    def _call(self, x):
        """Return ``self(x)``."""
        x_norm = self.__norm(x)

        if x_norm > 1:
            return np.inf
        else:
            return 0

    @property
    def proximal(self):
        """The proximal operator."""
        # Implement proximal via duality
        return proximal_cconj(self.convex_conj.proximal)

    @property
    def convex_conj(self):
        """Convex conjugate of the unit ball indicator of the  nuclear norm.

        The convex conjugate is the dual nuclear norm  where the dual norm is
        obtained by taking the conjugate exponent of both the outer and
        singular vector exponents.
        """
        return NuclearNorm(self.domain,
                           conj_exponent(self.__norm.outernorm.exponent),
                           conj_exponent(self.__norm.pwisenorm.exponent))

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {}, {})'.format(self.__class__.__name__,
                                         self.domain,
                                         self.__norm.outernorm.exponent,
                                         self.__norm.pwisenorm.exponent)


class MoreauEnvelope(Functional):

    """Moreau envelope of a convex functional.

    The Moreau envelope is a way to smooth an arbitrary convex functional
    such that its gradient can be computed given the proximal of the original
    functional.
    The new functional has the same critical points as the original.
    It is also called the Moreau-Yosida regularization.

    Note that the only computable property of the Moreau envelope is the
    gradient, the functional itself cannot be evaluated efficiently.

    See `Proximal Algorithms`_ for more information.

    Notes
    -----
    The Moreau envelope of a convex functional
    :math:`f : \mathcal{X} \\rightarrow \mathbb{R}` multiplied by a scalar
    :math:`\\sigma` is defined by

    .. math::
        \mathrm{env}_{\\sigma  f}(x) =
        \\inf_{y \\in \\mathcal{X}}
        \\left\{ \\frac{1}{2 \\sigma} \| x - y \|_2^2 + f(y) \\right\}

    The gradient of the envelope is given by

    .. math::
        [\\nabla \mathrm{env}_{\\sigma  f}](x) =
        \\frac{1}{\\sigma} (x - \mathrm{prox}_{\\sigma  f}(x))

    Example: if :math:`f = \| \cdot \|_1`, then

    .. math::
        [\mathrm{env}_{\\sigma  \| \cdot \|_1}(x)]_i =
        \\begin{cases}
            \\frac{1}{2 \\sigma} x_i^2 & \\text{if } |x_i| \leq \\sigma \\\\
            |x_i| - \\frac{\\sigma}{2} & \\text{if } |x_i| > \\sigma,
        \\end{cases}

    which is the usual Huber functional.

    References
    ----------
    .. _Proximal Algorithms: \
https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
    """

    def __init__(self, functional, sigma=1.0):
        """Initialize an instance.

        Parameters
        ----------
        functional : `Functional`
            The functional ``f`` in the definition of the Moreau envelope that
            is to be smoothed.
        sigma : positive float
            The scalar ``sigma`` in the definition of the Moreau envelope.
            Larger values mean stronger smoothing.

        Examples
        --------
        Create smoothed l1 norm:

        >>> space = odl.rn(3)
        >>> l1_norm = odl.solvers.L1Norm(space)
        >>> smoothed_l1 = MoreauEnvelope(l1_norm)
        """
        self.__functional = functional
        self.__sigma = sigma
        super().__init__(space=functional.domain,
                         linear=False)

    @property
    def functional(self):
        """The functional that has been regularized."""
        return self.__functional

    @property
    def sigma(self):
        """Regularization constant, larger means stronger regularization."""
        return self.__sigma

    @property
    def gradient(self):
        """The gradient operator."""
        return (ScalingOperator(self.domain, 1 / self.sigma) -
                (1 / self.sigma) * self.functional.proximal(self.sigma))


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
