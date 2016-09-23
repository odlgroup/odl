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

from odl.solvers.functional.functional import Functional
from odl.operator.operator import Operator
from odl.solvers.nonsmooth.proximal_operators import (
    proximal_l1, proximal_cconj_l1, proximal_l2, proximal_cconj_l2,
    proximal_l2_squared, proximal_const_func, proximal_box_constraint,
    proximal_cconj, proximal_cconj_kl, proximal_cconj_kl_cross_entropy,
    combine_proximals)
from odl.operator import (ZeroOperator, ScalingOperator, DiagonalOperator)
from odl.space import ProductSpace


__all__ = ('L1Norm', 'L2Norm', 'L2NormSquared', 'ZeroFunctional',
           'ConstantFunctional', 'IndicatorLpUnitBall', 'IndicatorBox',
           'IndicatorNonnegativity', 'KullbackLeibler',
           'KullbackLeiblerCrossEntropy', 'SeparableSum')


class L1Norm(Functional):

    """The functional corresponding to L1-norm.

    The L1-norm, ``||x||_1``,  is defined as the integral/sum of ``|x|``.

    Notes
    -----
    If the functional is defined on an :math:`\mathbb{R}^n`-like space, the
    :math:`L_1`-norm is defined as

    .. math::

        \| x \|_1 = \\sum_{i = 1}^n |x_i|.

    If the functional is defined on an :math:`L_2`-like space, the
    :math:`L_1`-norm is defined as

    .. math::

        \| x \|_1 = \\int_{\Omega} |x| dx.
    """

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `FnBase`
            Domain of the functional.
        """
        super().__init__(space=space, linear=False, grad_lipschitz=np.nan)

    # TODO: update when integration operator is in place: issue #440
    def _call(self, x):
        """Return the L1-norm of ``x``."""
        return x.ufunc.absolute().inner(self.domain.one())

    # TODO: remove inner class when ufunc operators are in place: issue #567
    @property
    def gradient(self):
        """Gradient operator of the functional.

        The functional is not differentiable in ``x=0``. However, when
        evaluating the gradient operator in this point it will return 0.
        """
        functional = self

        class L1Gradient(Operator):

            """The gradient operator of this functional."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)

            def _call(self, x):
                """Apply the gradient operator to the given point."""
                return x.ufunc.sign()

        return L1Gradient()

    @property
    def proximal(self):
        """Return the proximal factory of the functional.

        See Also
        --------
        proximal_l1 : proximal factory for the L1-norm.
        """
        return proximal_l1(space=self.domain)

    @property
    def convex_conj(self):
        """The convex conjugate functional of the L1-norm."""
        return IndicatorLpUnitBall(self.domain, exponent=np.inf)


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
        self.__exponent = float(exponent)

    @property
    def exponent(self):
        """Exponent corresponding to the norm."""
        return self.__exponent

    # TODO: update when integration operator is in place: issue #440
    def _call(self, x):
        """Apply the functional to the given point."""
        if self.exponent == 1:
            x_norm = x.ufunc.absolute().inner(self.domain.one())
        elif self.exponent == 2:
            x_norm = x.norm()
        elif self.exponent == np.inf:
            x_norm = x.ufunc.absolute().ufunc.max()
        else:
            tmp = x.ufunc.absolute()
            tmp.ufunc.power(self.exponent, out=tmp)
            x_norm = np.power(tmp.inner(self.domain.one()), 1 / self.exponent)

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
            # TODO: Add Lp-norm functional.
            raise NotImplementedError('currently not implemented')

    @property
    def proximal(self):
        """Return the proximal factory of the functional.

        See Also
        --------
        proximal_cconj_l1 : proximal factory for convex conjuagte of L1-norm.
        proximal_cconj_l2 : proximal factory for convex conjuagte of L2-norm.
        """
        if self.exponent == np.inf:
            return proximal_cconj_l1(space=self.domain)
        elif self.exponent == 2:
            return proximal_cconj_l2(space=self.domain)
        else:
            raise NotImplementedError('currently not implemented')


class L2Norm(Functional):

    """The functional corresponding to the L2-norm.

    The L2-norm, ``||x||_2``,  is defined as the square-root out of the
    integral/sum of ``x^2``.
    """

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `FnBase`
            Domain of the functional.
        """
        super().__init__(space=space, linear=False, grad_lipschitz=np.nan)

    # TODO: update when integration operator is in place: issue #440
    def _call(self, x):
        """Return the L2-norm of ``x``."""
        return np.sqrt(x.inner(x))

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        functional = self

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
                    # The derivative is not defined for 0.
                    raise ValueError('The gradient of the L2 functional is '
                                     'not defined for the zero element.')
                else:
                    return x / norm_of_x

        return L2Gradient()

    @property
    def proximal(self):
        """Return the proximal factory of the functional.

        See Also
        --------
        proximal_l2 : proximal factory for L2-norm.
        """
        return proximal_l2(space=self.domain)

    @property
    def convex_conj(self):
        """The convex conjugate functional of the L2-norm."""
        return IndicatorLpUnitBall(self.domain, exponent=2)


class L2NormSquared(Functional):

    """The functional corresponding to the squared L2-norm.

    The squared L2-norm, ``||x||_2^2``,  is defined as the integral/sum of
    ``x^2``.
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
        """Return the proximal factory of the functional.

        See Also
        --------
        proximal_l2_squared : proximal factory for squared L2-norm.
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
        """Return the proximal factory of the functional."""
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
        functional = self

        class ConstantFunctionalConvexConj(Functional):

            """The convex conjugate functional of the constant functional.

            It does not implement `gradient` since it is not differentiable
            anywhere.
            """

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, linear=False)

            def _call(self, x):
                """Apply the functional to the given point."""

                if x.norm() == 0:
                    # In this case x is the zero-element.
                    return -functional.constant
                else:
                    return np.inf

            @property
            def convex_conj(self):
                """The convex conjugate functional.

                Notes
                -----
                By the Fenchel-Moreau theorem the convex conjugate functional
                is the constant functional [BC2011]_.
                """
                return functional

            @property
            def proximal(self):
                """Return the proximal factory of the functional.

                This is the zero operator.
                """
                def zero_proximal(sigma=1.0):
                    """Proximal factory for zero operator.

                    Parameters
                    ----------
                    sigma : positive float
                        Step size parameter. Default: 1.0"""
                    return ZeroOperator(self.domain)

                return zero_proximal

        return ConstantFunctionalConvexConj()


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
        >>> import odl
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
            x.ufunc.maximum(self.lower, out=tmp)
        elif self.lower is None and self.upper is not None:
            x.ufunc.minimum(self.upper, out=tmp)
        elif self.lower is not None and self.upper is not None:
            x.ufunc.maximum(self.lower, out=tmp)
            tmp.ufunc.minimum(self.upper, out=tmp)
        else:
            tmp.assign(x)

        return np.inf if x.dist(tmp) > 0 else 0

    @property
    def proximal(self):
        """Return the proximal factory of the functional."""
        return proximal_box_constraint(self.domain, self.lower, self.upper)


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
        >>> import odl
        >>> space = odl.rn(3)
        >>> func = IndicatorNonnegativity(space)
        >>> func([0, 1, 2])  # all points positive
        0
        >>> func([0, 1, -3])  # one point negative
        inf
        """
        IndicatorBox.__init__(self, space, lower=0, upper=None)


class KullbackLeibler(Functional):

    """The Kullback-Leibler divergence functional.
    Notes
    -----
    The functional :math:`F` is given by
    .. math::
        \\sum_{i} \left( x_i - g_i + g_i \log \left( \\frac{g_i}{ pos(x_i) }
        \\right) \\right) + I_{x \\geq 0}(x)
    where :math:`g` is the prior, and :math:`I_{x \\geq 0}(x)` is the indicator
    function on nonnegative elements.
    KL based objectives are common in MLEM optimization problems and are often
    used as data-matching term when data noise governed by a multivariate
    Poisson probability distribution is significant.
    The functional is related to the Kullback-Leibler cross entropy functional
    `KullbackLeiblerCrossEntropy`. The KL cross entropy is the one
    diescribed in `this Wikipedia article
    <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_, and
    the functional :math:`F` is obtained by switching place of the prior and
    the varialbe in the KL cross entropy functional. See the See Also section.
    See Also
    --------
    KullbackLeiblerConvexConj : the convex conjugate functional
    KullbackLeiblerCrossEntropy : related functional
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
        """Return the proximal factory of the functional.
        See Also
        --------
        proximal_cconj_kl : proximal factory for convex conjugate of KL.
        proximal_cconj : proximal of the convex conjugate of a functional
        """
        return proximal_cconj(proximal_cconj_kl(space=self.domain,
                                                g=self.prior))

    @property
    def convex_conj(self):
        """The convex conjugate functional of the KL-functional."""
        return KullbackLeiblerConvexConj(self.domain, self.prior)


class KullbackLeiblerConvexConj(Functional):

    """The convex conjugate of Kullback-Leibler divergence functional.
    Notes
    -----
    The functional is given by
    .. math::
        \\sum_i \left(-g_i \ln(pos({1_X}_i - x_i)) \\right) +
        I_{1_X - x \geq 0}(x)
    where :math:`g` is the prior, and :math:`I_{1_X - x \geq 0}(x)` is the
    indicator function on :math:`x \leq 1`.
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
        """Return the proximal factory of the functional.
        See Also
        --------
        proximal_cconj_kl : proximal factory for convex conjugate of KL.
        proximal_cconj : proximal of the convex conjugate of a functional
        """
        return proximal_cconj_kl(space=self.domain, g=self.prior)

    @property
    def convex_conj(self):
        """The convex conjugate functional of the conjugate KL-functional."""
        return KullbackLeibler(self.domain, self.prior)


class KullbackLeiblerCrossEntropy(Functional):

    """The Kullback-Leibler Cross Entropy divergence functional.
    Notes
    -----
    The functional :math:`F` is given by
    .. math::
        \\sum_{i} \left( g_i - x_i + x_i \log \left( \\frac{x_i}{ pos(g_i) }
        \\right) \\right) + I_{x \\geq 0}(x)
    where :math:`g` is the prior, and :math:`I_{x \\geq 0}(x)` is the indicator
    function on nonnegative elements.
    `Wikipedia article on Kullback Leibler divergence
    <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_.
    For further information about the functional, see for example `this article
    <http://ieeexplore.ieee.org/document/1056144/?arnumber=1056144>`_.
    The KL cross entropy functional :math:`F`, described above, is related to
    another functional which is also know as KL divergence. This functional
    is often used as data discrepancy term in inverse problems, when data is
    corrupted with Poisson noise. This functional is obtained by changing place
    of the prior and the variable. See the See Also section.
    See Also
    --------
    KullbackLeibler : related functional
    KullbackLeiblerCrossEntropyConvexConj : the convex conjugate functional
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
            tmp = ((1 - x + x * np.log(x)).inner(self.domain.one()))
        else:
            tmp = ((self.prior - x + x * np.log(x / self.prior))
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
        """Return the proximal factory of the functional.
        See Also
        --------
        proximal_cconj_kl_cross_entropy : proximal factory for convex conjugate
        of KL cross entropy.
        proximal_cconj : proximal of the convex conjugate of a functional
        """
        return proximal_cconj(proximal_cconj_kl_cross_entropy(
            space=self.domain, g=self.prior))

    @property
    def convex_conj(self):
        """The convex conjugate functional of the KL-functional."""
        return KullbackLeiblerCrossEntropyConvexConj(self.domain, self.prior)


class KullbackLeiblerCrossEntropyConvexConj(Functional):
    """The convex conjugate of Kullback-Leibler Cross Entorpy functional.
    Notes
    -----
    The functional is given by
    .. math::
        \\sum_i g_i (exp(x_i) - 1)
    where :math:`g` is the prior.
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
        """Return the proximal factory of the functional.
        See Also
        --------
        proximal_cconj_kl_cross_entropy : proximal factory for convex conjugate
        of KL cross entropy.
        """
        return proximal_cconj_kl_cross_entropy(space=self.domain, g=self.prior)

    @property
    def convex_conj(self):
        """The convex conjugate functional of the conjugate KL-functional."""
        return KullbackLeiblerCrossEntropy(self.domain, self.prior)


class SeparableSum(Functional):

    """The functional corresponding to separable sum of functionals.

    The separable sum of functionals ``f_1, f_2, ..., f_n`` is given by::

        h(x_1, x_2, ..., x_n) = sum_i^n f_i(x_i)

    The separable sum is thus defined for any collection of functionals with
    the same range.
    """

    def __init__(self, *functionals):
        """Initialize a new instance.

        Parameters
        ----------
        functionals1, ..., functionalsN : `Functional`
            The functionals in the sum.
        """
        domains = [func.domain for func in functionals]
        domain = ProductSpace(*domains)
        linear = all(func.is_linear for func in functionals)

        self.functionals = functionals

        super().__init__(space=domain, linear=linear)

    def _call(self, x):
        """Return the squared L2-norm of ``x``."""
        return sum(fi(xi) for xi, fi in zip(x, self.functionals))

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        gradients = [func.gradient for func in self.functionals]
        return DiagonalOperator(*gradients)

    @property
    def proximal(self):
        """Return the proximal factory of the functional.

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


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
