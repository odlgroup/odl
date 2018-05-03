# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Default functionals defined on any space similar to R^n or L^2."""

from __future__ import absolute_import, division, print_function

from numbers import Integral

import numpy as np

from odl.operator import (
    DiagonalOperator, Operator, PointwiseNorm, ScalingOperator, ZeroOperator)
from odl.solvers.functional.functional import (
    Functional, FunctionalQuadraticPerturb)
from odl.solvers.nonsmooth.proximal_operators import (
    proximal_separable_sum, proximal_indicator_box, proximal_const_func,
    proximal_convex_conj, proximal_convex_conj_kl,
    proximal_convex_conj_kl_cross_entropy, proximal_indicator_linf_unit_ball,
    proximal_indicator_linf_l2_unit_ball, proximal_indicator_l2_unit_ball,
    proximal_huber, proximal_l1, proximal_l1_l2, proximal_l2)
from odl.space import ProductSpace
from odl.util import (
    REPR_PRECISION, conj_exponent, moveaxis, npy_printoptions, repr_string,
    signature_string_parts, attribute_repr_string)

__all__ = ('ZeroFunctional', 'ConstantFunctional', 'ScalingFunctional',
           'IdentityFunctional',
           'LpNorm', 'L1Norm', 'GroupL1Norm', 'L2Norm', 'L2NormSquared',
           'Huber', 'NuclearNorm',
           'IndicatorZero', 'IndicatorBox', 'IndicatorNonnegativity',
           'IndicatorLpUnitBall', 'IndicatorGroupLinfUnitBall',
           'IndicatorNuclearNormUnitBall',
           'KullbackLeibler', 'KullbackLeiblerCrossEntropy',
           'QuadraticForm',
           'SeparableSum', 'MoreauEnvelope')


class LpNorm(Functional):

    r"""The p-norm as a functional.

    This functional is defined as

    .. math::
        \| x \|_p &= \left(\sum_{i=1}^n |x_i|^p \right)^{1/p}
        \quad (\mathbb{R^n-}\text{like space}) \\
        \| x \|_p &= \left(\int |x(t)|^p \mathrm{d}t \right)^{1/p}
        \quad (L^p-\text{like space}).

    If :math:`n` is a multi-index, i.e., :math:`\mathbb{R}^n` is a tensor
    space, the above definition is applied to the flattened array.
    """

    def __init__(self, space, exponent):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `TensorSpace`
            Domain of the functional.
        exponent : float
            Exponent ``p`` of the norm.

        Examples
        --------
        >>> space = odl.rn(2)
        >>> l2norm = odl.solvers.LpNorm(space, exponent=2)
        >>> l2norm([3, 4])
        5.0
        >>> l1norm = odl.solvers.LpNorm(space, exponent=1)
        >>> l1norm([3, 4])
        7.0
        """
        super(LpNorm, self).__init__(
            space=space, linear=False, grad_lipschitz=np.nan)

        self.__exponent = float(exponent)

    @property
    def exponent(self):
        """The exponent p of the p-norm."""
        return self.__exponent

    # TODO(#440): update when integration operator is in place
    def _call(self, x):
        """Return ``self(x)``."""
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
        r"""Convex conjugate of the p-norm.

        The convex conjugate of a norm is the indicator function of
        the unit ball in the dual norm,

        .. math::
            B_{\|\cdot\|_*} = \{ t\,|\, \|t\|_* \leq 1 \},

        which takes value 0 inside the set and :math:`\infty` outside.
        The dual norm :math:`\|\cdot\|_*` is the :math:`q`-norm with the
        conjugate exponent :math:`q = p / (p - 1)` (with :math:`q = \infty`
        for :math:`p = 1` and vice versa).

        See Also
        --------
        IndicatorLpUnitBall
        """
        return IndicatorLpUnitBall(self.domain,
                                   exponent=conj_exponent(self.exponent))

    @property
    def proximal(self):
        """The `proximal factory` of the p-norm.

        See Also
        --------
        odl.solvers.nonsmooth.proximal_operators.proximal_l1 :
            proximal factory for the L1-norm.
        odl.solvers.nonsmooth.proximal_operators.proximal_l2 :
            proximal factory for the L2-norm.
        """
        if self.exponent == 1:
            return proximal_l1(self.domain)
        elif self.exponent == 2:
            return proximal_l2(self.domain)
        else:
            raise NotImplementedError('`proximal` only implemented for p=1 or '
                                      'p=2')

    @property
    def gradient(self):
        r"""Gradient operator of the p-norm.

        The functional is not differentiable in ``x=0``. This implementation
        evaluates to 0 in that case.

        The gradient of :math:`\|\cdot\|_p` is given as:

        - :math:`p = 1`:

          .. math::
              \nabla \|\cdot\|_1(x) &= \big[\text{sign}(x_i)\big]_i
              \quad (\mathbb{R^n-}\text{like space}) \\
              \nabla \|\cdot\|_1(x) &= t \mapsto \text{sign}\big(x(t)\big)
              \quad (L^p-\text{like space})

        - :math:`p = 2`:

          .. math::
              \nabla \|\cdot\|_2(x) &= \left[\frac{x_i}{\|x\|_2}\right]_i
              \quad (\mathbb{R^n-}\text{like space}) \\
              \nabla \|\cdot\|_2(x) &= t \mapsto \frac{x(t)}{\|x\|_2}
              \quad (L^p-\text{like space})

        - otherwise:

          .. math::
              \nabla \|\cdot\|_p(x) &= \left[
                  \frac{|x_i|^{p-2}\, x_i}{\|x\|_p^{p - 1}}\right]_i
              \quad (\mathbb{R^n-}\text{like space}) \\
              \nabla \|\cdot\|_p(x) &= t \mapsto
                  \frac{|x(t)|^{p-2}\, x(t)}{\|x\|_p^{p-1}}
              \quad (L^p-\text{like space})

        .. note::
            The gradient is currently only implemented for ``p == 1`` and
            ``p == 2``.
        """
        functional = self

        if self.exponent == 1:
            class L1Gradient(Operator):

                """The gradient operator of this functional."""

                def _call(self, x):
                    """Return ``self(x)``."""
                    return x.ufuncs.sign()

                def derivative(self, x):
                    """Return the derivative operator.

                    The derivative is zero almost everywhere, hence this
                    implementation yields the zero operator.
                    """
                    return ZeroOperator(self.domain)

                def __repr__(self):
                    """Return ``repr(self)``.

                    Examples
                    --------
                    >>> space = odl.rn(2)
                    >>> l1norm = odl.solvers.LpNorm(space, exponent=1)
                    >>> l1norm.gradient
                    LpNorm(rn(2), exponent=1.0).gradient
                    """
                    return attribute_repr_string(repr(functional), 'gradient')

            return L1Gradient(self.domain, self.domain, linear=False)

        elif self.exponent == 2:
            class L2Gradient(Operator):

                """The gradient operator of this functional."""

                def _call(self, x):
                    """Return ``self(x)``."""
                    norm_of_x = x.norm()
                    if norm_of_x == 0:
                        return self.domain.zero()
                    else:
                        return x / norm_of_x

                def __repr__(self):
                    """Return ``repr(self)``.

                    Examples
                    --------
                    >>> space = odl.rn(2)
                    >>> l2norm = odl.solvers.LpNorm(space, exponent=2)
                    >>> l2norm.gradient
                    LpNorm(rn(2), exponent=2.0).gradient
                    """
                    return attribute_repr_string(repr(functional), 'gradient')

            return L2Gradient(self.domain, self.domain, linear=False)

        else:
            raise NotImplementedError(
                '`gradient` only implemented for `p == 1` and `p == 2`')

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> space = odl.rn(2)
        >>> l1norm = odl.solvers.LpNorm(space, exponent=1)
        >>> l1norm
        LpNorm(rn(2), exponent=1.0)
        """
        posargs = [self.domain]
        optargs = [('exponent', self.exponent, None)]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


class IndicatorLpUnitBall(Functional):

    r"""Indicator functional on the unit ball in the p-norm.

    This functional is defined as

    .. math::
        \iota_{B_p}(x) =
        \begin{cases}
            0      & \text{if } \|x\|_p \leq 1, \\
            \infty & \text{else,}
        \end{cases}

    where :math:`B_p` is the unit ball in the p-norm.

    See Also
    --------
    LpNorm
    """

    def __init__(self, space, exponent):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `TensorSpace`
            Domain of the functional.
        exponent : int or infinity
            Specifies wich norm to use.
        """
        super(IndicatorLpUnitBall, self).__init__(space=space, linear=False)
        self.__norm = LpNorm(space, exponent)
        self.__exponent = float(exponent)

    @property
    def exponent(self):
        """Exponent corresponding to the norm."""
        return self.__exponent

    def _call(self, x):
        """Return ``self(x)``."""
        x_norm = self.__norm(x)

        if x_norm > 1:
            return np.inf
        else:
            return 0

    @property
    def convex_conj(self):
        r"""Convex conjugate of the indicator of the p-norm unit ball.

        The convex conjugate of the indicator function of a norm ball,
        that is a function taking the value 0 inside and :math:`\infty`
        outside, is the dual norm :math:`\|\cdot\|_*`, in this case the
        Lp-norm with conjugate exponent :math:`q = p / (p - 1)`
        (:math:`1` and :math:`\infty` are conjugate to each other).

        See Also
        --------
        LpNorm
        """
        if self.exponent == np.inf:
            return L1Norm(self.domain)
        elif self.exponent == 2:
            return L2Norm(self.domain)
        else:
            return LpNorm(self.domain, exponent=conj_exponent(self.exponent))

    @property
    def proximal(self):
        r"""The `proximal factory` of the functional.

        The proximal operator of an indicator function of a set :math:`B`
        is the orthogonal projection

        .. math::
            P_B(x) = \mathrm{arg\,min}_{y \in B} \|x - y\|_2.

        .. note::
            The proximal operator is currently only implemented for ``p == 1``
            and ``p == 2``.

        See Also
        --------
        proximal_convex_conj_l1 :
            `proximal factory` for convex conjuagte of L1-norm.
        proximal_indicator_l2_unit_ball :
            `proximal factory` for convex conjuagte of L2-norm.
        """
        if self.exponent == np.inf:
            return proximal_indicator_linf_unit_ball(self.domain)
        elif self.exponent == 2:
            return proximal_indicator_l2_unit_ball(self.domain)
        else:
            raise NotImplementedError('`gradient` only implemented for p=2 or '
                                      'p=inf')

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> space = odl.rn(2)
        >>> op = odl.solvers.IndicatorLpUnitBall(space, exponent=1)
        >>> op
        IndicatorLpUnitBall(rn(2), exponent=1.0)
        """
        posargs = [self.domain]
        optargs = [('exponent', self.exponent, None)]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


# TODO(kohr-h): make work for `TensorSpace`
class GroupL1Norm(Functional):

    r"""The mixed L1-Lp norm (or cross norm) for vector-valued functions.

    On an :math:`\mathbb{R}^{m \times n}`-like space, the group
    :math:`L^1` norm, denoted :math:`\| \cdot \|_{\times, p}` is defined as

    .. math::
        \|\mathbf{x}\|_{\times, p} =
        \sum_{j=1}^n \left(\sum_{i=1}^m |x_{i,j}|^p\right)^{1/p}

    On an :math:`(L^p)^m`-like space, norm is given as

    .. math::
        \|\mathbf{x}\|_{\times, p} =
        \int \left(\sum_{i=1}^m |x_i(x)|^p\right)^{1/p}
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
        >>> pspace = odl.rn(2) ** 2
        >>> op = odl.solvers.GroupL1Norm(pspace)
        >>> op([[3, 3], [4, 4]])
        10.0

        Set exponent of inner (p) norm:

        >>> op2 = odl.solvers.GroupL1Norm(pspace, exponent=1)
        >>> op2([[3, 3], [4, 4]])
        14.0
        """
        if not isinstance(vfspace, ProductSpace):
            raise TypeError('`space` must be a `ProductSpace`')
        if not vfspace.is_power_space:
            raise TypeError('`space.is_power_space` must be `True`')

        super(GroupL1Norm, self).__init__(
            space=vfspace, linear=False, grad_lipschitz=np.nan)
        self.__pointwise_norm = PointwiseNorm(vfspace, exponent=exponent)

    @property
    def pointwise_norm(self):
        """Operator for computing the norm in each point."""
        return self.__pointwise_norm

    def _call(self, x):
        """Return ``self(x)``."""
        # TODO(#440): use integration operator when available
        pointwise_norm = self.pointwise_norm(x)
        return pointwise_norm.inner(pointwise_norm.space.one())

    @property
    def gradient(self):
        r"""Gradient operator of the functional.

        The functional is not differentiable in ``x=0``. This implementation
        sets evaluates to 0 in that case.

        The gradient of :math:`\|\cdot\|_{\times, p}` is given as:

        - :math:`p = 1` (equivalent to the :math:`L^1` norm):

          .. math::
              \nabla \|\cdot\|_{\times, 1}(\mathbf{x}) &= \big[
                  \text{sign}(x_{i,j})\big]_{i,j}
              \quad (\mathbb{R^{m \times n}-}\text{like space}) \\
              \nabla \|\cdot\|_{\times, 1}(\mathbf{x}) &= t \mapsto
                  \big[\text{sign}\big(x_i(t)\big)\big]_i
              \quad ((L^p)^m-\text{like space})

        - :math:`p = 2`:

          .. math::
              \nabla \|\cdot\|_{\times, 2}(\mathbf{x}) &= \left[
                  \frac{x_{i,j}}{\|\mathbf{x}_j\|_2}\right]_{i,j}
              \quad (\mathbb{R^{m \times n}-}\text{like space}) \\
              \nabla \|\cdot\|_{\times, 2}(\mathbf{x}) &= t \mapsto
                  \left[\frac{x_i(t)}{\|\mathbf{x}(t)\|_2}\right]_i
              \quad ((L^p)^m-\text{like space})

        - otherwise:

          .. math::
              \nabla \|\cdot\|_{\times, p}(\mathbf{x}) &= \left[
                  \frac{|x_{i,j}|^{p-2}\, x_{i,j}}{\|\mathbf{x}_j\|_p^{p-1}}
              \right]_{i,j}
              \quad (\mathbb{R^{m \times n}-}\text{like space}) \\
              \nabla \|\cdot\|_{\times, p}(\mathbf{x}) &= t \mapsto
                  \left[\frac{|x_i(t)|^{p-2}\, x_i(t)}{
                              \|\mathbf{x}(t)\|_p^{p-1}}
              \right]_i
              \quad ((L^p)^m-\text{like space})
        """
        functional = self

        class GroupL1Gradient(Operator):

            """The gradient operator of the `GroupL1Norm` functional."""

            def _call(self, x, out):
                """Return ``self(x)``."""
                pwnorm_x = functional.pointwise_norm(x)
                pwnorm_x.ufuncs.sign(out=pwnorm_x)
                functional.pointwise_norm.derivative(x).adjoint(pwnorm_x,
                                                                out=out)
                return out

            def __repr__(self):
                """Return ``repr(self)``.

                Examples
                --------
                >>> pspace = odl.rn(2) ** 2
                >>> op = odl.solvers.GroupL1Norm(pspace)
                >>> op.gradient
                GroupL1Norm(ProductSpace(rn(2), 2), exponent=2.0).gradient
                """
                return attribute_repr_string(repr(functional), 'gradient')

        return GroupL1Gradient(self.domain, self.domain, linear=False)

    @property
    def proximal(self):
        """The ``proximal factory`` of the functional.

        .. note::
            The proximal operator is currently only implemented for ``p == 1``
            and ``p == 2``.

        See Also
        --------
        proximal_l1 : `proximal factory` for the L1 norm.
        proximal_l1_l2 : `proximal factory` for the L1-L2 norm.
        """
        if self.pointwise_norm.exponent == 1:
            return proximal_l1(self.domain)
        elif self.pointwise_norm.exponent == 2:
            return proximal_l1_l2(self.domain)
        else:
            raise NotImplementedError('`proximal` only implemented for p = 1 '
                                      'or 2')

    @property
    def convex_conj(self):
        """Convex conjugate of the group L1 norm.

        The convex conjugate of a norm is the indicator function of
        the unit ball with respect to the dual norm,

        .. math::
            B_{\|\cdot\|_*} = \{ t\,|\, \|t\|_* \leq 1 \},

        which takes value 0 inside the set and :math:`\infty` outside.

        See Also
        --------
        IndicatorGroupLinfUnitBall
        """
        conj_exp = conj_exponent(self.pointwise_norm.exponent)
        return IndicatorGroupLinfUnitBall(self.domain, exponent=conj_exp)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> pspace = odl.rn(2) ** 2
        >>> l1_2_norm = odl.solvers.GroupL1Norm(pspace, exponent=2)
        >>> l1_2_norm
        GroupL1Norm(ProductSpace(rn(2), 2), exponent=2.0)
        """
        posargs = [self.domain]
        optargs = [('exponent', self.pointwise_norm.exponent, None)]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


class IndicatorGroupLinfUnitBall(Functional):

    r"""Indicator functional of the unit ball in the group L^inf norm.

    This functional is defined as

    .. math::
        \iota_{B_{\infty, p}}(\mathbf{x}) =
        \begin{cases}
            0      & \text{if } \|\mathbf{x}\|_{\times, p} \leq 1, \\
            \infty & \text{else,}
        \end{cases}

    where :math:`B_{\infty, p}` is the unit ball in the norm

    .. math::
        \|\mathbf{x}\|_{\infty, p} = \sup_t |\mathbf{x}(t)|_p,

    and the p-norm :math:`|\mathbf{x}(t)|_p` is taken along the components
    of :math:`\mathbf{x}`.

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
            It has to be a product space of identical spaces, i.e., a
            power space.
        exponent : non-zero float, optional
            Exponent of the norm in each point. Values between
            0 and 1 are currently not supported due to numerical
            instability. Infinity gives the supremum norm.
            Default: ``vfspace.exponent``, usually 2.

        Examples
        --------
        >>> pspace = odl.rn(2) ** 2
        >>> op = odl.solvers.IndicatorGroupLinfUnitBall(pspace)
        >>> op([[0.1, 0.5], [0.2, 0.3]])
        0
        >>> op([[3, 3], [4, 4]])
        inf

        Set exponent of inner (p) norm:

        >>> op2 = odl.solvers.IndicatorGroupLinfUnitBall(pspace, exponent=1)
        """
        if not isinstance(vfspace, ProductSpace):
            raise TypeError('`space` must be a `ProductSpace`')
        if not vfspace.is_power_space:
            raise TypeError('`space.is_power_space` must be `True`')

        super(IndicatorGroupLinfUnitBall, self).__init__(
            space=vfspace, linear=False, grad_lipschitz=np.nan)
        self.pointwise_norm = PointwiseNorm(vfspace, exponent=exponent)

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

        .. note::
            The proximal operator is currently only implemented for
            ``p == inf`` and ``p == 2``.

        See Also
        --------
        proximal_convex_conj_l1 :
            `proximal factory` for the L1 norm's convex conjugate.
        proximal_convex_conj_l1_l2 :
            `proximal factory` for the L1-L2 norm's convex conjugate.
        """
        if self.pointwise_norm.exponent == float('inf'):
            return proximal_indicator_linf_unit_ball(self.domain)
        elif self.pointwise_norm.exponent == 2:
            return proximal_indicator_linf_l2_unit_ball(self.domain)
        else:
            raise NotImplementedError('`proximal` only implemented for '
                                      'p = inf or 2')

    @property
    def convex_conj(self):
        r"""Convex conjugate of the group L^inf unit ball indicator.

        The convex conjugate of the indicator function of a norm ball,
        that is a function taking the value 0 inside and :math:`\infty`
        outside, is the dual norm :math:`\|\cdot\|_*`, in this case the
        group L1 norm with conjugate exponent :math:`q = p / (p - 1)`
        (:math:`1` and :math:`\infty` are conjugate to each other).

        See Also
        --------
        GroupL1Norm
        """
        conj_exp = conj_exponent(self.pointwise_norm.exponent)
        return GroupL1Norm(self.domain, exponent=conj_exp)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> pspace = odl.rn(2) ** 2
        >>> op = odl.solvers.IndicatorGroupLinfUnitBall(pspace)
        >>> op
        IndicatorGroupLinfUnitBall(ProductSpace(rn(2), 2))
        """
        posargs = [self.domain]
        optargs = [('exponent', self.pointwise_norm.exponent,
                    self.domain.exponent)]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


class L1Norm(LpNorm):

    r"""The 1-norm as a functional.

    This functional is defined as

    .. math::
        \| x \|_1 &= \sum_{i=1}^n |x_i|
        \quad (\mathbb{R^n-}\text{like space}) \\
        \| x \|_1 &= \int |x(t)| \mathrm{d}t
        \quad (L^p-\text{like space}).

    If :math:`n` is a multi-index, i.e., :math:`\mathbb{R}^n` is a tensor
    space, the above definition is applied to the flattened array.

    See Also
    --------
    LpNorm
    GroupL1Norm
    """

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `TensorSpace`
            Domain of the functional.
        """
        super(L1Norm, self).__init__(space=space, exponent=1)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> space = odl.rn(2)
        >>> op = odl.solvers.L1Norm(space)
        >>> op
        L1Norm(rn(2))
        """
        posargs = [self.domain]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, [])
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


class L2Norm(LpNorm):

    r"""The 2-norm or Euclidean norm as a functional.

    This functional is defined as

    .. math::
        \| x \|_2 &= \left(\sum_{i=1}^n |x_i|^2 \right)^{1/2}
        \quad (\mathbb{R^n-}\text{like space}) \\
        \| x \|_2 &= \left(\int |x(t)|^2 \mathrm{d}t \right)^{1/2}
        \quad (L^p-\text{like space}).

    If :math:`n` is a multi-index, i.e., :math:`\mathbb{R}^n` is a tensor
    space, the above definition is applied to the flattened array.

    See Also
    --------
    L2NormSquared
    LpNorm
    """

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `TensorSpace`
            Domain of the functional.
        """
        super(L2Norm, self).__init__(space=space, exponent=2)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> space = odl.rn(2)
        >>> op = odl.solvers.L2Norm(space)
        >>> op
        L2Norm(rn(2))
        """
        posargs = [self.domain]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, [])
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


class L2NormSquared(Functional):

    r"""The squared 2-norm as a functional.

    This functional is defined as

    .. math::
        \| x \|_2^2 &= \sum_{i=1}^n |x_i|^2
        \quad (\mathbb{R^n-}\text{like space}) \\
        \| x \|_2^2 &= \int |x(t)|^2 \mathrm{d}t
        \quad (L^p-\text{like space}).

    If :math:`n` is a multi-index, i.e., :math:`\mathbb{R}^n` is a tensor
    space, the above definition is applied to the flattened array.

    In contrast to the non-squared 2-norm, this functional is differentiable
    and has a well-defined gradient everywhere.

    See Also
    --------
    L2Norm
    L1Norm
    """

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `TensorSpace`
            Domain of the functional.
        """
        super(L2NormSquared, self).__init__(
            space=space, linear=False, grad_lipschitz=2)

    # TODO(#440): use integration operator when available
    def _call(self, x):
        """Return ``self(x)``."""
        return x.inner(x)

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        return ScalingOperator(self.domain, 2.0)

    @property
    def proximal(self):
        r"""Return the `proximal factory` of the functional.

        The proximal of the squared L2 norm is the scaling

        .. math::
            \mathrm{prox}_{\sigma \|\cdot\|_2^2}(x) =
            \frac{x}{1 + 2 \sigma}.
        """
        def l2_squared_prox_factory(sigma):
            """Return the L2 squared proximal operator for ``sigma``."""
            return ScalingOperator(1 / (1 + 2 * sigma), self.domain)

        return l2_squared_prox_factory

    @property
    def convex_conj(self):
        r"""The convex conjugate functional of the squared L2 norm.

        Notes
        -----
        The conjugate functional of :math:`\| \cdot \|_2^2` is
        :math:`\frac{1}{4}\| \cdot \|_2^2`
        """
        return (1 / 4) * L2NormSquared(self.domain)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> space = odl.rn(2)
        >>> op = odl.solvers.L2NormSquared(space)
        >>> op
        L2NormSquared(rn(2))
        """
        posargs = [self.domain]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, [])
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


class ConstantFunctional(Functional):

    """Functional mapping all inputs to the same constant."""

    def __init__(self, space, constant):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            Domain of the functional.
        constant : element in ``domain.field``
            The constant value of the functional.
        """
        super(ConstantFunctional, self).__init__(
            space=space, linear=(constant == 0), grad_lipschitz=0)
        self.__constant = self.range.element(constant)

    @property
    def constant(self):
        """The constant value of the functional."""
        return self.__constant

    def _call(self, x):
        """Return ``self(x)``."""
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
        r"""Convex conjugate of the constant functional.

        The convex conjugate of a constant functional :math:`f(x) = c`
        is an indicator function of the singleton set :math:`\{0\}` that
        takes the value :math:`-c` instead of 0:

         .. math::
            f^*(x) =
            \begin{cases}
                -c     & \text{if } x = 0, \\
                \infty & \text{else}.
            \end{cases}
        """
        return IndicatorZero(self.domain, -self.constant)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> space = odl.rn(2)
        >>> op = odl.solvers.ConstantFunctional(space, 1.5)
        >>> op
        ConstantFunctional(rn(2), constant=1.5)
        """
        posargs = [self.domain]
        optargs = [('constant', self.constant, None)]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


class ZeroFunctional(ConstantFunctional):

    """Functional that maps everything to zero."""

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            Domain of the functional.
        """
        super(ZeroFunctional, self).__init__(space=space, constant=0)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> space = odl.rn(2)
        >>> op = odl.solvers.ZeroFunctional(space)
        >>> op
        ZeroFunctional(rn(2))
        """
        posargs = [self.domain]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, [])
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


class ScalingFunctional(Functional, ScalingOperator):

    """Functional that scales the input argument by a value.

    Since the range of a functional is always a field, the domain of this
    functional must also be a field, i.e. real or complex numbers.
    """

    def __init__(self, field, scalar):
        """Initialize a new instance.

        Parameters
        ----------
        field : `Field`
            Domain of the functional.
        scalar : element in ``domain``
            The constant value to scale by.

        Examples
        --------
        >>> field = odl.RealNumbers()
        >>> func = odl.solvers.ScalingFunctional(field, 3)
        >>> func(5)
        15.0
        """
        Functional.__init__(self, space=field, linear=True, grad_lipschitz=0)
        ScalingOperator.__init__(self, field, scalar)

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        return ConstantFunctional(self.domain, self.scalar)


class IdentityFunctional(ScalingFunctional):

    """Functional that maps a scalar to itself.

    See Also
    --------
    odl.operator.IdentityOperator
    """

    def __init__(self, field):
        """Initialize a new instance.

        Parameters
        ----------
        field : `Field`
            Domain of the functional.
        """
        super(IdentityFunctional, self).__init__(field, 1.0)


class IndicatorBox(Functional):

    r"""Indicator functional on some box shaped domain.

    The indicator with lower bound :math:`a` and upper bound :math:`b`
    (can be scalars, vectors or functions) is
    defined as

    .. math::
        \iota_{[a,b]}(x) =
        \begin{cases}
            0      & \text{if } a \leq x \leq b \text{ everywhere}, \\
            \infty & \text{else}.
        \end{cases}
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
        >>> func = odl.solvers.IndicatorBox(space, 0, 2)
        >>> func([0, 1, 2])  # all points inside
        0
        >>> func([0, 1, 3])  # one point outside
        inf
        """
        super(IndicatorBox, self).__init__(space, linear=False)
        self.lower = lower
        self.upper = upper

    def _call(self, x):
        """Return ``self(x)``."""
        # Since the proximal projects onto our feasible set we can simply
        # check if it changes anything
        proj = self.proximal(1)(x)
        return np.inf if x.dist(proj) > 0 else 0

    @property
    def proximal(self):
        """Return the `proximal factory` of the functional.

        The proximal operator of the box indicator functional is a projection
        onto that box, i.e., setting the input equal to the lower bound where
        it is smaller, and equal to the upper bound where it is larger.

        See Also
        --------
        proximal_indicator_box
        """
        return proximal_indicator_box(self.domain, self.lower, self.upper)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> func = odl.solvers.IndicatorBox(space, 0, 2)
        >>> func
        IndicatorBox(rn(3), lower=0, upper=2)
        """
        posargs = [self.domain]
        optargs = [('lower', self.lower, None),
                   ('upper', self.upper, None)]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


class IndicatorNonnegativity(IndicatorBox):

    r"""Indicator on the set of non-negative numbers.

    The nonnegativity indicator is defined as

    .. math::
        \iota_0(x) =
        \begin{cases}
            0      & \text{if } x \geq 0 \text{ everywhere}, \\
            \infty & \text{else}.
        \end{cases}
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
        >>> func = odl.solvers.IndicatorNonnegativity(space)
        >>> func([0, 1, 2])  # all points positive
        0
        >>> func([0, 1, -3])  # one point negative
        inf
        """
        super(IndicatorNonnegativity, self).__init__(
            space, lower=0, upper=None)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> func = odl.solvers.IndicatorNonnegativity(space)
        >>> func
        IndicatorNonnegativity(rn(3))
        """
        posargs = [self.domain]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, [])
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


class IndicatorZero(Functional):

    r"""Indicator functional of the singleton set {0}.

    This functional is defined as

    .. math::
        \iota_{\{0\}, c}(x) =
        \begin{cases}
            c      & \text{if } x = 0 \text{ everywhere}, \\
            \infty & \text{else},
        \end{cases}

    where :math:`c` is a user-chosen constant value.
    """

    def __init__(self, space, constant=0):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            Domain of the functional.
        constant : element in ``domain.field``, optional
            The constant value that the functional takes in zero.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> func = odl.solvers.IndicatorZero(space)
        >>> func([0, 0, 0])
        0
        >>> func([0, 0, 1])
        inf

        >>> func = odl.solvers.IndicatorZero(space, constant=2)
        >>> func([0, 0, 0])
        2
        """
        super(IndicatorZero, self).__init__(space, linear=False)
        self.__constant = constant

    @property
    def constant(self):
        """The constant value of the functional if ``x=0``."""
        return self.__constant

    def _call(self, x):
        """Return ``self(x)``."""
        if x.norm() == 0:
            return self.constant
        else:
            return np.inf

    @property
    def convex_conj(self):
        """Convex conjugate of this functional.

        The convex conjugate of the singleton set :math:`\{0\}` with
        constant :math:`c` is the functional mapping everything to the
        constant :math:`-c`.
        """
        return ConstantFunctional(self.domain, -self.constant)

    @property
    def proximal(self):
        """A proximal factory for this functional.

        It returns the zero operator.
        """
        def zero_proximal(sigma=1.0):
            """Proximal factory for zero operator.

            Parameters
            ----------
            sigma : positive float, optional
                Step size parameter (not used).
            """
            return ZeroOperator(self.domain)

        return zero_proximal

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> func = odl.solvers.IndicatorZero(space)
        >>> func
        IndicatorZero(rn(3))
        """
        posargs = [self.domain]
        optargs = [('constant', self.constant, 0)]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


class KullbackLeibler(Functional):

    r"""The Kullback-Leibler divergence functional.

    Notes
    -----
    The Kullback-Leibler divergence with prior :math:`g \geq 0` is defined as

    .. math::
        \text{KL}(x)
        &=
        \begin{cases}
            \sum_{i} \left( x_i - g_i + g_i \ln \left( \frac{g_i}{x_i}
            \right) \right) & \text{if } x_i > 0 \text{ for all } i,
            \\
            +\infty & \text{otherwise.}
        \end{cases}
        \quad (\mathbb{R}^n\text{-like space}) \\[2ex]
        \text{KL}(x)
        &=
        \begin{cases}
            \int \left(
                x(t) - g(t) + g(t) \ln\left(\frac{g(t)}{x(t)}\right)
            \right)\, \mathrm{d}t  & \text{if } x(t) > 0 \text{ for all } t,
            \\
            +\infty & \text{otherwise.}
        \end{cases}
        \quad (L^p-\text{like space})

    Note that we use the common convention :math:`0 \ln 0 := 0`.
    KL-based objectives are common in MLEM optimization problems and are often
    used as data-matching term when data noise governed by a multivariate
    Poisson probability distribution is significant.

    This functional is related to the `KullbackLeiblerCrossEntropy`
    described in `this Wikipedia article
    <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_,
    in that they have flipped roles of variable :math:`x` and prior :math:`g`.

    For a theoretical exposition, see `[Csiszar1991]
    <http://www.jstor.org/stable/2241918>`_.

    See Also
    --------
    KullbackLeiblerConvexConj : the convex conjugate functional
    KullbackLeiblerCrossEntropy : related functional

    References
    ----------
    [Csiszar1991] I. Csiszar.
    *Why Least Squares and Maximum Entropy? An Axiomatic Approach to
    Inference for Linear Inverse Problems.*
    The Annals of Statistics, 19/4 (1991), pp 2032–-2066.
    """

    def __init__(self, space, prior=None):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `TensorSpace`
            Domain of the functional.
        prior : ``space`` `element-like`, optional
            Depending on the context, the prior, target or data
            distribution, assumed to be nonnegative.
            The default ``None`` is equivalent to a prior of all ones.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> prior = 3 * space.one()
        >>> kl = odl.solvers.KullbackLeibler(space, prior=prior)
        """
        super(KullbackLeibler, self).__init__(
            space=space, linear=False, grad_lipschitz=np.nan)

        if prior is not None and prior not in self.domain:
            raise ValueError('`prior` not in `domain`'
                             ''.format(prior, self.domain))

        self.__prior = prior

    @property
    def prior(self):
        """The prior in the Kullback-Leibler functional."""
        return self.__prior

    # TODO(#440): use integration operator when available
    def _call(self, x):
        """Return ``self(x)``.

        If any component of ``x`` is non-positive, the value is positive
        infinity.
        """
        # Lazy import to improve `import odl` time
        import scipy.special

        if self.prior is None:
            integrand = x - 1 - np.log(x)
        else:
            integrand = (x - self.prior +
                         scipy.special.xlogy(self.prior, self.prior / x))

        integral = integrand.inner(self.domain.one())
        if np.isnan(integral):
            # In this case, some element was less than or equal to zero
            return np.inf
        else:
            return integral

    @property
    def gradient(self):
        r"""The gradient operator of the Kullback-Leibler divergence.

        For a prior :math:`g` is given by

        .. math::
            \nabla \text{KL}(x) = 1 - \frac{g}{x}.

        The gradient is not defined if any component of :math:`x` is
        non-positive.
        """
        functional = self

        class KLGradient(Operator):

            """The gradient operator of this functional."""

            def _call(self, x):
                """Return ``self(x)``."""
                if functional.prior is None:
                    return (-1.0) / x + 1
                else:
                    return (-functional.prior) / x + 1

            def __repr__(self):
                """Return ``repr(self)``.

                Examples
                --------
                >>> space = odl.rn(3)
                >>> kl = odl.solvers.KullbackLeibler(space)
                >>> kl.gradient
                KullbackLeibler(rn(3)).gradient
                """
                return attribute_repr_string(repr(functional), 'gradient')

        return KLGradient(self.domain, self.domain, linear=False)

    @property
    def proximal(self):
        """A `proximal factory` for this functional.

        See Also
        --------
        odl.solvers.nonsmooth.proximal_operators.proximal_convex_conj_kl :
            `proximal factory` for convex conjugate of KL.
        odl.solvers.nonsmooth.proximal_operators.proximal_convex_conj :
            Proximal of the convex conjugate of a functional.
        """
        return proximal_convex_conj(
            proximal_convex_conj_kl(space=self.domain, g=self.prior))

    @property
    def convex_conj(self):
        """The convex conjugate of the KL functional.

        See Also
        --------
        KullbackLeiblerConvexConj
        """
        return KullbackLeiblerConvexConj(self.domain, self.prior)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> prior = 3 * space.one()
        >>> odl.solvers.KullbackLeibler(space, prior=prior)
        KullbackLeibler(rn(3), prior=rn(3).element([ 3.,  3.,  3.]))
        """
        posargs = [self.domain]
        optargs = [('prior', self.prior, None)]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


class KullbackLeiblerConvexConj(Functional):

    r"""The convex conjugate of the Kullback-Leibler divergence functional.

    Notes
    -----
    The convex conjugate :math:`\text{KL}^*` of the KL divergence with
    prior :math:`g \geq 0` is given by

    .. math::
        \text{KL}^*(x)
        &=
        \begin{cases}
            \sum_{i} \left( -g_i \ln(1 - x_i) \right) & \text{if }
            x_i < 1 \text{ for all } i,
            \\
            +\infty & \text{otherwise.}
        \end{cases}
        \quad (\mathbb{R}^n\text{-like space}) \\[2ex]
        \text{KL}^*(x)
        &=
        \begin{cases}
            \int \big(-g(t)\ln\left(1 - x(t)\big)
            \right)\, \mathrm{d}t  & \text{if } x(t) < 1 \text{ for all } t,
            \\
            +\infty & \text{otherwise.}
        \end{cases}
        \quad (L^p-\text{like space})

    See Also
    --------
    KullbackLeibler : convex conjugate
    """

    def __init__(self, space, prior=None):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `TensorSpace`
            Domain of the functional.
        prior : ``space`` `element-like`, optional
            Depending on the context, the prior, target or data
            distribution, assumed to be nonnegative.
            The default ``None`` is equivalent to a prior of all ones.
        """
        super(KullbackLeiblerConvexConj, self).__init__(
            space=space, linear=False, grad_lipschitz=np.nan)

        if prior is not None and prior not in self.domain:
            raise ValueError('`prior` not in `domain`'
                             ''.format(prior, self.domain))

        self.__prior = prior

    @property
    def prior(self):
        """The prior in the convex conjugate of the KL functional."""
        return self.__prior

    # TODO(#440): use integration operator when available
    def _call(self, x):
        """Return ``self(x)``.

        If any component of ``x`` is larger than or equal to 1, the value is
        positive infinity.
        """
        # Lazy import to improve `import odl` time
        import scipy.special

        if self.prior is None:
            integral = (-1.0 * (np.log1p(-x))).inner(self.domain.one())
        else:
            integrand = -scipy.special.xlog1py(self.prior, -x)
            integral = integrand.inner(self.domain.one())
        if np.isnan(integral):
            # In this case, some element was larger than or equal to one
            return np.inf
        else:
            return integral

    @property
    def gradient(self):
        """Gradient operator of this functional.

        The gradient of the convex conjugate of the KL divergence is given
        by

        .. math::
            \nabla \text{KL}^*(x) = \frac{g}{1 - x}.

        The gradient is not defined in points where any component of :math:`x`
        is (larger than or) equal to one.
        """
        functional = self

        class KLCCGradient(Operator):

            """The gradient operator of this functional."""

            def _call(self, x):
                """Return ``self(x)``."""
                if functional.prior is None:
                    return 1.0 / (1 - x)
                else:
                    return functional.prior / (1 - x)

            def __repr__(self):
                """Return ``repr(self)``.

                Examples
                --------
                >>> space = odl.rn(3)
                >>> kl_cc = odl.solvers.KullbackLeibler(space).convex_conj
                >>> kl_cc.gradient
                KullbackLeiblerConvexConj(rn(3)).gradient
                """
                return attribute_repr_string(repr(functional), 'gradient')

        return KLCCGradient(self.domain, self.domain, linear=False)

    @property
    def proximal(self):
        """A `proximal factory` for this functional.

        See Also
        --------
        odl.solvers.nonsmooth.proximal_operators.proximal_convex_conj_kl :
            `proximal factory` for convex conjugate of KL.
        odl.solvers.nonsmooth.proximal_operators.proximal_convex_conj :
            Proximal of the convex conjugate of a functional.
        """
        return proximal_convex_conj_kl(space=self.domain, g=self.prior)

    @property
    def convex_conj(self):
        """The convex conjugate functional of the KL convex conjugate.

        This is the original KL divergence.
        """
        return KullbackLeibler(self.domain, self.prior)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.domain]
        optargs = [('prior', self.prior, None)]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


class KullbackLeiblerCrossEntropy(Functional):

    r"""The Kullback-Leibler Cross Entropy divergence functional.

    Notes
    -----
    The Kullback-Leibler cross entropy with prior :math:`g > 0` is
    defined as

    .. math::
        \widetilde{\text{KL}}(x)
        &=
        \begin{cases}
            \sum_{i} \left( g_i - x_i + x_i \ln \left( \frac{x_i}{g_i}
            \right) \right) & \text{if } x_i \geq 0 \text{ for all } i,
            \\
            +\infty & \text{otherwise.}
        \end{cases}
        \quad (\mathbb{R}^n\text{-like space}) \\[2ex]
        \widetilde{\text{KL}}(x)
        &=
        \begin{cases}
            \int \left(
                g(t) - x(t) + x(t) \ln\left(\frac{x(t)}{g(t)}\right)
            \right)\, \mathrm{d}t  & \text{if } x(t) \geq 0 \text{ for all } t,
            \\
            +\infty & \text{otherwise.}
        \end{cases}
        \quad (L^p-\text{like space})

    Note that we use the common convention :math:`0 \ln 0 := 0`.
    This variant of the `KullbackLeibler` functional that flips the roles of
    :math:`x` and :math:`g` is more often used in statistics (for the
    comparison of probablility distributions) than in inverse problems.

    For more details see `this Wikipedia article
    <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_,
    or `[Csiszar1991] <http://www.jstor.org/stable/2241918>`_. or `[SJ1980]
    <http://ieeexplore.ieee.org/document/1056144/?arnumber=1056144>`_
    for a theoretical explanation.

    See Also
    --------
    KullbackLeibler : related functional
    KullbackLeiblerCrossEntropyConvexConj : the convex conjugate

    References
    ----------
    [Csiszar1991] I. Csiszar.
    *Why Least Squares and Maximum Entropy? An Axiomatic Approach to
    Inference for Linear Inverse Problems.*
    The Annals of Statistics, 19/4 (1991), pp 2032–-2066.

    [SJ1980] Shore, J and Johnson, R.
    *Axiomatic derivation of the principle of maximum entropy and the
    principle of minimum cross-entropy.*
    IEEE Transactions on Information Theory, 26/1 (1980), pp 26--37.
    """

    def __init__(self, space, prior=None):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `TensorSpace`
            Domain of the functional.
        prior : ``space`` `element-like`, optional
            Depending on the context, the prior, target or data
            distribution. It is assumed to be nonnegative.
            Default: if None it is take as the one-element.
        """
        super(KullbackLeiblerCrossEntropy, self).__init__(
            space=space, linear=False, grad_lipschitz=np.nan)

        if prior is not None and prior not in self.domain:
            raise ValueError('`prior` not in `domain`'
                             ''.format(prior, self.domain))

        self.__prior = prior

    @property
    def prior(self):
        """The prior in the Kullback-Leibler functional."""
        return self.__prior

    # TODO(#440): use integration operator when available
    def _call(self, x):
        """Return ``self(x)``.

        If any component of ``x`` is non-positive, the value is positive
        infinity.
        """
        # Lazy import to improve `import odl` time
        import scipy.special

        if self.prior is None:
            integrand = 1 - x + scipy.special.xlogy(x, x)
        else:
            integrand = (self.prior - x +
                         scipy.special.xlogy(x, x / self.prior))

        integral = integrand.inner(self.domain.one())
        if np.isnan(integral):
            # In this case, some element was less than or equal to zero
            return np.inf
        else:
            return integral

    @property
    def gradient(self):
        r"""Gradient operator of this functional.

        The gradient of the KL cross entropy is given by

        .. math::
            \nabla \widetilde{\text{KL}}(x) = \ln\left(\frac{x}{g}\right).

        The gradient is not defined in points where one or more components
        of :math:`x` are less than or equal to 0.
        """
        functional = self

        class KLCrossEntropyGradient(Operator):

            """The gradient operator."""

            def _call(self, x):
                """Return ``self(x)``."""
                if functional.prior is None:
                    return np.log(x)
                else:
                    return np.log(x / functional.prior)

            def __repr__(self):
                """Return ``repr(self)``.

                Examples
                --------
                >>> space = odl.rn(3)
                >>> kl_xent = odl.solvers.KullbackLeiblerCrossEntropy(space)
                >>> kl_xent.gradient
                KullbackLeiblerCrossEntropy(rn(3)).gradient
                """
                return attribute_repr_string(repr(functional), 'gradient')

        return KLCrossEntropyGradient(self.domain, self.domain, linear=False)

    @property
    def proximal(self):
        """Return a `proximal factory` for this functional.

        See Also
        --------
        odl.solvers.nonsmooth.proximal_operators.\
proximal_convex_conj_kl_cross_entropy :
            `proximal factory` for convex conjugate of the KL cross entropy.
        odl.solvers.nonsmooth.proximal_operators.proximal_convex_conj :
            Proximal of the convex conjugate of a functional.
        """
        return proximal_convex_conj(proximal_convex_conj_kl_cross_entropy(
            space=self.domain, g=self.prior))

    @property
    def convex_conj(self):
        """The convex conjugate of the KL cross entropy.

        See Also
        --------
        KullbackLeiblerCrossEntropyConvexConj
        """
        return KullbackLeiblerCrossEntropyConvexConj(self.domain, self.prior)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.domain]
        optargs = [('prior', self.prior, None)]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


class KullbackLeiblerCrossEntropyConvexConj(Functional):

    r"""The convex conjugate of the Kullback-Leibler cross entropy.

    Notes
    -----
    The convex conjugate :math:`\widetilde{\text{KL}}^*` of the KL cross
    entropy with prior :math:`g \geq 0` is given by

    .. math::
        \widetilde{\text{KL}}^*(x)
        &= \sum_{i} g_i\, (\mathrm{e}^{x_i} - 1)
        \quad (\mathbb{R}^n\text{-like space}) \\[2ex]
        \widetilde{\text{KL}}^*(x)
        &= \int g(t))\, (\mathrm{e}^{x(t)} - 1)\, \mathrm{d}t
        \quad (L^p-\text{like space})

    See Also
    --------
    KullbackLeiblerCrossEntropy : convex conjugate
    """

    def __init__(self, space, prior=None):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `TensorSpace`
            Domain of the functional.
        prior : ``space`` `element-like`, optional
            Depending on the context, the prior, target or data
            distribution. It is assumed to be nonnegative.
            Default: if None it is take as the one-element.
        """
        super(KullbackLeiblerCrossEntropyConvexConj, self).__init__(
            space=space, linear=False, grad_lipschitz=np.nan)

        if prior is not None and prior not in self.domain:
            raise ValueError('`prior` not in `domain`'
                             ''.format(prior, self.domain))

        self.__prior = prior

    @property
    def prior(self):
        """The prior in convex conjugate Kullback-Leibler Cross Entorpy."""
        return self.__prior

    # TODO(#440): use integration operator when available
    def _call(self, x):
        """Return ``self(x)``."""
        if self.prior is None:
            return (np.exp(x) - 1).inner(self.domain.one())
        else:
            return (self.prior * (np.exp(x) - 1)).inner(self.domain.one())

    @property
    def gradient(self):
        """Gradient operator of this functional.

        The gradient of the convex conjugate of the KL cross entropy is

        .. math::
            \nabla \widetilde{\text{KL}}^*(x) = g\, \mathrm{e}^x,

        where multiplication and exponential are taken pointwise.
        """
        # Avoid circular import
        from odl.ufunc_ops import exp

        if self.prior is None:
            return exp(self.domain)
        else:
            return self.prior * exp(self.domain)

    @property
    def proximal(self):
        """Return the `proximal factory` of the functional.

        See Also
        --------
        odl.solvers.nonsmooth.proximal_operators.\
proximal_convex_conj_kl_cross_entropy :
            `proximal factory` for convex conjugate of the KL cross entropy.
        """
        return proximal_convex_conj_kl_cross_entropy(space=self.domain,
                                                     g=self.prior)

    @property
    def convex_conj(self):
        """Biconjugate of the KL cross entropy, the original functional."""
        return KullbackLeiblerCrossEntropy(self.domain, self.prior)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.domain]
        optargs = [('prior', self.prior, None)]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


class SeparableSum(Functional):

    r"""The functional corresponding to separable sum of functionals.

    The separable sum of functionals ``f_1, f_2, ..., f_n`` is given by ::

        h(x_1, x_2, ..., x_n) = sum_i^n f_i(x_i)

    The separable sum is thus defined for any collection of functionals with
    the same range.

    Notes
    -----
    The separable sum of functionals :math:`f_1, f_2, ..., f_n` is defined
    as

    .. math::
        h(x_1, x_2, ..., x_n) = \sum_{i=1}^n f_i(x_i)

    The separation carries over to important derived properties:

    - The gradient is a `DiagonalOperator`:

      .. math::
          [\nabla h](x_1, x_2, ..., x_n) =
          [\nabla f_1(x_i), \nabla f_2(x_i), ..., \nabla f_n(x_i)]

    - The convex conjugate is also a separable sum:

      .. math::
          [h^*](y_1, y_2, ..., y_n) = \sum_{i=1}^n f_i^*(y_i)

    - The proximal operator is separated as well:

      .. math::
          \mathrm{prox}_{\sigma h}(x_1, x_2, ..., x_n) =
          [\mathrm{prox}_{\sigma f_1}(x_1),
          \mathrm{prox}_{\sigma f_2}(x_2),
           \dots,
           \mathrm{prox}_{\sigma f_n}(x_n)].

      If :math:`\sigma = (\sigma_1, \sigma_2, \ldots, \sigma_n)` is a vector,
      the individual parameters are distributed, too:

      .. math::
          \mathrm{prox}_{\sigma h}(x_1, x_2, ..., x_n) =
          [\mathrm{prox}_{\sigma_1 f_1}(x_1),
           \mathrm{prox}_{\sigma_2 f_2}(x_2),
           ...,
           \mathrm{prox}_{\sigma_n f_n}(x_n)].
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

        The `proximal` factory allows using vector-valued stepsizes:

        >>> x = f_sum.domain.one()
        >>> f_sum.proximal([0.5, 2.0])(x)
        ProductSpace(rn(3), 2).element(
            [
                [ 0.5,  0.5,  0.5],
                [ 0.,  0.,  0.]
            ]
        )

        Create functional ``f([x1, ... ,xn]) = sum_i ||xi||_1``:

        >>> f_sum = odl.solvers.SeparableSum(l1, 5)
        """
        # Make a power space if the second argument is an integer
        if (len(functionals) == 2 and
                isinstance(functionals[1], Integral)):
            functionals = [functionals[0]] * functionals[1]

        domains = [func.domain for func in functionals]
        domain = ProductSpace(*domains)
        linear = all(func.is_linear for func in functionals)

        super(SeparableSum, self).__init__(space=domain, linear=linear)
        self.__functionals = tuple(functionals)

    def _call(self, x):
        """Return ``self(x)``."""
        return sum(fi(xi) for xi, fi in zip(x, self.functionals))

    @property
    def functionals(self):
        """The summands of the functional."""
        return self.__functionals

    def __getitem__(self, indices):
        """Return ``self[index]``.

        Parameters
        ----------
        indices : index expression
            Object determining which parts of the sum to extract.

        Returns
        -------
        subfunctional : `Functional` or `SeparableSum`
            Functional corresponding to the given indices.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> l1 = odl.solvers.L1Norm(space)
        >>> l2 = odl.solvers.L2Norm(space)
        >>> f_sum = odl.solvers.SeparableSum(l1, l2, 2 * l2)

        Extract a sub-functional via integer index:

        >>> f_sum[0]
        L1Norm(rn(3))

        Extract a subset of functionals:

        >>> f_sum[:2]
        SeparableSum(L1Norm(rn(3)), L2Norm(rn(3)))
        """
        result = self.functionals[indices]
        if isinstance(result, tuple):
            return SeparableSum(*result)
        else:
            return result

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        gradients = [func.gradient for func in self.functionals]
        return DiagonalOperator(*gradients)

    @property
    def proximal(self):
        """Return the `proximal factory` of the functional.

        The proximal operator separates over separable sums.

        See Also
        --------
        proximal_separable_sum
        """
        proximals = [func.proximal for func in self.functionals]
        return proximal_separable_sum(*proximals)

    @property
    def convex_conj(self):
        """The convex conjugate functional.

        Convex conjugate distributes over separable sums, so the result is
        simply the separable sum of the convex conjugates.
        """
        convex_conjs = [func.convex_conj for func in self.functionals]
        return SeparableSum(*convex_conjs)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> l1 = odl.solvers.L1Norm(space)
        >>> l2 = odl.solvers.L2Norm(space)
        >>> odl.solvers.SeparableSum(l1, 3)
        SeparableSum(L1Norm(rn(3)), 3)
        >>> odl.solvers.SeparableSum(l1, l2, 2 * l2)
        SeparableSum(
            L1Norm(rn(3)),
            L2Norm(rn(3)),
            FunctionalLeftScalarMult(L2Norm(rn(3)), 2)
        )
        """
        if all(func is self.functionals[0] for func in self.functionals[1:]):
            posargs = [self.functionals[0], len(self.functionals)]
        else:
            posargs = self.functionals

        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, [])
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


class QuadraticForm(Functional):

    r"""Functional for a general quadratic form.

    This functional represents the quadradic form :math:`Q: X \to \mathbb{R}`
    defined as

    .. math::
        Q(x) = \langle x, A(x)\rangle + \langle x, b\rangle + c,

    with an operator :math:`A: X \to X`, a vector :math:`b \in X` and a
    constant :math:`c \in \mathbb{R}`.
    """

    def __init__(self, operator=None, vector=None, constant=0):
        r"""Initialize a new instance.

        All parameters are optional, but at least one of ``op`` and ``vector``
        have to be provided so the space can be inferred.

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

        Examples
        --------
        >>> space = odl.rn(3)
        >>> op = odl.ScalingOperator(space, 2)
        >>> vec = space.one()
        >>> const = 2.0
        >>> quad_form = odl.solvers.QuadraticForm(op, vec, const)
        >>> quad_form(space.one())
        11.0
        """
        if operator is not None:
            domain = operator.domain
        elif vector is not None:
            domain = vector.space
        else:
            raise ValueError('need to provide at least one of `operator` and '
                             '`vector`')

        if (operator is not None and vector is not None and
                vector not in operator.domain):
            raise ValueError('domain of `operator` and space of `vector` need '
                             'to match')

        super(QuadraticForm, self).__init__(
            space=domain, linear=(operator is None and constant == 0))

        self.__operator = operator
        self.__vector = vector
        self.__constant = constant

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
            return x.inner(self.vector) + self.constant
        elif self.vector is None:
            return x.inner(self.operator(x)) + self.constant
        else:
            tmp = self.operator(x)
            tmp += self.vector
            return x.inner(tmp) + self.constant

    @property
    def gradient(self):
        """Gradient operator of the quadratic form.

        The gradient of a quadratic form

        .. math::
            Q(x) = \langle x, A(x)\rangle + \langle x, b\rangle + c

        is given by

        .. math::
            \nabla Q(x) = A'(x)^* x + A(x) + b,

        which simplifies to

        .. math::
            \nabla Q(x) = (A^* + A) x + b

        for linear operators.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> op = odl.ScalingOperator(space, 2)
        >>> vec = space.one()
        >>> const = 2.0
        >>> quad_form = odl.solvers.QuadraticForm(op, vec, const)
        >>> grad = quad_form.gradient
        >>> grad(-space.one())  # (2*I + 2*I)([-1, -1, -1]) + 1
        rn(3).element([-3., -3., -3.])
        >>> grad.derivative(space.one())
        OperatorSum(
            ScalingOperator(rn(3), scalar=2.0),
            ScalingOperator(rn(3), scalar=2.0)
        )
        """
        func = self

        class QuadraticFormGradient(Operator):

            """Gradient of a quadratic form."""

            def _call(self, x):
                """Return ``self(x)``."""
                if func.operator is None:
                    return func.vector

                tmp = func.operator(x)
                tmp += func.operator.derivative(x).adjoint(x)

                if func.vector is not None:
                    tmp += func.vector

                return tmp

            def derivative(self, x):
                """Second derivative of the quadratic form.

                Only defined if the operator is linear.
                """
                if func.operator.is_linear:
                    return func.operator + func.operator.adjoint
                else:
                    raise NotImplementedError(
                        'derivative of the quadratic form gradient only '
                        'implemented for linear operators')

            def __repr__(self):
                """Return ``repr(self)``.

                Examples
                --------
                >>> space = odl.rn(3)
                >>> op = odl.ScalingOperator(space, 2)
                >>> vec = space.one()
                >>> const = 2.0
                >>> quad_form = odl.solvers.QuadraticForm(op, vec, const)
                """
                return attribute_repr_string(repr(func), 'gradient')

        return QuadraticFormGradient(self.domain, self.domain,
                                     linear=(self.vector is None))

    @property
    def convex_conj(self):
        r"""The convex conjugate functional of a quadratic form.

        Notes
        -----
        The convex conjugate of the quadratic form
        :math:`Q(x) = \langle x, Ax \rangle + \langle x, b \rangle + c`
        with linar operator :math:`A` is given by

        .. math::
            Q^* (x) =
            \langle (x - b), A^-1 (x - b) \rangle - c =
            \langle x , A^-1 x \rangle - \langle x, A^-* b \rangle -
            \langle x, A^-1 b \rangle + \langle b, A^-1 b \rangle - c.

        If the quadratic part of the functional is zero it is instead given
        by a translated indicator function on zero, i.e., if

        .. math::
            Q(x) = \langle x, b \rangle + c,

        then

        .. math::
            Q^*(x^*) =
            \begin{cases}
                -c & \text{if } x^* = b \\
                \infty & \text{else.}
            \end{cases}

        See Also
        --------
        IndicatorZero
        """
        if self.operator is None:
            func = IndicatorZero(space=self.domain, constant=-self.constant)
            if self.vector is None:
                return func
            else:
                return func.translated(self.vector)

        if self.vector is None:
            # Handle trivial case separately
            return QuadraticForm(operator=self.operator.inverse,
                                 constant=-self.constant)
        else:
            opinv = self.operator.inverse
            vector = -opinv.adjoint(self.vector) - opinv(self.vector)
            constant = self.vector.inner(opinv(self.vector)) - self.constant

            return QuadraticForm(operator=opinv, vector=vector,
                                 constant=constant)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> op = odl.ScalingOperator(space, 2)
        >>> vec = space.one()
        >>> const = 2.0
        >>> odl.solvers.QuadraticForm(vector=vec)
        QuadraticForm(vector=rn(3).element([ 1.,  1.,  1.]))
        >>> odl.solvers.QuadraticForm(op, vec, const)
        QuadraticForm(
            operator=ScalingOperator(rn(3), scalar=2.0),
            vector=rn(3).element([ 1.,  1.,  1.]),
            constant=2.0
        )
        """
        optargs = [('operator', self.operator, None),
                   ('vector', self.vector, None),
                   ('constant', self.constant, 0)]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts([], optargs)
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


class NuclearNorm(Functional):

    r"""Nuclear norm for matrix-valued functions.

    For a matrix-valued function
    :math:`f : \Omega \rightarrow \mathbb{R}^{n \times m}`,
    the nuclear norm with parameters :math:`p` and :math:`q` is defined by

    .. math::
        \left( \int_\Omega \big\|\sigma(f(x))\big\|_p^q d x \right)^{1/q},

    where :math:`\sigma(f(x))` is the vector of singular values of the matrix
    :math:`f(x)` and :math:`\| \cdot \|_p` is the usual :math:`p`-norm on
    :math:`\mathbb{R}^{\min(n, m)}`.

    For a detailed description of its properties, e.g, its proximal, convex
    conjugate and more, see `[Du+2016] <https://arxiv.org/abs/1508.01308>`_.

    References
    ----------
    [Du+2016] J. Duran, M. Moeller, C. Sbert, and D. Cremers.
    *Collaborative Total Variation: A General Framework for Vectorial TV
    Models.* SIAM Journal of Imaging Sciences 9/1 (2016), pp 116--151.
    """

    def __init__(self, space, outer_exp=1, singular_vector_exp=2):
        """Initialize a new instance.

        Parameters
        ----------
        space : `ProductSpace` of `ProductSpace` of `TensorSpace`
            Domain of the functional.
        outer_exp : {1, 2, inf}, optional
            Exponent for the outer norm.
        singular_vector_exp : {1, 2, inf}, optional
            Exponent for the norm for the singular vectors.

        Examples
        --------
        Nuclear norm of a matrix-valued function with all ones in 3 points.
        The singular values are [2, 0], resulting in a 2-norm of 2.
        Since there are 3 points, the expected total value is 6.

        >>> r3 = odl.rn(3)
        >>> space = r3 ** (2, 2)
        >>> norm = odl.solvers.NuclearNorm(space)
        >>> norm(space.one())
        6.0
        """
        if (not isinstance(space, ProductSpace) or
                not isinstance(space[0], ProductSpace)):
            raise TypeError('`space` must be a `ProductSpace` of '
                            '`ProductSpace`s')
        if (not space.is_power_space or not space[0].is_power_space):
            raise TypeError('`space` must be of the form `TensorSpace^(nxm)`')

        super(NuclearNorm, self).__init__(
            space=space, linear=False, grad_lipschitz=np.nan)

        self.outernorm = LpNorm(self.domain[0, 0], exponent=outer_exp)
        self.pwisenorm = PointwiseNorm(self.domain[0],
                                       exponent=singular_vector_exp)
        self.pshape = (len(self.domain), len(self.domain[0]))

    def _call(self, x):
        """Return ``self(x)``."""
        # Convert to array with "outer" indices last
        arr = moveaxis(x.asarray(), [0, 1], [-2, -1])
        svd_diag = np.linalg.svd(arr, compute_uv=False)

        # Rotate the axes so the svd-direction is first
        s_reordered = moveaxis(svd_diag, -1, 0)

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
        # TODO(kohr-h): document the math of the proximal
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
                super(NuclearNormProximal, self).__init__(
                    func.domain, func.domain, linear=False)

            def _call(self, x):
                """Return ``self(x)``."""
                arr = moveaxis(x.asarray(), [0, 1], [-2, -1])

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
                    s_reordered = moveaxis(s, -1, 0)
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
                return moveaxis(result, [-2, -1], [0, 1])

        return NuclearNormProximal

    @property
    def convex_conj(self):
        """Convex conjugate of the nuclear norm.

        The convex conjugate is the indicator function on the unit ball of
        the dual norm where the dual norm is obtained by taking the conjugate
        exponent of both the outer and singular vector exponents.

        See Also
        --------
        IndicatorNuclearNormUnitBall
        """
        return IndicatorNuclearNormUnitBall(
            self.domain,
            conj_exponent(self.outernorm.exponent),
            conj_exponent(self.pwisenorm.exponent))

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> space = r3 ** (2, 2)
        >>> odl.solvers.NuclearNorm(space, singular_vector_exp=1.0)
        NuclearNorm(
            ProductSpace(ProductSpace(rn(3), 2), 2),
            singular_vector_exp=1.0
        )
        """
        posargs = [self.domain]
        optargs = [('outer_exp', self.outernorm.exponent, 1.0),
                   ('singular_vector_exp', self.pwisenorm.exponent, 2.0)]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


class IndicatorNuclearNormUnitBall(Functional):
    r"""Indicator on unit ball of nuclear norm for matrix valued functions.

    For a matrix-valued function
    :math:`f : \Omega \rightarrow \mathbb{R}^{n \times m}`,
    the nuclear norm with parameters :math:`p` and :math:`q` is defined by

    .. math::
        \left( \int_\Omega \|\sigma(f(x))\|_p^q d x \right)^{1/q},

    where :math:`\sigma(f(x))` is the vector of singular values of the matrix
    :math:`f(x)` and :math:`\| \cdot \|_p` is the usual :math:`p`-norm on
    :math:`\mathbb{R}^{\min(n, m)}`.

    This functional is defined as the indicator on the unit ball of the nuclear
    norm, that is, 0 if the nuclear norm is less than 1, and infinity else.

    For a detailed description of its properties, e.g, its proximal, convex
    conjugate and more, see `[Du+2016] <https://arxiv.org/abs/1508.01308>`_.

    References
    ----------
    [Du+2016] J. Duran, M. Moeller, C. Sbert, and D. Cremers.
    *Collaborative Total Variation: A General Framework for Vectorial TV
    Models.* SIAM Journal of Imaging Sciences 9/1 (2016), pp 116--151.
    """

    def __init__(self, space, outer_exp=1, singular_vector_exp=2):
        """Initialize a new instance.

        Parameters
        ----------
        space : `ProductSpace` of `ProductSpace` of `TensorSpace`
            Domain of the functional.
        outer_exp : {1, 2, inf}, optional
            Exponent for the outer norm.
        singular_vector_exp : {1, 2, inf}, optional
            Exponent for the norm for the singular vectors.

        Examples
        --------
        Indicator evaluated at a matrix-valued function with all ones
        in 3 points. The singular values are [2, 0], which result in a 2-norm
        of 2. Since there are 3 points, the expected total value is 6.
        Since the nuclear norm is larger than 1, the indicator is infinity.

        >>> r3 = odl.rn(3)
        >>> space = r3 ** (2, 2)
        >>> func = odl.solvers.IndicatorNuclearNormUnitBall(space)
        >>> func(space.one())
        inf
        """
        super(IndicatorNuclearNormUnitBall, self).__init__(
            space=space, linear=False, grad_lipschitz=np.nan)
        self.__norm = NuclearNorm(space, outer_exp, singular_vector_exp)

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
        # TODO(kohr-h): document math
        return proximal_convex_conj(self.convex_conj.proximal)

    @property
    def convex_conj(self):
        """Convex conjugate of the unit ball indicator of the nuclear norm.

        The convex conjugate is the dual nuclear norm where the dual norm is
        obtained by taking the conjugate exponent of both the outer and
        singular vector exponents.

        See Also
        --------
        NuclearNorm
        """
        return NuclearNorm(self.domain,
                           conj_exponent(self.__norm.outernorm.exponent),
                           conj_exponent(self.__norm.pwisenorm.exponent))

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.domain]
        optargs = [('outer_exp', self.outernorm.exponent, 1.0),
                   ('singular_vector_exp', self.pwisenorm.exponent, 2.0)]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


class MoreauEnvelope(Functional):

    r"""Moreau envelope of a convex functional.

    The Moreau envelope is a way to smooth an arbitrary convex functional
    such that its gradient can be computed given the proximal of the original
    functional.
    The new functional has the same critical points as the original.
    It is also called the Moreau-Yosida regularization.

    Note that the only computable property of the Moreau envelope is the
    gradient, the functional itself cannot be evaluated without solving
    a minimization problem.

    See `Proximal Algorithms`_ for more information.

    The Moreau envelope of a convex functional
    :math:`f : \mathcal{X} \rightarrow \mathbb{R}` with parameter
    :math:`\sigma > 0` is defined by

    .. math::
        \mathrm{env}_{\sigma, f}(x) =
        \inf_{y \in \mathcal{X}}
        \left\{ \frac{1}{2 \sigma} \| x - y \|_2^2 + f(y) \right\}

    The gradient of the envelope is given by

    .. math::
        [\nabla \mathrm{env}_{\sigma, f}](x) =
        \frac{1}{\sigma} (x - \mathrm{prox}_{\sigma f}(x))

    Example: If :math:`f = \| \cdot \|_1`, then

    .. math::
        [\mathrm{env}_{\sigma,  \| \cdot \|_1}(x)]_i =
        \begin{cases}
            \frac{1}{2 \sigma} x_i^2 & \text{if } |x_i| \leq \sigma \\
            |x_i| - \frac{\sigma}{2} & \text{if } |x_i| > \sigma,
        \end{cases}

    which is the `Huber` functional.

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
        sigma : positive float, optional
            The scalar ``sigma`` in the definition of the Moreau envelope.
            Larger values mean stronger smoothing.

        Examples
        --------
        Create smoothed l1 norm:

        >>> space = odl.rn(3)
        >>> l1_norm = odl.solvers.L1Norm(space)
        >>> smoothed_l1 = odl.solvers.MoreauEnvelope(l1_norm)
        """
        super(MoreauEnvelope, self).__init__(
            space=functional.domain, linear=False)
        self.__functional = functional
        self.__sigma = float(sigma)

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

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> space = odl.rn(3)
        >>> l1_norm = odl.solvers.L1Norm(space)
        >>> odl.solvers.MoreauEnvelope(l1_norm)
        MoreauEnvelope(L1Norm(rn(3)))
        """
        posargs = [self.functional]
        optargs = [('sigma', self.sigma, 1.0)]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


class Huber(Functional):
    r"""The Huber functional.

    The Huber norm is the integral over a smoothed pointwise norm,

    .. math::
        H_\gamma(x) = \int_\Omega h_{\gamma}(|x(t)|_2) \mathrm{d}t

    where :math:`|\cdot|_2` denotes the Euclidean norm for vector-valued
    functions, which reduces to the absolute value for scalar-valued
    functions. The function :math:`h_\gamma` with smoothing
    :math:`\gamma` is given by

    .. math::
        h_{\gamma}(s) =
        \begin{cases}
            \frac{1}{2 \gamma} s^2 & \text{if } |s| \leq \gamma \\
            |s| - \frac{\gamma}{2} & \text{else}
        \end{cases}.

    The Huber norm is also the Moreau envelope of the 1-norm with smoothing
    parameter :math:`\gamma`.

    If :math:`\gamma > 0`, the functional is non-smooth and corresponds to
    the usual L1 norm. For :math:`gamma > 0`, it has a
    :math:`1/\gamma`-Lipschitz gradient, so that its convex conjugate is
    :math:`\gamma`-strongly convex.
    """

    def __init__(self, space, gamma):
        """Initialize a new instance.

        Parameters
        ----------
        space : `TensorSpace`
            Domain of the functional.
        gamma : float
            Smoothing parameter of the Huber functional.

        Examples
        --------
        Example of initializing the Huber functional:

        >>> space = odl.uniform_discr(0, 1, 14)
        >>> gamma = 0.1
        >>> huber_norm = odl.solvers.Huber(space, gamma=0.1)

        Check that if all elements are > ``gamma`` we get the L1 norm up to a
        constant:

        >>> x = 2 * gamma * space.one()
        >>> tol = 1e-5
        >>> constant = gamma / 2 * space.one().inner(space.one())
        >>> f = odl.solvers.L1Norm(space) - constant
        >>> abs(huber_norm(x) - f(x)) < tol
        True

        Check that if all elements are < ``gamma`` we get the squared L2 norm
        times the weight ``1/(2*gamma)``:

        >>> x = gamma / 2 * space.one()
        >>> f = 1 / (2 * gamma) * odl.solvers.L2NormSquared(space)
        >>> abs(huber_norm(x) - f(x)) < tol
        True

        Compare Huber- and L1 norm for vanishing smoothing ``gamma=0``:

        >>> x = odl.phantom.white_noise(space)
        >>> huber_norm = odl.solvers.Huber(space, gamma=0)
        >>> l1_norm = odl.solvers.L1Norm(space)
        >>> abs(huber_norm(x) - l1_norm(x)) < tol
        True

        Redo previous example for a product space in two dimensions:

        >>> domain = odl.uniform_discr([0, 0], [1, 1], [5, 5])
        >>> space = odl.ProductSpace(domain, 2)
        >>> x = odl.phantom.white_noise(space)
        >>> huber_norm = odl.solvers.Huber(space, gamma=0)
        >>> l1_norm = odl.solvers.GroupL1Norm(space, 2)
        >>> abs(huber_norm(x) - l1_norm(x)) < tol
        True
        """
        self.__gamma = float(gamma)

        if self.gamma > 0:
            grad_lipschitz = 1 / self.gamma
        else:
            grad_lipschitz = np.inf

        super(Huber, self).__init__(
            space=space, linear=False, grad_lipschitz=grad_lipschitz)

    @property
    def gamma(self):
        """The smoothing parameter of the Huber norm functional."""
        return self.__gamma

    def _call(self, x):
        """Return ``self(x)``."""
        if isinstance(self.domain, ProductSpace):
            norm = PointwiseNorm(self.domain, exponent=2)(x)
        else:
            norm = x.ufuncs.absolute()

        if self.gamma > 0:
            tmp = norm.ufuncs.square()
            tmp *= 1 / (2 * self.gamma)

            index = norm.ufuncs.greater_equal(self.gamma)
            tmp[index] = norm[index] - self.gamma / 2
        else:
            tmp = norm

        return tmp.inner(tmp.space.one())

    @property
    def convex_conj(self):
        """The convex conjugate"""
        # TODO(kohr-h): document math
        if isinstance(self.domain, ProductSpace):
            norm = GroupL1Norm(self.domain, exponent=2)
        else:
            norm = L1Norm(self.domain)

        return FunctionalQuadraticPerturb(norm.convex_conj,
                                          quadratic_coeff=self.gamma / 2)

    @property
    def proximal(self):
        """Return the ``proximal factory`` of the functional.

        See Also
        --------
        odl.solvers.proximal_huber : `proximal factory` for the Huber
            norm.
        """
        return proximal_huber(space=self.domain, gamma=self.gamma)

    @property
    def gradient(self):
        r"""Gradient operator of the functional.

        The gradient of the Huber functional is given by

            .. math::
                \nabla f_{\gamma}(x) =
                \begin{cases}
                \frac{1}{\gamma} x & \text{if } \|x\|_2 \leq \gamma \\
                \frac{1}{\|x\|_2} x & \text{else}
                \end{cases}.

        Examples
        --------
        Check that the gradient norm is less than the norm of the one element:

        >>> space = odl.uniform_discr(0, 1, 14)
        >>> norm_one = space.one().norm()
        >>> x = odl.phantom.white_noise(space)
        >>> huber_norm = odl.solvers.Huber(space, gamma=0.1)
        >>> grad = huber_norm.gradient(x)
        >>> tol = 1e-5
        >>> grad.norm() <=  norm_one + tol
        True

        Redo previous example for a product space in two dimensions:

        >>> domain = odl.uniform_discr([0, 0], [1, 1], [5, 5])
        >>> space = odl.ProductSpace(domain, 2)
        >>> norm_one = space.one().norm()
        >>> x = odl.phantom.white_noise(space)
        >>> huber_norm = odl.solvers.Huber(space, gamma=0.2)
        >>> grad = huber_norm.gradient(x)
        >>> tol = 1e-5
        >>> grad.norm() <=  norm_one + tol
        True
        """

        functional = self

        class HuberGradient(Operator):

            """The gradient operator of this functional."""

            def _call(self, x):
                """Return ``self(x)``."""
                if isinstance(self.domain, ProductSpace):
                    norm = PointwiseNorm(self.domain, exponent=2)(x)
                else:
                    norm = x.ufuncs.absolute()

                grad = x / functional.gamma

                index = norm.ufuncs.greater_equal(functional.gamma)
                if isinstance(self.domain, ProductSpace):
                    for xi, gi in zip(x, grad):
                        gi[index] = xi[index] / norm[index]
                else:
                    grad[index] = x[index] / norm[index]

                return grad

            def __repr__(self):
                """Return ``repr(self)``.

                Examples
                --------
                >>> space = odl.uniform_discr(0, 1, 14)
                >>> huber = odl.solvers.Huber(space, gamma=0.1)
                >>> huber.gradient
                Huber(uniform_discr(0.0, 1.0, 14), gamma=0.1).gradient
                """
                return attribute_repr_string(repr(functional), 'gradient')

        return HuberGradient(self.domain, self.domain, linear=False)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> space = odl.uniform_discr(0, 1, 14)
        >>> odl.solvers.Huber(space, gamma=0.1)
        Huber(uniform_discr(0.0, 1.0, 14), gamma=0.1)
        """
        posargs = [self.domain]
        optargs = [('gamma', self.gamma, None)]
        with npy_printoptions(precision=REPR_PRECISION):
            inner_parts = signature_string_parts(posargs, optargs)
        return repr_string(self.__class__.__name__, inner_parts,
                           allow_mixed_seps=False)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
