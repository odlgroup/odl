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
from odl.solvers.advanced.proximal_operators import (
    proximal_l1, proximal_cconj_l1, proximal_l2, proximal_cconj_l2,
    proximal_l2_squared, proximal_const_func, proximal_box_constraint)

from odl.operator.default_ops import (ZeroOperator, ScalingOperator)


__all__ = ('L1Norm', 'L2Norm', 'L2NormSquared', 'ZeroFunctional',
           'ConstantFunctional', 'IndicatorLpUnitBall', 'IndicatorBox')


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


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
