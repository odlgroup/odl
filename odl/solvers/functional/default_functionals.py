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

"""Default functionals defined on any (reasonable) space."""

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
    proximal_l2_squared)

from odl.operator.default_ops import (ZeroOperator, IdentityOperator)


__all__ = ('L1Norm', 'L2Norm', 'L2NormSquare', 'ZeroFunctional',
           'ConstantFunctional')


# TODO: Implement some of the missing gradients

class L1Norm(Functional):

    """The functional corresponding to L1-norm.

    The L1-norm, ``||x||_1``,  is defined as the integral/sum of ``|x|``.

    References
    ----------
    Wikipedia article on `Norms
        <https://en.wikipedia.org/wiki/Norm_(mathematics)>`_.
    """

    def __init__(self, domain):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace`
            Domain of the functional.
        """
        super().__init__(domain=domain, linear=False, convex=True,
                         concave=False, smooth=False, grad_lipschitz=np.inf)

    def _call(self, x):
        """Applies the functional to the given point.

        Parameters
        ----------
        x : `LinearSpaceVector`
            Element in the domain of the functional.

        Returns
        -------
        out : `field` element
            Result of the evaluation.
        """
        return np.abs(x).inner(self.domain.one())

    @property
    def gradient(x):
        """Gradient operator of the functional.

        Notes
        -----
        The operator that corresponds to the mapping

        .. math::

            x \\to \\nabla f(x)

        where :math:`\\nabla f(x)` is the element used to evaluate
        derivatives in a direction :math:`d` by
        :math:`\\langle \\nabla f(x), d \\rangle`.
        """
        # It does not exist on all of the space
        raise NotImplementedError('functional not differential.')

    def proximal(self, sigma=1.0):
        """Return the proximal operator of the functional.

        Parameters
        ----------
        sigma : positive float, optional
            Step size-like parameter

        Returns
        -------
        out : Operator
            Domain and range equal to domain of functional
        """

        return proximal_l1(space=self.domain)(sigma)

    @property
    def conjugate_functional(self):
        """The convex conjugate functional of the L1-norm."""
        return IndicatorLinftyUnitBall(self.domain)


class IndicatorLinftyUnitBall(Functional):

    """The indicator function on the unit ball, in the L-infinity norm.

    This functional is defined as

        ``f(x) = 0 if ||x||_infty <= 1, infty else,``

    where ``||x||_infty`` is the infinity norm

        ``||x||_infty = max(|x|)``.

    Notes
    -----
    This functional is the convex conjugate functional of the L1-norm.
    """

    def __init__(self, domain):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace`
            Domain of the functional.
        """
        super().__init__(domain=domain, linear=False)

    def _call(self, x):
        """Applies the functional the given point.

        Parameters
        ----------
        x : `LinearSpaceVector`
            Element in the domain of the functional.

        Returns
        -------
        out : `field` element
            Result of the evaluation.
        """
        if np.max(np.abs(x)) > 1:
            return np.inf
        else:
            return 0

    @property
    def gradient(x):
        """Gradient operator of the functional.

        Notes
        -----
        The operator that corresponds to the mapping

        .. math::

            x \\to \\nabla f(x)

        where :math:`\\nabla f(x)` is the element used to evaluate
        derivatives in a direction :math:`d` by
        :math:`\\langle \\nabla f(x), d \\rangle`.
        """
        # It does not exist on all of the space, only for ||x|| < 1
        # where it is the ZeroOperator
        raise NotImplementedError('functional not differential.')

    @property
    def conjugate_functional(self):
        """The conjugate functional of IndicatorLinftyUnitBall.

        Notes
        -----
        Since the functional is the conjugate functional of the L1-norm, and
        since the L1-norm is proper, convex and lower-semicontinuous,
        by the Fenchel-Moreau theorem the convex conjugate functional
        of the convex conjugate functional of IndicatorLinftyUnitBall, also
        known as the biconjugate of the L1-norm, is the L1-norm [BC2011]_.
        """
        return L1Norm(domain=self.domain)

    def proximal(self, sigma=1.0):
        """Return the proximal operator of the functional.

        Parameters
        ----------
        sigma : positive float, optional
            Step size-like parameter

        Returns
        -------
        out : Operator
            Domain and range equal to domain of functional
        """

        return proximal_cconj_l1(space=self.domain)(sigma)


class L2Norm(Functional):

    """The functional corresponding to the L2-norm.

    The L2-norm, ``||x||_2``,  is defined as the square-root out of the
    integral/sum of ``x^2``.

    References
    ----------
    Wikipedia article on `Norms
        <https://en.wikipedia.org/wiki/Norm_(mathematics)>`_.
        """

    def __init__(self, domain):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace`
            Domain of the functional.
        """
        super().__init__(domain=domain, linear=False, convex=True,
                         concave=False, smooth=False, grad_lipschitz=np.inf)

    def _call(self, x):
        """Applies the functional to the given point.

        Parameters
        ----------
        x : `LinearSpaceVector`
            Element in the domain of the functional.

        Returns
        -------
        out : `field` element
            Result of the evaluation.
        """
        return np.sqrt(x.inner(x))

    @property
    def gradient(self):
        """Gradient operator of the functional.

        Notes
        -----
        The operator that corresponds to the mapping

        .. math::

            x \\to \\nabla f(x)

        where :math:`\\nabla f(x)` is the element used to evaluate
        derivatives in a direction :math:`d` by
        :math:`\\langle \\nabla f(x), d \\rangle`.
        """
        functional = self

        class L2Gradient(Operator):

            """The gradient operator of this functional."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)

            def _call(self, x):
                """Applies the gradient operator to the given point.

                Parameters
                ----------
                x : `LinearSpaceVector`
                    Element in the domain of the functional to which the
                    gradient operator is applied. The element must have a
                    non-zero norm

                Returns
                -------
                out : `LinearSpaceVector`
                    Result of the evaluation of the gradient operator. An
                    element in the domain of the functional.
                """
                norm_of_x = x.norm()
                if norm_of_x == 0:
                    # The derivative is not defined for such a point.
                    raise ValueError('The gradient of the L2 functional is '
                                     'not defined for elements x {} with norm '
                                     '0.'.format(x))
                else:
                    return x / norm_of_x

        return L2Gradient()

    def proximal(self, sigma=1.0):
        """Return the proximal operator of the functional.

        Parameters
        ----------
        sigma : positive float, optional
            Step size-like parameter

        Returns
        -------
        out : Operator
            Domain and range equal to domain of functional
        """

        return proximal_l2(space=self.domain)(sigma)

    @property
    def conjugate_functional(self):
        """The convex conjugate functional of the L2-norm."""
        return IndicatorL2UnitBall(self.domain)


class IndicatorL2UnitBall(Functional):

    """The indicator function on the unit ball, in the L2-norm.

    This functional is defined as

        ``f(x) = 0 if ||x||_2 <= 1, infty else,``

    where ``||x||_2`` is the L2-norm.

    Notes
    -----
    This functional is the convex conjugate functional of the L2-norm.
    """

    def __init__(self, domain):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace`
            Domain of the functional.
        """
        super().__init__(domain=domain, linear=False)

    def _call(self, x):
        """Applies the functional to the given point.

        Parameters
        ----------
        x : `LinearSpaceVector`
            Element in the domain of the functional.

        Returns
        -------
        out : `field` element
            Result of the evaluation.
        """
        if x.norm() > 1:
            return np.inf
        else:
            return 0

    @property
    def conjugate_functional(self):
        """The conjugate functional of IndicatorL2UnitBall.

        Notes
        -----
        Since IndicatorL2UnitBall is the convex conjugate functional of the
        L2-norm, and since the L2-norm is proper, convex and
        lower-semicontinuous, by the Fenchel-Moreau theorem the convex
        conjugate functional of IndicatorL2UnitBall, also known as the
        biconjugate of the L2-norm, is the L2-norm [BC2011]_.
        """
        return L2Norm(domain=self.domain)

    @property
    def gradient(x):
        """Gradient operator of the functional.

        Notes
        -----
        The operator that corresponds to the mapping

        .. math::

            x \\to \\nabla f(x)

        where :math:`\\nabla f(x)` is the element used to evaluate
        derivatives in a direction :math:`d` by
        :math:`\\langle \\nabla f(x), d \\rangle`.
        """
        # It does not exist on all of the space, only for ||x|| < 1
        # where it is the ZeroOperator.
        raise NotImplementedError('functional not differential.')

    def proximal(self, sigma=1.0):
        """Return the proximal operator of the functional.

        Parameters
        ----------
        sigma : positive float, optional
            Step size-like parameter

        Returns
        -------
        out : Operator
            Domain and range equal to domain of functional
        """

        return proximal_cconj_l2(space=self.domain)(sigma)


class L2NormSquare(Functional):

    """The functional corresponding to the squared L2-norm.

    The squared L2-norm, ``||x||_2^2``,  is defined as the integral/sum of
    ``x^2``.

    References
    ----------
    Wikipedia article on `Norms
        <https://en.wikipedia.org/wiki/Norm_(mathematics)>`_."""

    def __init__(self, domain):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace`
            Domain of the functional.
        """
        super().__init__(domain=domain, linear=False, convex=True,
                         concave=False, smooth=True, grad_lipschitz=2)

    def _call(self, x):
        """Applies the functional to the given point.

        Parameters
        ----------
        x : `LinearSpaceVector`
            Element in the domain of the functional.

        Returns
        -------
        out : `field` element
            Result of the evaluation.
        """
        return x.inner(x)

    @property
    def gradient(self):
        """Gradient operator of the functional.

        Notes
        -----
        The operator that corresponds to the mapping

        .. math::

            x \\to \\nabla f(x)

        where :math:`\\nabla f(x)` is the element used to evaluate
        derivatives in a direction :math:`d` by
        :math:`\\langle \\nabla f(x), d \\rangle`.
        """
        functional = self

        class L2SquareGradient(Operator):

            """Gradient operator of this functional."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, functional.domain,
                                 linear=True)

            def _call(self, x):
                """Applies the gradient operator to the given point.

                Parameters
                ----------
                x : `LinearSpaceVector`
                    Element in the domain of the functional to which the
                    gradient operator is applied.

                Returns
                -------
                out : `LinearSpaceVector`
                    Result of the evaluation of the gradient operator. An
                    element in the domain of the functional.
                """
                return 2.0 * x

        return L2SquareGradient()

    def proximal(self, sigma=1.0):
        """Return the proximal operator of the functional.

        Parameters
        ----------
        sigma : positive float, optional
            Step size-like parameter

        Returns
        -------
        out : Operator
            Domain and range equal to domain of functional
        """
        return proximal_l2_squared(space=self.domain)(sigma)

    @property
    def conjugate_functional(self):
        """The convex conjugate functional of the squared L2-norm.

        Notes
        -----
        The conjugate functional of :math:`\|x\|_2^2` is
        :math:`\\frac{1}{4}\|x\|_2^2`
        """
        return 1.0 / 4.0 * L2NormSquare(domain=self.domain)


class ConstantFunctional(Functional):

    """The constant functional.

    This functional maps all elements in the domain to a given, constant value.
    """

    def __init__(self, domain, constant):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace`
            Domain of the functional.
        constant : element in `domain.field`
            The constant value of the functional
        """
        super().__init__(domain=domain, linear=True, convex=True,
                         concave=True, smooth=True, grad_lipschitz=0)

        if constant not in self.range:
            raise TypeError('constant {} not in the range {}.'
                            ''.format(constant, self.range))

        self._constant = constant

    @property
    def constant(self):
        return self._constant

    def _call(self, x):
        """Applies the functional to the given point.

        Parameters
        ----------
        x : `LinearSpaceVector`
            Element in the domain of the functional.

        Returns
        -------
        out : `field` element
            Result of the evaluation.
        """
        return self._constant

    @property
    def gradient(self):
        """Gradient operator of the functional.

        Notes
        -----
        The operator that corresponds to the mapping

        .. math::

            x \\to \\nabla f(x)

        where :math:`\\nabla f(x)` is the element used to evaluate
        derivatives in a direction :math:`d` by
        :math:`\\langle \\nabla f(x), d \\rangle`.
        """
        return ZeroOperator(self.domain)

    def proximal(self, sigma=1.0):
        """Return the proximal operator of the functional.

        The proximal operator for the constant functional is the identity
        operator.

        Parameters
        ----------
        sigma : positive float, optional
            Step size-like parameter

        Returns
        -------
        out : Operator
            Domain and range equal to domain of functional
        """

        return IdentityOperator(self.domain)

    @property
    def conjugate_functional(self):
        """Convex conjugate functional of the constant functional.

        This functional is defined as

            ``f^*(x) = -constant if x=0, infty else``.
        """
        functional = self

        class ConjugateToConstantFunctional(Functional):

            """The convex conjugate functional of this functional."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, linear=False, convex=True)
                self._zero_element = self.domain.zero()
                self._constant = functional._constant

            @property
            def constant(self):
                """Returns the constant that the constant functional maps
                to.
                """
                return self._constant

            def _call(self, x):
                """Applies the functional to the given point.

                Parameters
                ----------
                x : `LinearSpaceVector`
                    Element in the domain of the functional.

                Returns
                -------
                out : `field` element
                    Result of the evaluation.
                """

                if x == self._zero_element:
                    return -self._constant
                else:
                    return np.inf

            @property
            def gradient(x):
                """Gradient operator of the functional.

                Notes
                -----
                The operator that corresponds to the mapping

                .. math::

                    x \\to \\nabla f(x)

                where :math:`\\nabla f(x)` is the element used to evaluate
                derivatives in a direction :math:`d` by
                :math:`\\langle \\nabla f(x), d \\rangle`.
                """
                # It does not exist anywhere
                raise NotImplementedError('functional not differential.')

            @property
            def conjugate_functional(self):
                """The conjugate functional of the conjugate functional of the
                constant functional.

                Notes
                -----
                Since the constant functional is proper, convex and
                lower-semicontinuous, by the Fenchel-Moreau theorem the convex
                conjugate functional of the convex conjugate functional, also
                known as the biconjugate, is the functional itself [BC2011]_.
                """
                return ConstantFunctional(self.domain, self._constant)

            def proximal(self, sigma=1.0):
                """Return the proximal operator of the functional.

                Note that this is the zero-operator

                Parameters
                ----------
                sigma : positive float, optional
                    Step size-like parameter

                Returns
                -------
                out : Operator
                    Domain and range equal to domain of functional
                """

                return ZeroOperator(self.domain)

        return ConjugateToConstantFunctional()


class ZeroFunctional(ConstantFunctional):

    """The zero-functional.

    The zero-functional maps all elements in the domain to zero.
    """

    def __init__(self, domain):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace`
            Domain of the functional.
        """
        super().__init__(domain=domain, constant=0)
