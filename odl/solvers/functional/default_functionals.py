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
from odl.solvers.advanced.proximal_operators import (proximal_cconj,
                                                     proximal_l1,
                                                     proximal_cconj_l1,
                                                     proximal_l2,
                                                     proximal_cconj_l2,
                                                     proximal_l2_squared,
                                                     proximal_cconj_l2_squared)

from odl import (ZeroOperator, IdentityOperator)

# TODO: Import correct proximals from the already existing library

# TODO: Add weights to all existing functionals (e.g., a * ||x||_2). This is
# important, as it might change the proximal and the cc in nontrivial ways

__all__ = ('L1Norm', 'L2Norm', 'L2NormSquare', 'ZeroFunctional',
           'ConstantFunctional')


class L1Norm(Functional):
    """The functional corresponding to L1-norm."""

    def __init__(self, domain):
        """Initalizes an instance of the L1-norm functional.

        Parameters
        ----------
        domain : `LinearSpace`
            Set of elements on which the functional can be evaluat
        """
        super().__init__(domain=domain, linear=False, convex=True,
                         concave=False, smooth=False, grad_lipschitz=np.inf)

    def _call(self, x):
        """Applies the functional to the given point.

        Returns
        -------
        `self(x)` : `element` in the `field`of the ``domain``.
            Evaluation of the functional.
        """
        return np.abs(x).inner(self.domain.one())

    @property
    def gradient(x):
        """Gradient operator of the L1-functional."""
        raise NotImplementedError

    def proximal(self, sigma=1.0):
        """Return the proximal operator of the L1-functional.

        Parameters
        ----------
        sigma : positive float, optional
            Regularization parameter of the proximal operator

        Returns
        -------
        out : Operator
            Domain and range equal to domain of functional
        """
        functional = self

        class L1Proximal(Operator):
            """The proximal operator of the L1-functional"""
            def __init__(self):
                """Initialize a new instance of the L1Proximal."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)
                self.sigma = sigma

            # TODO: Check that this works for complex x
            def _call(self, x):
                """Applies the proximal operator to the given point.

                Parameters
                ----------
                x : `LinearSpaceVector`
                    Element in the domain of the functional to which the
                    proximal operator is applied.

                Returns
                -------
                `self(x)` : `LinearSpaceVector`
                    Evaluation of the proximal operator. An element in the
                    domain of the functional.
                """
                return np.maximum(np.abs(x) - sigma, 0) * np.sign(x)

        return L1Proximal()

    @property
    def conjugate_functional(self):
        """The convex conjugate functional of the L1-norm."""
        functional = self

        class L1ConjugateFunctional(Functional):
            """The convex conjugate functional of the L1-norm."""
            def __init__(self):
                """Initialize a new instance of the L1ConjugateFunctional."""
                super().__init__(functional.domain, linear=False)

            def _call(self, x):
                """Applies the convex conjugate functional of the L1-norm to
                the given point.

                Parameters
                ----------
                x : `LinearSpaceVector`
                    Element in the domain of the functional.

                Returns
                -------
                `self(x)` : `element` in the `field`of the ``domain``.
                    Evaluation of the functional.
                """
                if np.max(np.abs(x)) > 1:
                    return np.inf
                else:
                    return 0

        return L1ConjugateFunctional()


class L2Norm(Functional):
    """The functional corresponding to the L2-norm."""

    def __init__(self, domain):
        """Initalizes an instance of the L2-norm functional.

        Parameters
        ----------
        domain : `LinearSpace`
            Set of elements on which the functional can be evaluat
        """
        super().__init__(domain=domain, linear=False, convex=True,
                         concave=False, smooth=False, grad_lipschitz=np.inf)

    def _call(self, x):
        """Applies the functional to the given point.

        Returns
        -------
        `self(x)` : `element` in the `field`of the ``domain``.
            Evaluation of the functional.
        """
        return np.sqrt(np.abs(x).inner(np.abs(x)))

    @property
    def gradient(self):
        """Gradient operator of the L2-functional."""
        functional = self

        class L2Gradient(Operator):
            """The gradient operator for the L2-functional."""
            def __init__(self):
                """Initialize and instance of the gradient operator for the
                L2-functional.
                """
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
                `self(x)` : `LinearSpaceVector`
                    Evaluation of the gradient operator. An element in the
                    domain of the functional.
                """
                norm_of_x = x.norm()
                if norm_of_x == 0:
                    # The derivative is not defined for such a point.
                    raise ValueError('The gradient of the L2 functional is '
                                     'not defined for elements x {} with norm '
                                     '0.'.format(x))
                else:
                    return x / x.norm()

        return L2Gradient()

    def proximal(self, sigma=1.0):
        """Return the proximal operator of the L2-functional.

        Parameters
        ----------
        sigma : positive float, optional
            Regularization parameter of the proximal operator

        Returns
        -------
        out : Operator
            Domain and range equal to domain of functional
        """
        functional = self

        class L2Proximal(Operator):
            """The proximal operator of the L2-functional."""
            def __init__(self):
                """Initialize a new instance of the L2Proximal."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)
                self.sigma = sigma

            # TODO: Check that this works for complex x
            def _call(self, x):
                """Applies the proximal operator to the given point.

                Parameters
                ----------
                x : `LinearSpaceVector`
                    Element in the domain of the functional to which the
                    proximal operator is applied.

                Returns
                -------
                `self(x)` : `LinearSpaceVector`
                    Evaluation of the proximal operator. An element in the
                    domain of the functional.
                """
                return np.maximum(x.norm() - sigma, 0) * (x / x.norm())

        return L2Proximal()

    @property
    def conjugate_functional(self):
        """The convex conjugate functional of the L2-norm."""
        functional = self

        class L2ConjugateFunctional(Functional):
            """The convex conjugate functional of the L2-norm."""
            def __init__(self):
                """Initialize a new instance of the L2ConjugateFunctional."""
                super().__init__(functional.domain, linear=False)

            def _call(self, x):
                """Applies the convex conjugate functional of the L2-norm to
                the given point.

                Parameters
                ----------
                x : `LinearSpaceVector`
                    Element in the domain of the functional.

                Returns
                -------
                `self(x)` : `element` in the `field`of the ``domain``.
                    Evaluation of the functional.
                """
                if x.norm() > 1:
                    return np.inf
                else:
                    return 0

        return L2ConjugateFunctional()


class L2NormSquare(Functional):
    """The functional corresponding to the squared L2-norm."""

    def __init__(self, domain):
        """Initalizes an instance of the squared L2-norm functional.

        Parameters
        ----------
        domain : `LinearSpace`
            Set of elements on which the functional can be evaluat
        """
        super().__init__(domain=domain, linear=False, convex=True,
                         concave=False, smooth=True, grad_lipschitz=2)

    def _call(self, x):
        """Applies the functional to the given point.

        Returns
        -------
        `self(x)` : `element` in the `field`of the ``domain``.
            Evaluation of the functional.
        """
        return np.abs(x).inner(np.abs(x))

    @property
    def gradient(self):
        """Gradient operator of the squared L2-functional."""
        functional = self

        class L2SquareGradient(Operator):
            """Gradient operator of the squared L2-functional."""
            def __init__(self):
                """Initialize and instance of the gradient operator for the
                squared L2-functional.
                """
                super().__init__(functional.domain, functional.domain,
                                 linear=False)

            def _call(self, x):
                """Applies the gradient operator to the given point.

                Parameters
                ----------
                x : `LinearSpaceVector`
                    Element in the domain of the functional to which the
                    gradient operator is applied.

                Returns
                -------
                `self(x)` : `LinearSpaceVector`
                    Evaluation of the gradient operator. An element in the
                    domain of the functional.
                """
                return 2.0 * x

        return L2SquareGradient()

    def proximal(self, sigma=1.0):
        """Return the proximal operator of the squared L2-functional.

        Parameters
        ----------
        sigma : positive float, optional
            Regularization parameter of the proximal operator

        Returns
        -------
        out : Operator
            Domain and range equal to domain of functional
        """
        functional = self

        class L2SquareProximal(Operator):
            """The proximal operator of the squared L2-functional."""
            def __init__(self):
                """Initialize a new instance of the proximal operator for the
                squared L2-functional.
                """
                super().__init__(functional.domain, functional.domain,
                                 linear=False)
                self.sigma = sigma

            # TODO: Check that this works for complex x
            def _call(self, x):
                """Applies the proximal operator to the given point.

                Parameters
                ----------
                x : `LinearSpaceVector`
                    Element in the domain of the functional to which the
                    proximal operator is applied.

                Returns
                -------
                `self(x)` : `LinearSpaceVector`
                    Evaluation of the proximal operator. An element in the
                    domain of the functional.
                """
                return x * (1.0 / (2 * self.sigma + 1))

        return L2SquareProximal()

    @property
    def conjugate_functional(self):
        """The convex conjugate functional of the squared L2-norm."""
        functional = self

        class L2SquareConjugateFunctional(Functional):
            """The convex conjugate functional of the squared L2-norm."""
            def __init__(self):
                """Initialize a new instance of L2SquareConjugateFunctional."""
                super().__init__(functional.domain, linear=False)

            def _call(self, x):
                """Applies the convex conjugate functional of the squared
                L2-norm to the given point.

                Parameters
                ----------
                x : `LinearSpaceVector`
                    Element in the domain of the functional.

                Returns
                -------
                `self(x)` : `element` in the `field`of the ``domain``.
                    Evaluation of the functional.
                """
                return x.norm()**2 * (1.0 / 4.0)

            @property
            def gradient(self):
                """Gradient operator of the convex conjugate of the squared
                L2-norm.
                """
                functional = self

                class L2CCSquareGradient(Operator):
                    """Gradient operator of the convex conjugate of the convex
                    conjugate of the squared L2-norm.
                    """
                    def __init__(self):
                        """Initialize and instance of the gradient operator for
                        the convex conjugate of the squared L2-norm.
                        """
                        super().__init__(functional.domain, functional.domain,
                                         linear=False)

                    def _call(self, x):
                        """Applies the gradient operator to the given point.

                        Parameters
                        ----------
                        x : `LinearSpaceVector`
                            Element in the domain of the functional to which
                            the gradient operator is applied.

                        Returns
                        -------
                        `self(x)` : `LinearSpaceVector`
                            Evaluation of the gradient operator. An element in
                            the domain of the functional.
                        """
                        return x * (1.0 / 2.0)

                return L2CCSquareGradient()

            def proximal(self, sigma=1.0):
                """Return the proximal operator of the convex conjugate
                functional of the squared L2-functional.

                Parameters
                ----------
                sigma : positive float, optional
                    Regularization parameter of the proximal operator

                Returns
                -------
                out : Operator
                    Domain and range equal to domain of functional
                """

                func = self

                # Note: this implementation is take from proximal_operatros,
                # however soomewhat simplified to a less general case (to
                # start with).
                class ProximalCConjL2Squared(Operator):
                    """Proximal operator of the convex conj of the squared
                    l2-norm/dist.
                    """

                    def __init__(self, sigma):
                        """Initialize a new instance.

                        Parameters
                        ----------
                        sigma : positive `float`
                            Step size parameter
                        """
                        self.sigma = float(sigma)
                        super().__init__(domain=func.domain,
                                         range=func.domain)

                    def _call(self, x, out=None):
                        """Applies the proximal operator to the given point.

                        Parameters
                        ----------
                        x : `LinearSpaceVector`
                            Element in the domain of the functional to which
                            the proximal operator is applied.

                        Returns
                        -------
                        `self(x)` : `LinearSpaceVector`
                            Evaluation of the proximal operator. An element in
                            the domain of the functional.
                        """

                        sig = self.sigma
                        return x * 1.0 / (1 + 0.5 * sig)

                return ProximalCConjL2Squared(sigma)

        return L2SquareConjugateFunctional()


class ConstantFunctional(Functional):
    def __init__(self, domain, constant):
        """Initialize a ConstantFunctional instance.

        Parameters
        ----------
        domain : `LinearSpace`
            The space of elements which the functional is acting on.
        constant : element in `domain.field`
            The constant value of the functional
        """

        super().__init__(domain=domain, linear=True, convex=True,
                         concave=True, smooth=True, grad_lipschitz=0)

        if constant not in self.range:
            raise TypeError('constant {} not in the range {}.'
                            ''.format(constant, self.range))

        self._constant = constant

    def _call(self, x):
        """Applies the functional to the given point.

        Parameters
        ----------
        x : `LinearSpaceVector`
            Element in the domain of the functional.

        Returns
        -------
        `self(x)` : `float`
            Evaluation of the functional, which is a constant.
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

        where :math:`\\nabla f(x)` is the element used to evaluated
        derivatives in a direction :math:`d` by
        :math:`\\langle \\nabla f(x), d \\rangle`.
        """
        return ZeroOperator(self.domain)

    def proximal(self, sigma=1.0):
        """Return the proximal operator of the constant functional.

        The proximal operator for the constant functional is the identity
        operator.

        Parameters
        ----------
        sigma : positive float, optional
            Regularization parameter of the proximal operator

        Returns
        -------
        out : Operator
            Domain and range equal to domain of functional
        """

        return IdentityOperator(self.domain)

    @property
    def conjugate_functional(self):
        """Convex conjugate functional of the constant functional.

        This functional ``f^*``  is such that it maps the zero element to zero,
        the rest to infinity.
        """
        functional = self

        class ConstantFunctionalConjugateFunctional(Functional):
            """The convex conjugate functional to the constant functional."""

            def __init__(self):
                """Initialize a  ConstantFunctionalConjugateFunctional
                instance.
                """
                super().__init__(functional.domain, linear=False, convex=True)
                self.zero_element = self.domain.zero()

            def _call(self, x):
                """Applies the functional to the given point.

                Returns
                -------
                `self(x)` : `float`
                    Evaluation of the functional.
                """

                if x == self.zero_element:
                    return 0
                else:
                    return np.inf

        return ConstantFunctionalConjugateFunctional()


# TODO: Remove this one and simply make it a ConstantFunctional with constant
# 0. In doing so ZeroFunctional should inherite from ConstantFunctional, and
# init simply call super().__init__(domain, 0). However, what if 0 is not in
# domain.field?
class ZeroFunctional(Functional):
    def __init__(self, domain):
        """Initialize a ZeroFunctional instance.

        Parameters
        ----------
        domain : `LinearSpace`
            The space of elements which the functional is acting on.
        """
        super().__init__(domain=domain, linear=True, convex=True,
                         concave=True, smooth=True, grad_lipschitz=0)

    def _call(self, x):
        """Applies the functional to the given point.

        Returns
        -------
        `self(x)` : `float`
            Evaluation of the functional, which is always zero.
        """
        return 0

    @property
    def gradient(self):
        """Gradient operator of the functional.

        Notes
        -----
        The operator that corresponds to the mapping

        .. math::

            x \\to \\nabla f(x)

        where :math:`\\nabla f(x)` is the element used to evaluated
        derivatives in a direction :math:`d` by
        :math:`\\langle \\nabla f(x), d \\rangle`.
        """
        return ZeroOperator(self.domain)

    def proximal(self, sigma=1.0):
        """something...
        """
        # TODO: Update doc above

        return IdentityOperator(self.domain)

    @property
    def conjugate_functional(self):
        functional = self

        class ZeroFunctionalConjugateFunctional(Functional):
            """something...
            """
            # TODO: Update doc above

            def __init__(self):
                """Initialize a ZeroFunctionalConjugateFunctional instance.
                """
                super().__init__(functional.domain, linear=False, convex=True)
                self.zero_element = self.domain.zero()

            def _call(self, x):
                """Applies the functional to the given point.

                Returns
                -------
                `self(x)` : `float`
                    Evaluation of the functional.
                """

                if x == self.zero_element:
                    return 0
                else:
                    return np.inf

        return ZeroFunctionalConjugateFunctional()
