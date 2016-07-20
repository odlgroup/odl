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
from odl.solvers.advanced.proximal_operators import (proximal_l1,
                                                     proximal_cconj_l1,
                                                     proximal_l2,
                                                     proximal_cconj_l2,
                                                     proximal_l2_squared,
                                                     proximal_cconj_l2_squared)

from odl import (ZeroOperator, IdentityOperator)


__all__ = ('L1Norm', 'L2Norm', 'L2NormSquare', 'ZeroFunctional',
           'ConstantFunctional')


# TODO: Implement some of the missing gradients

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

        Parameters
        ----------
        x : `LinearSpaceVector`
            Element in the domain of the functional.

        Returns
        -------
        `self(x)` : `element` in the `field` of the ``domain``.
            Evaluation of the functional.
        """
        return np.abs(x).inner(self.domain.one())

    @property
    def gradient(x):
        """Gradient operator of the L1-functional."""
        # It does not exist on all of the space
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

        return proximal_l1(space=self.domain)(sigma)

    @property
    def conjugate_functional(self):
        """The convex conjugate functional of the L1-norm."""
        functional = self

        class L1ConjugateFunctional(Functional):
            """The convex conjugate functional of the L1-norm."""
            def __init__(self):
                """Initialize a new instance of the L1ConjugateFunctional."""
                super().__init__(functional.domain, linear=False)
                self._orig_func = functional

            def _call(self, x):
                """Applies the convex conjugate functional of the L1-norm to
                the given point.

                Parameters
                ----------
                x : `LinearSpaceVector`
                    Element in the domain of the functional.

                Returns
                -------
                `self(x)` : `element` in the `field` of the ``domain``.
                    Evaluation of the functional.
                """
                if np.max(np.abs(x)) > 1:
                    return np.inf
                else:
                    return 0

            @property
            def gradient(x):
                """Gradient operator of the conjugate functional of the
                L1-norm.
                """
                # It does not exist on all of the space, only for ||x|| < 1
                # where it is the ZeroOperator
                raise NotImplementedError

            @property
            def conjugate_functional(self):
                """The conjugate functional of the conjugate functional of the
                L1-norm.

                Notes
                -----
                Since the L1-norm is proper, convex and lower-semicontinuous,
                by the Fenchel-Moreau theorem the convex conjugate functional
                of the convex conjugate functional, also known as the
                biconjugate, is the functional itself [BC2011]_.
                """
                return self._orig_func

            def proximal(self, sigma=1.0):
                """Return the proximal operator of the conjugate functional of
                the L1-norm.

                Parameters
                ----------
                sigma : positive float, optional
                    Regularization parameter of the proximal operator

                Returns
                -------
                out : Operator
                    Domain and range equal to domain of functional
                """

                return proximal_cconj_l1(space=self.domain)(sigma)

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

        Parameters
        ----------
        x : `LinearSpaceVector`
            Element in the domain of the functional.

        Returns
        -------
        `self(x)` : `element` in the `field` of the ``domain``.
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

        return proximal_l2(space=self.domain)(sigma)

    @property
    def conjugate_functional(self):
        """The convex conjugate functional of the L2-norm."""
        functional = self

        class L2ConjugateFunctional(Functional):
            """The convex conjugate functional of the L2-norm."""
            def __init__(self):
                """Initialize a new instance of the L2ConjugateFunctional."""
                super().__init__(functional.domain, linear=False)
                self._orig_func = functional

            def _call(self, x):
                """Applies the convex conjugate functional of the L2-norm to
                the given point.

                Parameters
                ----------
                x : `LinearSpaceVector`
                    Element in the domain of the functional.

                Returns
                -------
                `self(x)` : `element` in the `field` of the ``domain``.
                    Evaluation of the functional.
                """
                if x.norm() > 1:
                    return np.inf
                else:
                    return 0

            @property
            def conjugate_functional(self):
                """The conjugate functional of the conjugate functional of the
                L2-norm.

                Notes
                -----
                Since the L2-norm is proper, convex and lower-semicontinuous,
                by the Fenchel-Moreau theorem the convex conjugate functional
                of the convex conjugate functional, also known as the
                biconjugate, is the functional itself [BC2011]_.
                """
                return self._orig_func

            @property
            def gradient(x):
                """Gradient operator of the conjugate functional of the
                constant functional.
                """
                # It does not exist on all of the space, only for ||x|| < 1
                # where it is the ZeroOperator.
                raise NotImplementedError

            def proximal(self, sigma=1.0):
                """Return the proximal operator of the conjugate functional of
                the L2-norm.

                Parameters
                ----------
                sigma : positive float, optional
                    Regularization parameter of the proximal operator

                Returns
                -------
                out : Operator
                    Domain and range equal to domain of functional
                """

                return proximal_cconj_l2(space=self.domain)(sigma)

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

        Parameters
        ----------
        x : `LinearSpaceVector`
            Element in the domain of the functional.

        Returns
        -------
        `self(x)` : `element` in the `field` of the ``domain``.
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
        return proximal_l2_squared(space=self.domain)(sigma)

    @property
    def conjugate_functional(self):
        """The convex conjugate functional of the squared L2-norm."""
        functional = self

        class L2SquareConjugateFunctional(Functional):
            """The convex conjugate functional of the squared L2-norm."""
            def __init__(self):
                """Initialize a new instance of L2SquareConjugateFunctional."""
                super().__init__(functional.domain, linear=False)
                self._orig_func = functional

            def _call(self, x):
                """Applies the convex conjugate functional of the squared
                L2-norm to the given point.

                Parameters
                ----------
                x : `LinearSpaceVector`
                    Element in the domain of the functional.

                Returns
                -------
                `self(x)` : `element` in the `field` of the ``domain``.
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

            @property
            def conjugate_functional(self):
                """The convex conjugate functional of the conjugate functional
                of squared L2-norm.

                Notes
                -----
                Since the squared L2-norm is proper, convex and
                lower-semicontinuous, by the Fenchel-Moreau theorem the convex
                conjugate functional of the convex conjugate functional, also
                known as the biconjugate, is the functional itself [BC2011]_.
                """
                return self._orig_func

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
                return proximal_cconj_l2_squared(space=self.domain)(sigma)

        return L2SquareConjugateFunctional()


class ConstantFunctional(Functional):
    """The constant functional.

    This functional maps all elements in the domain to a given, constant value.
    """
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
                """Initialize a ConstantFunctionalConjugateFunctional
                instance.
                """
                super().__init__(functional.domain, linear=False, convex=True)
                self.zero_element = self.domain.zero()
                self._constant = functional._constant

            def _call(self, x):
                """Applies the functional to the given point.

                Parameters
                ----------
                x : `LinearSpaceVector`
                    Element in the domain of the functional.

                Returns
                -------
                `self(x)` : `float`
                    Evaluation of the functional.
                """

                if x == self.zero_element:
                    return -self._constant
                else:
                    return np.inf

            @property
            def gradient(x):
                """Gradient operator of the conjugate functional of the
                constant functional.
                """
                # It does not exist anywhere
                raise NotImplementedError

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
                """Return the proximal operator of the conjugate functional of
                the constant functional.

                Note that this is the zero-operator

                Parameters
                ----------
                sigma : positive float, optional
                    Regularization parameter of the proximal operator

                Returns
                -------
                out : Operator
                    Domain and range equal to domain of functional
                """

                return ZeroOperator(self.domain)

        return ConstantFunctionalConjugateFunctional()


class ZeroFunctional(ConstantFunctional):
    """The zero-functional.

    The zero-functional maps all elements in the domain to zero
    """
    def __init__(self, domain):
        """Initialize a ZeroFunctional instance.

        Parameters
        ----------
        domain : `LinearSpace`
            The space of elements which the functional is acting on.
        """
        super().__init__(domain=domain, constant=0)
