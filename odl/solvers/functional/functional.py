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

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

from odl.operator.operator import (
    Operator, OperatorComp, OperatorLeftScalarMult, OperatorRightScalarMult,
    OperatorRightVectorMult, OperatorSum, OperatorPointwiseProduct)
from odl.operator.default_ops import (IdentityOperator, ConstantOperator)
from odl.solvers.nonsmooth import (proximal_arg_scaling, proximal_translation,
                                   proximal_quadratic_perturbation,
                                   proximal_const_func, proximal_cconj)


__all__ = ('Functional', 'FunctionalLeftScalarMult',
           'FunctionalRightScalarMult', 'FunctionalComp',
           'FunctionalRightVectorMult', 'FunctionalSum', 'FunctionalScalarSum',
           'FunctionalTranslation', 'FunctionalLinearPerturb',
           'FunctionalProduct', 'FunctionalQuotient', 'simple_functional')


class Functional(Operator):

    """Implementation of a functional class.

    A functional is an operator ``f`` that maps from some domain ``X`` to the
    field of scalars ``F`` associated with the domain:

        ``f : X -> F``.

    Notes
    -----
    The implementation of the functional class assumes that the domain
    :math:`X` is a Hilbert space and that the field of scalars :math:`F` is a
    is the real numbers. It is possible to create functions that do not fulfil
    these assumptions, however some mathematical results might not be valide in
    this case. For more information, see `the ODL functional guide
    <http://odlgroup.github.io/odl/guide/in_depth/functional_guide.html>`_.
    """

    def __init__(self, space, linear=False, grad_lipschitz=np.nan):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            The domain of this functional, i.e., the set of elements to
            which this functional can be applied.
        linear : bool, optional
            If `True`, the functional is considered as linear.
        grad_lipschitz : float, optional
            The Lipschitz constant of the gradient. Default: ``nan``
        """
        Operator.__init__(self, domain=space,
                          range=space.field, linear=linear)
        self.__grad_lipschitz = float(grad_lipschitz)

    @property
    def grad_lipschitz(self):
        """Lipschitz constant for the gradient of the functional"""
        return self.__grad_lipschitz

    @grad_lipschitz.setter
    def grad_lipschitz(self, value):
        "Setter for the Lipschitz constant for the gradient."""
        self.__grad_lipschitz = float(value)

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
        raise NotImplementedError

    @property
    def proximal(self):
        """Proximal factory of the functional.

        Notes
        -----
        The proximal operator of a function :math:`f` is an operator defined as

            .. math::

                prox_{\\sigma f}(x) = \\sup_{y} \\left\{ f(y) -
                \\frac{1}{2\\sigma} \| y-x \|_2^2 \\right\}.

        Proximal operators are often used in different optimization algorithms,
        especially when designed to handle nonsmooth functionals.

        A `proximal factory` is a function that, when called with a step
        length :math:`\\sigma`, returns the corresponding proximal operator.

        The nonsmooth solvers that make use of proximal operators to solve a
        given optimization problem take a `proximal factory` as input,
        i.e., a function returning a proximal operator. See for example
        `forward_backward_pd`.
        """
        raise NotImplementedError

    @property
    def convex_conj(self):
        """Convex conjugate functional of the functional.

        Notes
        -----
        The convex conjugate functional of a convex functional :math:`f(x)`,
        defined on a Hilber space, is defined as the functional

        .. math::

            f^*(x^*) = \\sup_{x} \{ \\langle x^*,x \\rangle - f(x)  \}.

        The concept is also known as the Legendre transformation.

        References
        ----------
        Wikipedia article on `Convex conjugate
        <https://en.wikipedia.org/wiki/Convex_conjugate>`_.

        Wikipedia article on `Legendre transformation
        <https://en.wikipedia.org/wiki/Legendre_transformation>`_.

        For literature references see, e.g., [Lue1969]_, [Roc1970]_.
        """
        return FunctionalDefaultConvexConjugate(self)

    def derivative(self, point):
        """Return the derivative operator in the given point.

        This function returns the linear operator given by::

            self.derivative(point)(x) == self.gradient(point).inner(x)

        Parameters
        ----------
        point : `domain` element
            The point in which the gradient is evaluated.

        Returns
        -------
        derivative : `Operator`
        """
        return self.gradient(point).T

    def translated(self, shift):
        """Return a translation of the functional.

        For a given functional ``f`` and an element ``translation`` in the
        domain of ``f``, this operation creates the functional
        ``f(. - translation)``.

        Parameters
        ----------
        translation : `domain` element
            Element in the domain of the functional

        Returns
        -------
        out : `FunctionalTranslation`
            The functional ``f(. - translation)``
        """
        return FunctionalTranslation(self, shift)

    def __mul__(self, other):
        """Return ``self * other``.

        If ``other`` is an `Operator`, this corresponds to composition with the
        operator:

            ``(func * op)(x) == func(op(x))``

        If ``other`` is a scalar, this corresponds to right multiplication of
        scalars with functionals:

            ``(func * scalar)(x) == func(scalar * x)``

        If ``other`` is a vector, this corresponds to right multiplication of
        vectors with functionals:

            ``(func * vector) == func(vector * x)``

        Note that left and right multiplications are generally different.

        Parameters
        ----------
        other : `Operator`, `domain` element or scalar
            `Operator`:
            The `Operator.range` of ``other`` must match this functional's
            `domain`.

            `domain` element:
            ``other`` must be an element of this functionals's
            `Functional.domain`.

            scalar:
            The `domain` of this functional must be a
            `LinearSpace` and ``other`` must be an element of the `field`
            of this functional's `domain`. Note that this `field` is also this
            functional's `range`.

        Returns
        -------
        mul : `Functional`
            Multiplication result.

            If ``other`` is an `Operator`, ``mul`` is a
            `FunctionalComp`.

            If ``other`` is a scalar, ``mul`` is a
            `FunctionalRightScalarMult`.

            If ``other`` is a vector, ``mul`` is a
            `FunctionalRightVectorMult`.
        """
        if isinstance(other, Operator):
            return FunctionalComp(self, other)
        elif other in self.range:
            # Left multiplication is more efficient, so we can use this in the
            # case of linear functional.
            if other == 0:
                from odl.solvers.functional.default_functionals import (
                    ConstantFunctional)
                return ConstantFunctional(self.domain,
                                          self(self.domain.zero()))
            elif self.is_linear:
                return FunctionalLeftScalarMult(self, other)
            else:
                return FunctionalRightScalarMult(self, other)
        elif other in self.domain:
            return FunctionalRightVectorMult(self, other)
        else:
            return super().__mul__(other)

    def __rmul__(self, other):
        """Return ``other * self``.

        If ``other`` is an `Operator`, since a functional is also an operator
        this corresponds to operator composition:

            ``(op * func)(x) == op(func(x))``

        If ``other`` is a scalar, this corresponds to left multiplication of
        scalars with functionals:

            ``(scalar * func)(x) == scalar * func(x)``

        If ``other`` is a vector,  since a functional is also an operator this
        corresponds to left multiplication of vectors with operators:

            ``(vector * func)(x) == vector * func(x)``

        Note that left and right multiplications are generally different.

        Parameters
        ----------
        other : `Operator`, `domain` element or scalar
            `Operator`:
            The `Operator.domain` of ``other`` must match this functional's
            `Functional.range`.

            `LinearSpaceElement`:
            ``other`` must be an element of this functionals's
            `Functional.range`.

            scalar:
            The `Operator.domain` of this operator must be a
            `LinearSpace` and ``other`` must be an
            element of the ``field`` of this operator's
            `Operator.domain`.

        Returns
        -------
        rmul : `Functional` or `Operator`
            Multiplication result.

            If ``other`` is an `Operator`, ``rmul`` is an `OperatorComp`.

            If ``other`` is a scalar, ``rmul`` is a
            `FunctionalLeftScalarMult`.

            If ``other`` is a vector, ``rmul`` is a
            `OperatorLeftVectorMult`.
        """
        if other in self.range:
            if other == 0:
                from odl.solvers.functional.default_functionals import (
                    ZeroFunctional)
                return ZeroFunctional(self.domain)
            else:
                return FunctionalLeftScalarMult(self, other)
        else:
            return super().__rmul__(other)

    def __add__(self, other):
        """Return ``self + other``.

        If ``other`` is a `Functional`, this corresponds to

            ``(func1 + func2)(x) == func1(x) + func2(x)``

        If ``other`` is a scalar, this corresponds to adding a scalar to the
        value of the functional:

            ``(func + scalar)(x) == func(x) + scalar``

        Parameters
        ----------
        other : `Functional` or scalar
            `Functional`:
            The `Functional.domain` and `Functional.range` of ``other``
            must match this functional's  `Functional.domain` and
            `Functional.range`.

            scalar:
            The scalar needs to be in this functional's `Functional.range`.

        Returns
        -------
        sum : `Functional`
            Addition result.

            If ``other`` is in ``Functional.range``, ``sum`` is a
            `FunctionalScalarSum`.

            If ``other`` is a `Functional`, ``sum`` is a `FunctionalSum`.
        """
        if other in self.domain.field:
            return FunctionalScalarSum(self, other)
        elif isinstance(other, Functional):
            return FunctionalSum(self, other)
        else:
            return super().__add__(other)

    # Since addition is commutative, right and left addition is the same
    __radd__ = __add__

    def __sub__(self, other):
        """Return ``self - other``."""
        return self + (-1) * other


class FunctionalLeftScalarMult(Functional, OperatorLeftScalarMult):

    """Scalar multiplication of functional from the left.

    Given a functional ``f`` and a scalar ``scalar``, this represents the
    functional

        ``(scalar * f)(x) == scalar * f(x)``.

    `Functional.__rmul__` takes care of the case scalar = 0.
    """

    def __init__(self, func, scalar):
        """Initialize a new instance.

        Parameters
        ----------
        func : `Functional`
            Functional to be scaled.
        scalar : float, nonzero
            Number with which to scale the functional.
        """
        if not isinstance(func, Functional):
            raise TypeError('`func` {!r} is not a `Functional` instance'
                            ''.format(func))

        self.__scalar = func.range.element(scalar)

        Functional.__init__(self, space=func.domain, linear=func.is_linear,
                            grad_lipschitz=(
                                np.abs(scalar) * func.grad_lipschitz))

        OperatorLeftScalarMult.__init__(self, operator=func, scalar=scalar)

    @property
    def scalar(self):
        """The scalar."""
        return self.__scalar

    @property
    def functional(self):
        """The original functional."""
        return self.operator

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        return self.scalar * self.functional.gradient

    @property
    def convex_conj(self):
        """Convex conjugate functional of the scaled functional.

        `Functional.__rmul__` takes care of the case scalar = 0.
        """
        if self.scalar <= 0:
            raise ValueError('scaling with nonpositive values have no convex '
                             'conjugate. Current value: {}.'
                             ''.format(self.scalar))

        return self.scalar * self.functional.convex_conj * (1.0 / self.scalar)

    @property
    def proximal(self):
        """Proximal factory of the scaled functional.

        `Functional.__rmul__` takes care of the case scalar = 0

        See Also
        --------
        proximal_const_func
        """

        if self.scalar < 0:
            raise ValueError('proximal operator of functional scaled with a '
                             'negative value {} is not well-defined'
                             ''.format(self.scalar))

        elif self.scalar == 0:
            # Should not get here. `Functional.__rmul__` takes care of the case
            # scalar = 0
            return proximal_const_func(self.domain)

        else:
            def proximal_left_scalar_mult(sigma=1.0):
                """Proximal operator for left scalar multiplication.

                    Parameters
                    ----------
                    sigma : positive float
                        Step size parameter. Default: 1.0
                """
                return self.functional.proximal(sigma * self.scalar)

            return proximal_left_scalar_mult


class FunctionalRightScalarMult(Functional, OperatorRightScalarMult):

    """Scalar multiplication of the argument of functional.

    Given a functional ``f`` and a scalar ``scalar``, this represents the
    functional

        ``(f * scalar)(x) == f(scalar * x)``.

    `Functional.__mul__` takes care of the case scalar = 0.
    """

    def __init__(self, func, scalar):
        """Initialize a new instance.

        Parameters
        ----------
        func : `Functional`
            The functional which will have its argument scaled.
        scalar : float, nonzero
            The scaling parameter with which the argument is scaled.
        """

        if not isinstance(func, Functional):
            raise TypeError('`func` {!r} is not a `Functional` instance'
                            ''.format(func))

        scalar = func.domain.field.element(scalar)

        Functional.__init__(self, space=func.domain, linear=func.is_linear,
                            grad_lipschitz=(
                                np.abs(scalar) * func.grad_lipschitz))

        OperatorRightScalarMult.__init__(self, operator=func, scalar=scalar)

    @property
    def functional(self):
        """The original functional."""
        return self.operator

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        return self.scalar * self.functional.gradient * self.scalar

    @property
    def convex_conj(self):
        """Convex conjugate functional of functional with scaled argument.

        `Functional.__mul__` takes care of the case scalar = 0.
        """
        return self.functional.convex_conj * (1 / self.scalar)

    @property
    def proximal(self):
        """Proximal factory of the functional.

        See Also
        --------
        proximal_arg_scaling
        """
        return proximal_arg_scaling(self.functional.proximal, self.scalar)


class FunctionalComp(Functional, OperatorComp):

    """Composition of a functional with an operator.

    Given a functional ``func`` and an operator ``op``, such that the range of
    the operator is equal to the domain of the functional, this corresponds to
    the functional

        ``(func * op)(x) == func(op(x))``.
    """

    def __init__(self, func, op):
        """Initialize a new instance.

        Parameters
        ----------
        func : `Functional`
            The left ("outer") operator
        op : `Operator`
            The right ("inner") operator. Its range must coincide with the
            domain of ``func``.
        """
        if not isinstance(func, Functional):
            raise TypeError('`fun` {!r} is not a `Functional` instance'
                            ''.format(func))

        OperatorComp.__init__(self, left=func, right=op)

        Functional.__init__(self, space=op.domain,
                            linear=(func.is_linear and op.is_linear),
                            grad_lipschitz=np.nan)

    @property
    def gradient(self):
        """Gradient of the compositon according to the chain rule."""
        func = self.left
        op = self.right

        class FunctionalCompositionGradient(Operator):

            """Gradient of the compositon according to the chain rule."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(op.domain, op.domain, linear=False)

            def _call(self, x):
                """Apply the gradient operator to the given point."""
                return op.derivative(x).adjoint(func.gradient(op(x)))

        return FunctionalCompositionGradient()


class FunctionalRightVectorMult(Functional, OperatorRightVectorMult):

    """Expression type for the functional right vector multiplication.

    Given a functional ``func`` and a vector ``y`` in the domain of ``func``,
    this corresponds to the functional

        ``(func * y)(x) == func(y * x)``.
    """

    def __init__(self, func, vector):
        """Initialize a new instance.

        Parameters
        ----------
        func : `Functional`
            The domain of ``func`` must be a ``vector.space``.
        vector : `domain` element
            The vector to multiply by.
        """
        if not isinstance(func, Functional):
            raise TypeError('`fun` {!r} is not a `Functional` instance'
                            ''.format(func))

        OperatorRightVectorMult.__init__(self, operator=func, vector=vector)

        Functional.__init__(self, space=func.domain)

    @property
    def functional(self):
        return self.operator

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        return self.vector * self.operator.gradient * self.vector

    @property
    def convex_conj(self):
        """Convex conjugate functional of the functional.

        This is only defined for vectors with no zero-elements.
        """
        return self.functional.convex_conj * (1.0 / self.vector)


class FunctionalSum(Functional, OperatorSum):

    """Expression type for the sum of functionals.

    ``FunctionalSum(func1, func2) == (x --> func1(x) + func2(x))``.
    """

    def __init__(self, left, right):
        """Initialize a new instance.

        Parameters
        ----------
        left, right : `Functional`
            The summands of the functional sum. Their `Functional.domain`
            and `Functional.range` must coincide.
        """
        if not isinstance(left, Functional):
            raise TypeError('`left` {!r} is not a `Functional` instance'
                            ''.format(left))
        if not isinstance(right, Functional):
            raise TypeError('`right` {!r} is not a `Functional` instance'
                            ''.format(right))

        OperatorSum.__init__(self, left, right)

        Functional.__init__(self, space=left.domain,
                            linear=(left.is_linear and right.is_linear),
                            grad_lipschitz=(left.grad_lipschitz +
                                            right.grad_lipschitz))

    @property
    def gradient(self):
        """Gradient operator of functional sum."""
        return self.left.gradient + self.right.gradient


class FunctionalScalarSum(FunctionalSum):

    """Expression type for the sum of a functional and a scalar.

    ``FunctionalScalarSum(func, scalar) == (x --> func(x) + scalar)``
    """

    def __init__(self, func, scalar):
        """Initialize a new instance.

        Parameters
        ----------
        func : `Functional`
            Functional to which the scalar is added.
        scalar : `element` in the `field` of the ``domain``
            The scalar to be added to the functional. The `field` of the
            ``domain`` is the range of the functional.
        """
        from odl.solvers.functional.default_functionals import (
            ConstantFunctional)

        if not isinstance(func, Functional):
            raise TypeError('`fun` {!r} is not a `Functional` instance'
                            ''.format(func))
        if scalar not in func.range:
            raise TypeError('`scalar` {} is not in the range of '
                            '`func` {!r}'.format(scalar, func))

        FunctionalSum.__init__(self, left=func,
                               right=ConstantFunctional(space=func.domain,
                                                        constant=scalar))

    @property
    def scalar(self):
        """The scalar that is added to the functional"""
        return self.right.constant

    @property
    def proximal(self):
        """Proximal factory of the FunctionalScalarSum."""
        return self.left.proximal

    @property
    def convex_conj(self):
        """Convex conjugate functional of FunctionalScalarSum."""
        return self.left.convex_conj - self.scalar


class FunctionalTranslation(Functional):

    """Implementation of the translated functional.

    Given a functional ``f`` and an element ``translation`` in the domain of
    ``f``, this corresponds to the functional ``f(. - translation)``.
    """

    def __init__(self, func, translation):
        """Initialize a new instance.

        Given a functional ``f(.)`` and a vector ``translation`` in the domain
        of ``f``, this corresponds to the functional ``f(. - translation)``.

        Parameters
        ----------
        func : `Functional`
            Functional which is to be translated.
        translation : `domain` element
            The translation.
        """
        if not isinstance(func, Functional):
            raise TypeError('`func` {!r} not a `Functional` instance'
                            ''.format(func))

        translation = func.domain.element(translation)

        super().__init__(space=func.domain, linear=False,
                         grad_lipschitz=func.grad_lipschitz)

        # TODO: Add case if we have translation -> scaling -> translation?
        if isinstance(func, FunctionalTranslation):
            self.__functional = func.functional
            self.__translation = func.translation + translation

        else:
            self.__functional = func
            self.__translation = translation

    @property
    def functional(self):
        """The original functional that has been translated."""
        return self.__functional

    @property
    def translation(self):
        """The translation."""
        return self.__translation

    def _call(self, x):
        """Evaluate the functional in a point ``x``."""
        return self.functional(x - self.translation)

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        return (self.functional.gradient *
                (IdentityOperator(self.domain) - self.translation))

    @property
    def proximal(self):
        """Proximal factory of the translated functional.

        See Also
        --------
        odl.solvers.nonsmooth.proximal_operators.proximal_translation
        """
        return proximal_translation(self.functional.proximal,
                                    self.translation)

    @property
    def convex_conj(self):
        """Convex conjugate functional of the translated functional.

        Notes
        -----
        Given a functional :math:`f`, the convex conjugate of a translated
        version :math:`f(\cdot - y)` is given by a linear pertubation of the
        convex conjugate of :math:`f`:

        .. math::
            (f( . - y))^* (x) = f^*(x) + <y, x>.

        For reference on the identity used, see [KP2015]_.
        """
        return FunctionalLinearPerturb(
            self.functional.convex_conj,
            self.translation)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{!r}.translated({!r})'.format(self.functional,
                                              self.translation)

    def __str__(self):
        """Return ``str(self)``."""
        return '{}.translated({})'.format(self.functional,
                                          self.translation)


class FunctionalLinearPerturb(Functional):

    """The ``Functional`` representing ``f(.) + <linear_term, .>``."""

    def __init__(self, func, linear_term):
        """Initialize a new instance.

        Parameters
        ----------
        func : `Functional`
            Function corresponding to ``f``.
        translation : `domain` element
            Element in domain of ``func``, corresponding to the translation.
        """
        if not isinstance(func, Functional):
            raise TypeError('`func` {} is not a `Functional` instance'
                            ''.format(func))

        super().__init__(space=func.domain,
                         linear=func.is_linear)

        # Only compute the grad_lipschitz if it is not inf
        if (not func.grad_lipschitz == np.inf and
                not np.isnan(func.grad_lipschitz)):
            self.__grad_lipschitz = (func.grad_lipschitz +
                                     linear_term.norm())

        self.__functional = func
        self.__linear_term = func.domain.element(linear_term)

    @property
    def functional(self):
        """The original functional."""
        return self.__functional

    @property
    def linear_term(self):
        """The translation."""
        return self.__linear_term

    def _call(self, x):
        """Apply the functional to the given point."""
        return self.functional(x) + x.inner(self.linear_term)

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        return self.functional.gradient + ConstantOperator(self.linear_term)

    @property
    def proximal(self):
        """Proximal factory of the linearly perturbed functional.

        See Also
        --------
        odl.solvers.nonsmooth.proximal_operators.\
proximal_quadratic_perturbation
        """
        return proximal_quadratic_perturbation(
            self.functional.proximal, a=0, u=self.linear_term)

    @property
    def convex_conj(self):
        """Convex conjugate functional of the functional.

        Notes
        -----
        Given a functional :math:`f`, the convex conjugate of a linearly
        perturbed version :math:`f(x) + <y, x>` is given by a translation of
        the convex conjugate of :math:`f`:

        .. math::
            (f(x) + <y, x>)^* (x) = f^*(x - y).

        For reference on the identity used, see [KP2015]_.
        """
        return self.functional.convex_conj.translated(
            self.linear_term)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self.functional, self.linear_term)

    def __str__(self):
        """Return ``str(self)``."""
        return '{}({}, {})'.format(self.__class__.__name__,
                                   self.functional, self.linear_term)


class FunctionalProduct(Functional, OperatorPointwiseProduct):

    """Product ``p(x) = f(x) * g(x)`` of two functionals ``f`` and ``g``."""

    def __init__(self, left, right):
        """Initialize a new instance.

        Parameters
        ----------
        left, right : `Functional`
            Functionals in the product. Need to have matching domains.

        Examples
        --------
        Construct the functional || . ||_2^2 * 3

        >>> space = odl.rn(2)
        >>> func1 = odl.solvers.L2NormSquared(space)
        >>> func2 = odl.solvers.ConstantFunctional(space, 3)
        >>> prod = odl.solvers.FunctionalProduct(func1, func2)
        >>> prod([2, 3])  # expect (2**2 + 3**2) * 3 = 39
        39.0
        """
        if not isinstance(left, Functional):
            raise TypeError('`left` {} is not a `Functional` instance'
                            ''.format(left))
        if not isinstance(right, Functional):
            raise TypeError('`right` {} is not a `Functional` instance'
                            ''.format(right))

        OperatorPointwiseProduct.__init__(self, left, right)
        Functional.__init__(self, left.domain, linear=False,
                            grad_lipschitz=np.nan)

    @property
    def gradient(self):
        """Gradient operator of the functional.

        Notes
        -----
        The derivative is computed using Leibniz's rule:

        .. math::
            [\\nabla (f g)](p) = g(p) [\\nabla f](p) + f(p) [\\nabla g](p)
        """
        func = self

        class FunctionalProductGradient(Operator):

            """Functional representing the gradient of ``f(.) * g(.)``."""

            def _call(self, x):
                return (func.right(x) * func.left.gradient(x) +
                        func.left(x) * func.right.gradient(x))

        return FunctionalProductGradient(self.domain, self.domain,
                                         linear=False)


class FunctionalQuotient(Functional):

    """Quotient ``p(x) = f(x) / g(x)`` of two functionals ``f`` and ``g``."""

    def __init__(self, dividend, divisor):
        """Initialize a new instance.

        Parameters
        ----------
        dividend, divisor : `Functional`
            Functionals in the quotient. Need to have matching domains.

        Examples
        --------
        Construct the functional || . ||_2 / 5

        >>> space = odl.rn(2)
        >>> func1 = odl.solvers.L2Norm(space)
        >>> func2 = odl.solvers.ConstantFunctional(space, 5)
        >>> prod = odl.solvers.FunctionalQuotient(func1, func2)
        >>> prod([3, 4])  # expect sqrt(3**2 + 4**2) / 5 = 1
        1.0
        """
        if not isinstance(dividend, Functional):
            raise TypeError('`dividend` {} is not a `Functional` instance'
                            ''.format(dividend))
        if not isinstance(divisor, Functional):
            raise TypeError('`divisor` {} is not a `Functional` instance'
                            ''.format(divisor))

        if dividend.domain != divisor.domain:
            raise ValueError('domains of the operators do not match')

        self.__dividend = dividend
        self.__divisor = divisor

        Functional.__init__(self, dividend.domain, linear=False,
                            grad_lipschitz=np.nan)

    @property
    def dividend(self):
        """The dividend of the quotient."""
        return self.__dividend

    @property
    def divisor(self):
        """The divisor of the quotient."""
        return self.__divisor

    def _call(self, x):
        """Apply the functional to the given point."""
        return self.dividend(x) / self.divisor(x)

    @property
    def gradient(self):
        """Gradient operator of the functional.

        Notes
        -----
        The derivative is computed using the quotient rule:

        .. math::
            [\\nabla (f / g)](p) = (g(p) [\\nabla f](p) -
                                    f(p) [\\nabla g](p)) / g(p)^2
        """
        func = self

        class FunctionalQuotientGradient(Operator):

            """Functional representing the gradient of ``f(.) / g(.)``."""

            def _call(self, x):
                """Apply the functional to the given point."""
                dividendx = func.dividend(x)
                divisorx = func.divisor(x)
                return ((1 / divisorx) * func.dividend.gradient(x) +
                        (- dividendx / divisorx**2) * func.divisor.gradient(x))

        return FunctionalQuotientGradient(self.domain, self.domain,
                                          linear=False)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self.dividend, self.divisor)

    def __str__(self):
        """Return ``str(self)``."""
        return '{}({}, {})'.format(self.__class__.__name__,
                                   self.dividend, self.divisor)


class FunctionalDefaultConvexConjugate(Functional):

    """The `Functional` representing ``F^*``, the convex conjugate of ``F``.

    This class does not provide a way to evaluate the functional, it is rather
    intended to be used for its `proximal`.

    Notes
    -----
    The proximal is found by using the Moreau identity

    .. math::
        \\text{prox}_{\\sigma F^*}(y) = y -
        \\sigma \\text{prox}_{F / \\sigma}(y / \\sigma)

    which allows the proximal of the convex conjugate to be calculated without
    explicit knowledge about the convex conjugate itself.
    """

    def __init__(self, func):
        """Initialize a new instance.

        Parameters
        ----------
        func : `Functional`
            Functional corresponding to F.
        """
        if not isinstance(func, Functional):
            raise TypeError('`func` {} is not a `Functional` instance'
                            ''.format(func))

        super().__init__(space=func.domain,
                         linear=func.is_linear)

        self.__convex_conj = func

    @property
    def convex_conj(self):
        """The original functional."""
        return self.__convex_conj

    @property
    def proximal(self):
        """Proximal factory using the Moreu identity.

        Returns
        -------
        proximal : proximal_cconj
            Proximal computed using the Moreu identity
        """
        return proximal_cconj(self.convex_conj.proximal)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{!r}.convex_conj'.format(self.convex_conj)

    def __str__(self):
        """Return ``str(self)``."""
        return '{}.convex_conj'.format(self.convex_conj)


def simple_functional(space, fcall=None, grad=None, prox=None, grad_lip=np.nan,
                      cconj_fcall=None, cconj_grad=None, cconj_prox=None,
                      cconj_grad_lip=np.nan, linear=False):
    """Simplified interface to create a functional with specific properties.

    Users may specify as many properties as is needed by the application.

    Parameters
    ----------
    space : `LinearSpace`
        Space that the functional should act on.
    fcall : callable, optional
        Function to evaluate when calling the functional.
    grad : callable or `Operator`, optional
        Gradient operator of the functional.
    prox : `proximal factory`, optional
        Proximal factory for the functional.
    grad_lip : float, optional
        lipschitz constant of the functional.
    cconj_fcall : callable, optional
        Function to evaluate when calling the convex conjugate functional.
    cconj_grad : callable or `Operator`, optional
        Gradient operator of the convex conjugate functional
    cconj_prox : `proximal factory`, optional
        Proximal factory for the convex conjugate functional.
    cconj_grad_lip : float, optional
        lipschitz constant of the convex conjugate functional.
    linear : bool, optional
        True if the operator is linear.

    Examples
    --------
    Create squared sum functional on rn:

    >>> def f(x):
    ...     return sum(xi**2 for xi in x)
    >>> def dfdx(x):
    ...     return 2 * x
    >>> space = odl.rn(3)
    >>> func = simple_functional(space, f, grad=dfdx)
    >>> func.domain
    rn(3)
    >>> func.range
    RealNumbers()
    >>> func([1, 2, 3])
    14.0
    >>> func.gradient([1, 2, 3])
    rn(3).element([2.0, 4.0, 6.0])
    """
    if grad is not None and not isinstance(grad, Operator):
        grad_in = grad

        class SimpleFunctionalGradient(Operator):

            """Gradient of a `SimpleFunctional`."""

            def _call(self, x):
                """Return ``self(x)``."""
                return grad_in(x)

        grad = SimpleFunctionalGradient(space, space, linear=False)

    if cconj_grad is not None and not isinstance(cconj_grad, Operator):
        cconj_grad_in = cconj_grad

        class SimpleFunctionalCConjGradient(Operator):

            """Gradient of the convex conj of a  `SimpleFunctional`."""

            def _call(self, x):
                """Return ``self(x)``."""
                return cconj_grad_in(x)

        cconj_grad = SimpleFunctionalCConjGradient(space, space, linear=False)

    class SimpleFunctional(Functional):

        """A simplified functional for examples."""

        def __init__(self):
            """Initialize an instance."""
            super().__init__(space, linear=linear, grad_lipschitz=grad_lip)

        def _call(self, x):
            """Return ``self(x)``."""
            if fcall is None:
                raise NotImplementedError('call not implemented')
            else:
                return fcall(x)

        @property
        def proximal(self):
            """Return the proximal of the operator."""
            if prox is None:
                raise NotImplementedError('proximal not implemented')
            else:
                return prox

        @property
        def gradient(self):
            """Return the gradient of the operator."""
            if grad is None:
                raise NotImplementedError('gradient not implemented')
            else:
                return grad

        @property
        def convex_conj(self):
            return simple_functional(space, fcall=cconj_fcall, grad=cconj_grad,
                                     prox=cconj_prox, grad_lip=cconj_grad_lip,
                                     cconj_fcall=fcall, cconj_grad=grad,
                                     cconj_prox=prox, cconj_grad_lip=grad_lip,
                                     linear=linear)

    return SimpleFunctional()


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
