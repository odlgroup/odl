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
from numbers import Number

from odl.operator.operator import (
    Operator, OperatorComp, OperatorLeftScalarMult, OperatorRightScalarMult,
    OperatorRightVectorMult, OperatorSum)
from odl.operator.default_ops import (ResidualOperator, IdentityOperator,
                                      ConstantOperator, ScalingOperator)
from odl.set.space import LinearSpaceVector
from odl.solvers.advanced import (proximal_arg_scaling, proximal_translation,
                                  proximal_quadratic_perturbation)


# TODO: Add missing functionals here
__all__ = ('Functional', 'FunctionalLeftScalarMult',
           'FunctionalRightScalarMult', 'FunctionalComp',
           'FunctionalRightVectorMult', 'FunctionalSum', 'FunctionalScalarSum',
           'TranslatedFunctional', 'ConvexConjugateTranslation',
           'ConvexConjugateFuncScaling', 'ConvexConjugateArgScaling',
           'ConvexConjugateLinearPerturb')


class Functional(Operator):

    """Implementation of a functional class.

    Notes
    -----
    Note that the implementation of the functional class has assued that the
    functionals are defined on a Hilbert space, i.e., that we consider
    functionals

    .. math::

            f : X \\to F,

    where :math:`X` is a Hilbert space and :math:`F` is a field of scalars
    associated with :math:`X`. This has been done in order to simplify the
    concept of *convex conjugate functional*. Since Hilbert spaces as selfdual
    the convex conjugate functional is defined as

    .. math::

            f^* : X \\to F,

            f^*(x^*) = \\sup_{x} \{ \\langle x^*,x \\rangle - f(x)  \}.

    See, e.g., [Lue1969]_, [Roc1970]_ and [BC2011]_.
    """

    # TODO: Update doc above. What to write?

    def __init__(self, domain, linear=False, smooth=False, concave=False,
                 convex=False, grad_lipschitz=np.inf):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `Set`
            The domain of this functional, i.e., the set of elements to
            which this functional can be applied
        linear : `bool`
            If `True`, the functional is considered as linear. In this
            case, ``domain`` and ``range`` have to be instances of
            `LinearSpace`, or `Field`.
        smooth : `bool`, optional
            If `True`, assume that the functional is continuously
            differentiable
        concave : `bool`, optional
            If `True`, assume that the functional is concave
        convex : `bool`, optional
            If `True`, assume that the functional is convex
        grad_lipschitz : 'float', optional
            The Lipschitz constant of the gradient.
        """

        Operator.__init__(self, domain=domain,
                          range=domain.field, linear=linear)
        self._is_smooth = bool(smooth)
        self._is_convex = bool(convex)
        self._is_concave = bool(concave)
        self._grad_lipschitz = float(grad_lipschitz)

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

    def proximal(self, sigma=1.0):
        """Return the proximal operator of the functional.

        Parameters
        ----------
        sigma : positive float, optional
            Regularization parameter of the proximal operator.

        Returns
        -------
        out : `Operator`
            Domain and range equal to domain of functional.

        Notes
        -----
        The nonsmooth solvers that make use of proximal operators in order to
        solve a given optimization problem, see for example
        `forward_backward_pd`, take a `proximal factory` as input. Note that
        ``Functional.proximal`` is in fact a `proximal factory`.
        """
        raise NotImplementedError

    @property
    def conjugate_functional(self):
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
        https://en.wikipedia.org/wiki/Legendre_transformation

        For literature references see, e.g., [Lue1969]_, [Roc1970]_.
        """
        raise NotImplementedError

    def derivative(self, point):
        """Return the derivative operator in the given point.

        This function returns the linear operator

            ``x --> <x, grad_f(point)>``,

        where ``grad_f(point)`` is the gradient of the functional in the point
        ``point``.

        Parameters
        ----------
        point : `LinearSpaceVector`
            The point in which the gradient is evaluated.

        Returns
        -------
        out : `Operator`
            The linear operator that maps ``x --> <x, grad_f(point)>``.
        """
        return self.gradient(point).T

    def translate(self, shift):
        """Return a translation of the functional.

        For a given functional ``f`` and an element ``translation`` in the
        domain of ``f``, this operation creates the functional
        ``f(. - translation)``.

        Parameters
        ----------
        translation : `LinearSpaceVector`
            Element in the domain of the functional

        Returns
        -------
        out : `TranslatedFunctional`
            The functional ``f(. - translation)``
        """
        if shift not in self.domain:
            raise TypeError('the translation {} is not in the domain of the'
                            'functional {!r}'.format(shift, self))

        return TranslatedFunctional(self, shift)

    def __mul__(self, other):
        """Return ``self * other``.

        If ``other`` is an operator, this corresponds to composition with the
        operator:

            ``func * op <==> (x --> func(op(x)))``

        If ``other`` is a scalar, this corresponds to right multiplication of
        scalars with functionals:

            ``func * scalar <==> (x --> func(scalar * x))``

        If ``other`` is a vector, this corresponds to right multiplication of
        vectors with functionals:

            ``func * vector <==> (x --> func(vector * x))``

        Note that left and right multiplications are generally different.

        Parameters
        ----------
        other : `Operator` or `LinearSpaceVector` or scalar
            `Operator`:
            The `Operator.range` of ``other`` must match this functionals
            `Functional.domain`.

            `LinearSpaceVector`:
            ``other`` must be an element of this functionals's
            `Functional.domain`.

            scalar:
            The `Functional.domain` of this functional must be a
            `LinearSpace` and ``other`` must be an element of the ``field``
            of this functional's `Functional.domain`. Note that this
            ``field`` is also this functional's `Functional.range`.

        Returns
        -------
        mul : `Functional`
            Multiplication result

            If ``other`` is an `Operator`, ``mul`` is a
            `FunctionalComp`.

            If ``other`` is a scalar, ``mul`` is a
            `FunctionalRightScalarMult`.

            If ``other`` is a vector, ``mul`` is a
            `FunctionalRightVectorMult`.

        Notes
        -------
        See also `Operator.__mul__`.
        """
        if isinstance(other, Operator):
            return FunctionalComp(self, other)
        elif isinstance(other, Number):
            # Left multiplication is more efficient, so we can use this in the
            # case of linear functional.
            if self.is_linear:
                return FunctionalLeftScalarMult(self, other)
            else:
                return FunctionalRightScalarMult(self, other)
        elif isinstance(other, LinearSpaceVector) and other in self.domain:
            return FunctionalRightVectorMult(self, other)
        else:
            return super().__mul__(other)

    def __rmul__(self, other):
        """Return ``other * self``.

        If ``other`` is an operator, since a functional is also an operator
        this corresponds to operator composition:

            ``op * func <==> (x --> op(func(x))``

        If ``other`` is a scalar, this corresponds to left multiplication of
        scalars with functionals:

            ``scalar * func <==> (x --> scalar * func(x))``

        If ``other`` is a vector,  since a functional is also an operator this
        corresponds to left multiplication of vectors with operators:

            ``vector * func <==> (x --> vector * func(x))``

        Note that left and right multiplications are generally
        different.

        Parameters
        ----------
        other : `Operator` or `LinearSpaceVector` or scalar
            `Operator`:
            The `Operator.domain` of ``other`` must match this functional's
            `Functional.range`.

            `LinearSpaceVector`:
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
            Multiplication result

            If ``other`` is an `Operator`, ``rmul`` is a `OperatorComp`.

            If ``other`` is a scalar, ``rmul`` is a
            `FunctionalLeftScalarMult`.

            If ``other`` is a vector, ``rmul`` is a
            `OperatorLeftVectorMult`.
        """
        if other in self.domain.field:
            return FunctionalLeftScalarMult(self, other)
        else:
            return super().__rmul__(other)

    def __add__(self, other):
        """Return ``self + other``.

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
        add : `Functional`
            Addition result

            If ``other`` is in ``Functional.range``, ``add`` is a
            `FunctionalScalarSum`.

            If ``other`` is a `Functional`, ``add`` is a `FunctionalSum`.
        """
        if other in self.domain.field:
            return FunctionalScalarSum(self, other)
        elif isinstance(other, Functional):
            return FunctionalSum(self, other)
        else:
            return NotImplemented

    def __radd__(self, other):
        """Return ``other + self``.

        Since addition is commutative, also for functionals, this is similar
        to `Functional.__add__`.
        """
        if other in self.domain.field:
            return FunctionalScalarSum(self, other)
        elif isinstance(other, Functional):
            return FunctionalSum(self, other)
        else:
            return NotImplemented

    # TODO: Do we want this? What if one is a Functional? What happens then?
    # What can we say?
    def __sub__(self, other):
        """Return ``self - other``."""
        return self.__add__(-1 * other)

    @property
    def is_smooth(self):
        """`True` if this functional is continuously differentiable."""
        return self._is_smooth

    @property
    def is_concave(self):
        """`True` if this functional is concave."""
        return self._is_concave

    @property
    def is_convex(self):
        """`True` if this functional is convex."""
        return self._is_convex

    @property
    def grad_lipschitz(self):
        """Lipschitz constant for the gradient of the functional"""
        return self._grad_lipschitz


class FunctionalLeftScalarMult(Functional, OperatorLeftScalarMult):

    """Scalar multiplication of functional from the left.

    Given a functional ``f`` and a scalar ``scalar``, this represents the
    functional

        ``(scalar * f)(x) = scalar * f(x)``.
    """

    def __init__(self, func, scalar):
        """Initialize a new instance.

        Parameters
        ----------
        func : `Functional`
            The functional that is to be scaled.
        scalar : `float`
            The scalar with which the functional is scaled.
        """

        if not isinstance(func, Functional):
            raise TypeError('functional {!r} is not a Functional instance.'
                            ''.format(func))

        scalar = func.range.element(scalar)

        if scalar > 0:
            Functional.__init__(self, domain=func.domain,
                                linear=func.is_linear,
                                smooth=func.is_smooth,
                                concave=func.is_concave,
                                convex=func.is_convex,
                                grad_lipschitz=(scalar * func.grad_lipschitz))
        elif scalar == 0:
            Functional.__init__(self, domain=func.domain,
                                linear=True,
                                smooth=True,
                                concave=True, convex=True,
                                grad_lipschitz=0)
        elif scalar < 0:
            Functional.__init__(self, domain=func.domain,
                                linear=func.is_linear,
                                smooth=func.is_smooth,
                                concave=func.is_convex,
                                convex=func.is_concave,
                                grad_lipschitz=(-scalar * func.grad_lipschitz))
        else:
            # It should not be possible to get here
            raise TypeError('comparison with scalar {} failed'.format(scalar))

        OperatorLeftScalarMult.__init__(self, operator=func, scalar=scalar)

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
        return self.scalar * self.operator.gradient

    @property
    def conjugate_functional(self):
        """Convex conjugate functional of the scaled functional."""
        # The helper function only allows positive scaling parameters.
        # Otherwise it gives an error.
        return ConvexConjugateFuncScaling(
            self.operator.conjugate_functional, self.scalar)

    def proximal(self, sigma=1.0):
        """Return the proximal operator of the scaled functional.

        Parameters
        ----------
        sigma : positive float, optional
            Regularization parameter of the proximal operator.

        Returns
        -------
        out : `Operator`
            Domain and range equal to domain of functional.
        """

        # TODO: This is a bit "to clever" (and potentially wrong... it assumes
        # that you always start with a convex functional). If a convex
        # functional is scaled with a negative scalar, we should not allow to
        # call for proximal (raise a ValueError). However, should we check for
        # convex here instead? FunctionalLeftScalarMult keeps track of convex
        # when initialized, depending on sign of the scalar.

        sigma = float(sigma)
        if sigma * self.scalar < 0:
            raise ValueError('The step lengt {} times the scalar {} needs to '
                             'be nonnegative.'.format(sigma, self.scalar))
        return self.operator.proximal(sigma * self.scalar)

    def derivative(self, point):
        """Returns the derivative operator in the given point.

        This function returns the linear operator

            ``x --> <x, grad_f(point)>``,

        where ``grad_f(point)`` is the gradient of the functional in the point
        ``point``.

        Parameters
        ----------
        point : `LinearSpaceVector`
            The point in which the gradient is evaluated.

        Returns
        -------
        out : `DerivativeOperator`
            The linear operator that maps ``x --> <x, grad_f(point)>``.
        """
        return self.scalar * self.operator.derivative(point)


class FunctionalRightScalarMult(Functional, OperatorRightScalarMult):

    """Scalar multiplication of the argument of functional.

    Given a functional ``f`` and a scalar ``scalar``, this represents the
    functional

        ``(f * scalar)(x) = f(scalar * x)``.
    """

    def __init__(self, func, scalar):
        """Initialize a new instance.

        Parameters
        ----------
        func : `Functional`
            The functional which will have its argument scaled.
        scal : `Scalar`
            The scaling parameter with which the argument is scaled.
        """

        if not isinstance(func, Functional):
            raise TypeError('functional {!r} is not a Functional instance.'
                            ''.format(func))

        scalar = func.range.element(scalar)

        if scalar == 0:
            Functional.__init__(self, domain=func.domain, linear=True,
                                smooth=True, concave=True, convex=True,
                                grad_lipschitz=0)
        elif scalar < 0 or scalar > 0:
            Functional.__init__(self, domain=func.domain,
                                linear=func.is_linear, smooth=func.is_smooth,
                                concave=func.is_concave, convex=func.is_convex,
                                grad_lipschitz=(np.abs(scalar) *
                                                func.grad_lipschitz))
        else:
            # It should not be possible to get here
            raise TypeError('comparison with scalar {} failed'.format(scalar))

        OperatorRightScalarMult.__init__(self, operator=func, scalar=scalar)

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
        return (self.scalar * self.operator.gradient *
                ScalingOperator(self.domain, self.scalar))

    @property
    def conjugate_functional(self):
        """Convex conjugate functional of functional with scaled argument."""
        return ConvexConjugateArgScaling(self.operator.conjugate_functional,
                                         self.scalar)

    def proximal(self, sigma=1.0):
        """Return the proximal operator of the functional with scaled argument.

        Parameters
        ----------
        sigma : positive float, optional
            Regularization parameter of the proximal operator.

        Returns
        -------
        out : `Operator`
            Domain and range equal to domain of functional.
        """
        sigma = float(sigma)
        return(proximal_arg_scaling(
            self.operator.proximal, self.scalar))(sigma)

    def derivative(self, point):
        """Returns the derivative operator in the given point.

        This function returns the linear operator

            ``x --> <x, grad_f(point)>``,

        where ``grad_f(point)`` is the gradient of the functional in the point
        ``point``.

        Parameters
        ----------
        point : `LinearSpaceVector`
            The point in which the gradient is evaluated.

        Returns
        -------
        out : `DerivativeOperator`
            The linear operator that maps ``x --> <x, grad_f(point)>``.
        """
        return self.scalar * self.operator.derivative(self.scalar * point)


class FunctionalComp(Functional, OperatorComp):

    """Composition of a functional with an operator.

    Given a functional ``func`` and an operator ``op``, such that the range of
    the operator is equal to the domain of the functional, this corresponds to
    the functional

        ``(func * op)(x) = func(op(x))``.
    """

    def __init__(self, func, op, tmp1=None, tmp2=None):
        """Initialize a new instance.

        Parameters
        ----------
        func : `Functional`
            The left ("outer") operator
        op : `Operator`
            The right ("inner") operator. Its range must coincide with the
            domain of ``func``.
        tmp1 : `element` of the range of ``op``, optional
            Used to avoid the creation of a temporary when applying ``op``
        tmp2 : `element` of the range of ``op``, optional
            Used to avoid the creation of a temporary when applying the
            gradient of ``func``
        """
        if not isinstance(func, Functional):
            raise TypeError('functional {!r} is not a Functional instance.'
                            ''.format(func))

        OperatorComp.__init__(self, left=func, right=op, tmp=tmp1)

        Functional.__init__(self, domain=func.domain,
                            linear=(func.is_linear and op.is_linear),
                            smooth=(func.is_smooth and op.is_linear),
                            concave=(func.is_concave and op.is_linear),
                            convex=(func.is_convex and op.is_linear),
                            grad_lipschitz=np.infty)

        if tmp2 is not None and tmp2 not in self._left.domain:
            raise TypeError('second temporary {!r} not in the domain '
                            '{!r} of the functional.'
                            ''.format(tmp2, self._left.domain))
        self._tmp2 = tmp2

    @property
    def gradient(self):
        """Gradient of the compositon according to the chain rule."""

        func = self.left
        op = self.right

        class CompositGradient(Operator):
            """Gradient of the compositon according to the chain rule."""
            def __init__(self):
                """Initialize a new instance."""
                super().__init__(func.domain, func.domain, linear=False)
                self._func = func
                self._op = op

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
                return self._op.derivative(x).adjoint(
                    self._func.gradient(self._op(x)))

        return CompositGradient()


class FunctionalRightVectorMult(Functional, OperatorRightVectorMult):

    """Expression type for the functional right vector multiplication.

    Given a functional ``func`` and a vector ``y`` in the domain of ``func``,
    this corresponds to the functional

        ``(func * y)(x) = func(y*x)``.
    """

    def __init__(self, func, vector):
        """Initialize a new instance.

        Parameters
        ----------
        func : `Functional`
            The domain of ``func`` must be a ``vector.space``.
        vector : `LinearSpaceVector` in ``func.domain``
            The vector to multiply by.
        """
        if not isinstance(func, Functional):
            raise TypeError('functional {!r} is not a Functional instance.'
                            ''.format(func))

        OperatorRightVectorMult.__init__(self, operator=func, vector=vector)

        # TODO: can some of the parameters convex, etc. be decided?
        Functional.__init__(self, domain=func.domain)

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
        return self.vector * self.operator.gradient * self.vector

    # TODO: can this be computed?
    def proximal(self, sigma=1.0):
        """Return the proximal operator of the functional.

        Parameters
        ----------
        sigma : positive float, optional
            Regularization parameter of the proximal operator.

        Returns
        -------
        out : `Operator`
            Domain and range equal to domain of functional.

        Notes
        -----
        The nonsmooth solvers that make use of proximal operators in order to
        solve a given optimization problem, see for example
        `forward_backward_pd`, take a `proximal factory` as input. Note that
        ``Functional.proximal`` is in fact a `proximal factory`.
        """
        raise NotImplementedError('there is no known expression for this')

    @property
    def conjugate_functional(self):
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
        https://en.wikipedia.org/wiki/Legendre_transformation

        For literature references see, e.g., [Lue1969]_, [Roc1970]_.
        """
        return self.operator.conjugate_functional * (1.0 / self.vector)


class FunctionalSum(Functional, OperatorSum):

    """Expression type for the sum of functionals.

    ``FunctionalSum(func1, func2) <==> (x --> func1(x) + func2(x))``.
    """

    def __init__(self, func1, func2, tmp_dom=None):
        """Initialize a new instance.

        Parameters
        ----------
        func1, func2 : `Functional`
            The summands of the functional sum. Their `Operator.domain`
            and `Operator.range` must coincide.
        tmp_dom : `Operator.domain` `element`, optional
            Used to avoid the creation of a temporary when applying the
            gradient.
        """
        # TODO: Tmp_dom, what is it used for?
        if not isinstance(func1, Functional):
            raise TypeError('functional 1 {!r} is not a Functional instance.'
                            ''.format(func1))
        if not isinstance(func2, Functional):
            raise TypeError('functional 2 {!r} is not a Functional instance.'
                            ''.format(func2))
        # TODO: Is this needed, or should it be left for OperatorSum to handle?
        if func1.range != func2.range:
            raise TypeError('the ranges of the functionals {!r} and {!r} do '
                            'not match'.format(func1.range, func2.range))
        if func1.domain != func2.domain:
            raise TypeError('the domains of the functionals {!r} and {!r} do '
                            'not match'.format(func1.domain, func2.domain))

        Functional.__init__(self, domain=func1.domain,
                            linear=(func1.is_linear and func2.is_linear),
                            smooth=(func1.is_smooth and func2.is_smooth),
                            concave=(func1.is_concave and func2.is_concave),
                            convex=(func1.is_convex and func2.is_convex),
                            grad_lipschitz=(func1.grad_lipschitz +
                                            func2.grad_lipschitz))

        OperatorSum.__init__(self, func1, func2, tmp_ran=None,
                             tmp_dom=tmp_dom)

    @property
    def gradient(self):
        """Gradient operator of functional sum."""
        return self.left.gradient + self.right.gradient

    def derivative(self, point):
        """Returns the derivative operator in the given point.

        This function returns the linear operator

            ``x --> <x, grad_f(point)>``,

        where ``grad_f(point)`` is the gradient of the functional in the point
        ``point``.

        Parameters
        ----------
        point : `LinearSpaceVector`
            The point in which the gradient is evaluated.

        Returns
        -------
        out : `DerivativeOperator`
            The linear operator that maps ``x --> <x, grad_f(point)>``.
        """
        return self._op1.derivative(point) + self._op2.derivative(point)


class FunctionalScalarSum(Functional, OperatorSum):

    """Expression type for the sum of a functional and a scalar.

    ``FunctionalScalarSum(func, scalar) <==> (x --> func(x) + scalar)``
    """

    def __init__(self, func, scalar, tmp_dom=None):
        """Initialize a new instance.

        Parameters
        ----------
        func : `Functional`
            Functional to which the scalar is added.
        scalar : `element` in the `field` of the ``domain``
            The scalar to be added to the functional. The `field` of the
            ``domain`` is the range of the functional.
        tmp_dom : `LinearSpaceVector`, optional
            ...
        """
        # TODO: what is tmp_dom used for?
        if not isinstance(func, Functional):
            raise TypeError('functional {!r} is no a Functional instance'
                            ''.format(func))
        if scalar not in func.range:
            raise TypeError('the scalar {} is not in the range of the'
                            'functional {!r}'.format(scalar, func))

        Functional.__init__(self, domain=func.domain, linear=func.is_linear,
                            smooth=func.is_smooth, concave=func.is_concave,
                            convex=func.is_convex,
                            grad_lipschitz=func.grad_lipschitz)

        OperatorSum.__init__(self, left=func,
                             right=ConstantOperator(vector=scalar,
                                                    domain=func.domain,
                                                    range=func.range),
                             tmp_ran=None, tmp_dom=tmp_dom)

    # TODO: Update this if ConstantOperator is updated.
    @property
    def scalar(self):
        return self.right.vector

    @property
    def gradient(self):
        """Gradient operator of the functional.

        Note that this is the same as the gradient of the original functional.

        Notes
        -----
        The operator that corresponds to the mapping

        .. math::

            x \\to \\nabla f(x)

        where :math:`\\nabla f(x)` is the element used to evaluate
        derivatives in a direction :math:`d` by
        :math:`\\langle \\nabla f(x), d \\rangle`.
        """
        return self.left.gradient

    def proximal(self, sigma=1.0):
        """Proximal operator of the FunctionalScalarSum.

        This is the same as the proximal operator of the original
        functional.

        Parameters
        ----------
        sigma : positive float, optional
            Regularization parameter of the proximal operator.

        Returns
        -------
        out : `Operator`
            Domain and range equal to domain of functional.
        """
        return self.left.proximal(sigma)

    @property
    def conjugate_functional(self):
        """Convex conjugate functional of FunctionalScalarSum."""
        return self.left.conjugate_functional - self.scalar

    def derivative(self, point):
        """Returns the derivative operator of FunctionalScalarSum.

        Parameters
        ----------
        point : `LinearSpaceVector`
            The point in which the gradient is evaluated.

        Returns
        -------
        out : `DerivativeOperator`
            The linear operator that maps ``x --> <x, grad_f(point)>``.
        """
        return self.left.derivative(point)


class TranslatedFunctional(Functional):

    """Implementation of the translated functional.

    Given a functional ``f`` and an element ``translation`` in the domain of
    ``f``, this corresponds to the functional ``f(. - translation)``.
    """

    # TODO: Should we check type of func, and if it is TranslatedFunctional
    # try to combined to only one TranslatedFunctional with a total
    # translation?
    def __init__(self, func, translation):
        """Initialize a new instance.

        Given a functional ``f(.)`` and a vector ``translation`` in the domain
        of ``f``, this corresponds to the functional ``f(. - y)``.

        Parameters
        ----------
        func : `Functional`
            Functional which is to be translated.
        translation : `LinearSpaceVector`
            Element in ``func.domain``, with which the argument is translated.
        """

        if not isinstance(func, Functional):
            raise TypeError('`func` {!r} not a `Functional` instance'
                            ''.format(func))

        if translation not in func.domain:
            raise TypeError('`translation` {!r} not in func.domain {!r}'
                            ''.format(translation.space, func.domain))

        super().__init__(domain=func.domain, linear=False,
                         smooth=func.is_smooth,
                         concave=func.is_concave,
                         convex=func.is_convex,
                         grad_lipschitz=func.grad_lipschitz)

        self._original_func = func
        self._translation = translation

    @property
    def original_func(self):
        return self._original_func

    @property
    def translation(self):
        return self._translation

    def _call(self, x):
        """Evaluates the functional in a point ``x``.

        Parameters
        ----------
        x : `LinearSpaceVector`
            Element in the domain of the functional.
            The point in which the functional is evaluated

        Returns
        -------
        `self(x)` : `float`
            Evaluation of the functional, which is a constant.
        """
        return self._original_func(x - self._translation)

    @property
    def gradient(self):
        """Gradient operator of the functional.

        The operator is given by a transation of the gradient operator of the
        original functional.

        Notes
        -----
        The operator that corresponds to the mapping

        .. math::

            x \\to \\nabla f(x)

        where :math:`\\nabla f(x)` is the element used to evaluate
        derivatives in a direction :math:`d` by
        :math:`\\langle \\nabla f(x), d \\rangle`.
        """

        # TODO: Update doc below.
        class TranslatedGradientOperator(Operator):

            """The gradient operator for a translated functional."""

            def __init__(self, translated_func, translation):
                """Initialize a new instance.

                Parameters
                ----------
                translated_func : `Functional`
                    Functional corresponding to ``F(. - translation)``
                translation : `LinearSpaceVector`
                    The translation with which ``translated_func`` has been
                    translated.

                """
                super().__init__(domain=translated_func.domain,
                                 range=translated_func.domain,
                                 linear=(translated_func.is_concave and
                                         translated_func.is_convex))

                self._original_grad = translated_func._original_func.gradient
                self._translation = translation

            def _call(self, x):
                """Evaluates the gradient in a point ``x``.

                Parameters
                ----------
                x : `LinearSpaceVector`
                    Element in the domain of the operator.

                Returns
                -------
                out : `LinearSpaceVector`
                    The gradient in the point ``x``. An element in the doimain
                    of the operator.
                """
                return self._original_grad(x - self._translation)

        return TranslatedGradientOperator(self, self._translation)

    def proximal(self, sigma=1.0):
        """Return the proximal operator of the translated functional.

        Parameters
        ----------
        sigma : positive float, optional
            Regularization parameter of the proximal operator.

        Returns
        -------
        out : `Operator`
            Domain and range equal to domain of functional.
        """
        return (proximal_translation(self._original_func.proximal,
                                     self._translation))(sigma)

    @property
    def conjugate_functional(self):
        """Convex conjugate functional of the translated functional."""
        return ConvexConjugateTranslation(
            self._original_func.conjugate_functional,
            self._translation)

    def derivative(self, point):
        """Returns the derivative operator, evaluated in the point
        ``point - translation``.

        Parameters
        ----------
        point : `LinearSpaceVector`
            The point in which the gradient is evaluated.

        Returns
        -------
        out : `Operator`
            The linear operator that maps ``x --> <x, grad_f(p)>``.
        """
        return self._original_func.derivative(point - self._translation)


class ConvexConjugateTranslation(Functional):

    """ The ``Functional`` representing ``(F( . - translation))^*``.

    This is a functional representing the conjugate functional of the
    translated function ``F(. - translation)``. It is calculated according to
    the rule

        ``(F( . - translation))^* (x) = F^*(x) + <translation, x>``,

    where ``translation`` is the translation of the argument.

    Parameters
    ----------
    convex_conj_f : `Functional`
        Function corresponding to ``F^*``.

    translation : `LinearSpaceVector`
        Element in domain of ``F^*``..

    Notes
    -----
    For reference on the identity used, see [KP2015]_.
    """

    def __init__(self, convex_conj_f, translation):
        """Initialize a new instance.

        Parameters
        ----------
        convex_conj_f : `Functional`
            Function corresponding to F^*.

        translation : `LinearSpaceVector`
            Element in domain of ``F^*``.
        """

        if translation is not None and not isinstance(translation,
                                                      LinearSpaceVector):
            raise TypeError(
                'vector {!r} not None or a LinearSpaceVector instance.'
                ''.format(translation))

        if translation not in convex_conj_f.domain:
            raise TypeError(
                'vector {} not in the domain of the functional {}.'
                ''.format(translation, convex_conj_f.domain))

        super().__init__(domain=convex_conj_f.domain,
                         linear=convex_conj_f.is_linear,
                         smooth=convex_conj_f.is_smooth,
                         concave=convex_conj_f.is_concave,
                         convex=convex_conj_f.is_convex)

        self._orig_convex_conj_f = convex_conj_f
        self._translation = translation

    @property
    def orig_convex_conj_f(self):
        return self._orig_convex_conj_f

    @property
    def translation(self):
        return self._translation

        # TODO:
        # The Lipschitz constant for the gradient can be bounded, by using
        # triangle inequality. However: is it the tightest bound?

    def _call(self, x):
        """Applies the functional to the given point.

        Parameters
        ----------
        x : `LinearSpaceVector`
            Element in the domain of the functional.
            The point in which the functional is evaluated

        Returns
        -------
        `self(x)` : `element` in the `field` of the ``domain``.
            Evaluation of the functional.
        """
        return self.orig_convex_conj_f(x) + x.inner(self.translation)

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
        return (self.orig_convex_conj_f.gradient +
                ConstantOperator(self.translation))

    def proximal(self, sigma=1.0):
        """Return the proximal operator of the ConvexConjugateTranslation
        functional.

        Parameters
        ----------
        sigma : positive float, optional
            Regularization parameter of the proximal operator

        Returns
        -------
        out : Operator
            Domain and range equal to domain of functional
        """
        return proximal_quadratic_perturbation(
            self.orig_convex_conj_f.proximal, a=0, u=self.translation)(sigma)

    # TODO: Should it be added?
    # Note: This would only be valide when f is proper convex and lower-
    # semincontinuous.
    @property
    def conjugate_functional(self):
        """Convex conjugate functional of the functional."""
        raise NotImplementedError


class ConvexConjugateFuncScaling(Functional):

    """ The ``Functional`` representing ``(scaling * F(.))^*``.

    This is a functional representing the conjugate functional of the scaled
    function ``scaling * F(.)``. This is calculated according to the rule

        ``(scaling * F(.))^* (x) = scaling * F^*(x/scaling)``,

    where ``scaling`` is the scaling parameter. Note that scaling is only
    allowed with strictly positive scaling parameters.

    Parameters
    ----------
    convex_conj_f : `Functional`
        Function corresponding to ``F^*``.

    scaling : `float`, positive
        Positive scaling parameter.

    Notes
    -----
    For reference on the identity used, see [KP2015]_.
    """

    def __init__(self, convex_conj_f, scaling):
        """Initialize a new instance.

        Parameters
        ----------
        convex_conj_f : `Functional`
            Functional corresponding to ``F^*``.

        scaling : `float`, positive
            Positive scaling parameter.
        """

        # TODO: scaling with zero gives the zero-functional. Should this be ok?
        scaling = float(scaling)
        if scaling <= 0:
            raise ValueError(
                'Scaling with nonpositive values is not allowed. Current '
                'value: {}.'.format(scaling))

        super().__init__(domain=convex_conj_f.domain,
                         linear=convex_conj_f.is_linear,
                         smooth=convex_conj_f.is_smooth,
                         concave=convex_conj_f.is_concave,
                         convex=convex_conj_f.is_convex)

        self._orig_convex_conj_f = convex_conj_f
        self._scaling = scaling

    @property
    def orig_convex_conj_f(self):
        return self._orig_convex_conj_f

    @property
    def scaling(self):
        return self._scaling

    def _call(self, x):
        """Applies the functional to the given point.

        Parameters
        ----------
        x : `LinearSpaceVector`
            Element in the domain of the functional.
            The point in which the functional is evaluated

        Returns
        -------
        `self(x)` : `float`
            Evaluation of the functional.
        """
        return self.scaling * self.orig_convex_conj_f(x * (1.0 / self.scaling))

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
        return self.orig_convex_conj_f.gradient * (1.0 / self.scaling)

    def proximal(self, sigma=1.0):
        """Return the proximal operator of the ConvexConjugateFuncScaling
        functional.

        Parameters
        ----------
        sigma : positive float, optional
            Regularization parameter of the proximal operator

        Returns
        -------
        out : Operator
            Domain and range equal to domain of functional
        """
        return proximal_arg_scaling(self.orig_convex_conj_f.proximal,
                                    scaling=(1.0 / self.scaling)
                                    )(self.scaling * sigma)

    # TODO: Should it be added?
    # Note: This would only be valide when f is proper convex and lower-
    # semincontinuous.
    @property
    def conjugate_functional(self):
        """Convex conjugate functional of the functional."""
        raise NotImplementedError


class ConvexConjugateArgScaling(Functional):

    """ The ``Functional`` representing ``(F( . * scaling))^*``.

    This is a functional representing the conjugate functional of the
    functional with scaled arguement: ``F(. * scaling)``. This is calculated
    according to the rule

        ``(F( . * scaling))^*(x) = F^*(x/scaling)``

    where ``scaling`` is the scaling parameter. Note that this does not allow
    for scaling with ``0``.

    Parameters
    ----------
    convex_conj_f : `Functional`
        Function corresponding to ``F^*``.

    scaling : `float`, nonzero
        The scaling parameter.

    Notes
    -----
    For reference on the identity used, see [KP2015]_.
    """

    def __init__(self, convex_conj_f, scaling):
        """Initialize a new instance.

        Parameters
        ----------
        convex_conj_f : `Functional`
            Functional corresponding to ``F^*``.

        scaling : 'float', nonzero
            The scaling parameter.
        """

        scaling = float(scaling)
        if scaling == 0:
            raise ValueError('Scaling with 0 is not allowed. Current value:'
                             ' {}.'.format(scaling))

        super().__init__(domain=convex_conj_f.domain,
                         linear=convex_conj_f.is_linear,
                         smooth=convex_conj_f.is_smooth,
                         concave=convex_conj_f.is_concave,
                         convex=convex_conj_f.is_convex)

        self._orig_convex_conj_f = convex_conj_f
        self._scaling = scaling

    @property
    def orig_convex_conj_f(self):
        return self._orig_convex_conj_f

    @property
    def scaling(self):
        return self._scaling

    def _call(self, x):
        """Applies the functional to the given point.

        Parameters
        ----------
        x : `LinearSpaceVector`
            Element in the domain of the functional.
            The point in which the functional is evaluated

        Returns
        -------
        `self(x)` : `float`
            Evaluation of the functional.
        """
        return self.orig_convex_conj_f(x * (1.0 / self.scaling))

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
        return ((1.0 / self.scaling) * self.orig_convex_conj_f.gradient *
                (1.0 / self.scaling))

    def proximal(self, sigma=1.0):
        """Return the proximal operator of the ConvexConjugateArgScaling
        functional.

        Parameters
        ----------
        sigma : positive float, optional
            Regularization parameter of the proximal operator

        Returns
        -------
        out : Operator
            Domain and range equal to domain of functional
        """
        return proximal_arg_scaling(self.orig_convex_conj_f.proximal,
                                    scaling=(1.0 / self.scaling))(sigma)

    # TODO: Should it be added?
    # Note: This would only be valide when f is proper convex and lower-
    # semincontinuous.
    @property
    def conjugate_functional(self):
        """Convex conjugate functional of the functional."""
        raise NotImplementedError


class ConvexConjugateLinearPerturb(Functional):

    """ The ``Functional`` representing ``(F(.) + <y,.>)^*``.


    This is a functional representing the conjugate functional of the linearly
    perturbed functional ``F(. * scaling) + <y,.>``. This is calculated
    according to the rule

        ``(F(.) + <y,.>)^* (x) = F^*(x - y)``

    where ``y`` is the linear perturbation.

    Parameters
    ----------
    convex_conj_f : `Functional`
        Function corresponding to ``F^*``.

    y : `LinearSpaceVector`
        Element in domain of ``F^*``.

    Notes
    -----
    For reference on the identity used, see [KP2015]_. Note that this is only
    valide for functionals with a domain that is a Hilbert space.
    """

    def __init__(self, convex_conj_f, y):
        """Initialize a new instance.

        Parameters
        ----------
        convex_conj_f : `Functional`
            Functional corresponding to ``F^*``.

        y : `LinearSpaceVector`
            Element in domain of ``F^*``.
        """
        if not isinstance(y, LinearSpaceVector):
            raise TypeError('vector {!r} not a LinearSpaceVector instance.'
                            ''.format(y))

        if y not in convex_conj_f.domain:
            raise TypeError('vector {!r} not in the domain of the functional '
                            '{!r}.'.format(y, convex_conj_f.domain))

        super().__init__(domain=convex_conj_f.domain,
                         linear=convex_conj_f.is_linear,
                         smooth=convex_conj_f.is_smooth,
                         concave=convex_conj_f.is_concave,
                         convex=convex_conj_f.is_convex)

        self._orig_convex_conj_f = convex_conj_f
        self._y = y

    @property
    def orig_convex_conj_f(self):
        return self._orig_convex_conj_f

    @property
    def y(self):
        return self._y

    def _call(self, x):
        """Applies the functional to the given point.

        Parameters
        ----------
        x : `LinearSpaceVector`
            Element in the domain of the functional.
            The point in which the functional is evaluated

        Returns
        -------
        `self(x)` : `float`
            Evaluation of the functional.
        """
        return self.orig_convex_conj_f(x - self._y)

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
        return (self.orig_convex_conj_f.gradient *
                ResidualOperator(IdentityOperator(self.domain), self.y))

    def proximal(self, sigma=1.0):
        """Return the proximal operator of the ConvexConjugateLinearPerturb
        functional.

        Parameters
        ----------
        sigma : positive float, optional
            Regularization parameter of the proximal operator

        Returns
        -------
        out : Operator
            Domain and range equal to domain of functional
        """
        return proximal_translation(self.orig_convex_conj_f.proximal,
                                    self.y)(sigma)

    # TODO: Should it be added?
    # Note: This would only be valide when f is proper convex and lower-
    # semincontinuous.
    @property
    def conjugate_functional(self):
        """Convex conjugate functional of the functional."""
        raise NotImplementedError
