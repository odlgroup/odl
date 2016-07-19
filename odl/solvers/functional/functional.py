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
                                      ConstantOperator)
from odl.set.space import LinearSpaceVector
from odl.solvers.advanced import (proximal_arg_scaling, proximal_translation,
                                  proximal_quadratic_perturbation)


__all__ = ('Functional', 'ConvexConjugateArgScaling',
           'ConvexConjugateFuncScaling', 'ConvexConjugateLinearPerturb',
           'ConvexConjugateTranslation', 'TranslatedFunctional')


# TODO: What if derivative is implemeted and not gradient?
# In this case we would like to use this in scalar-multiplication, etc.

class Functional(Operator):
    """Implementation of a functional class."""

    # TODO: Update doc above.

    def __init__(self, domain, linear=False, smooth=False, concave=False,
                 convex=False, grad_lipschitz=np.inf):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `Set`
            The domain of this operator, i.e., the set of elements to
            which this operator can be applied
        linear : `bool`
            If `True`, the operator is considered as linear. In this
            case, ``domain`` and ``range`` have to be instances of
            `LinearSpace`, or `Field`.
        domain : `LinearSpace`
            Set of elements on which the functional can be evaluated
        smooth : `bool`, optional
            If `True`, assume that the functional is continuously
            differentiable
        convex : `bool`, optional
            If `True`, assume that the functional is convex
        concave : `bool`, optional
            If `True`, assume that the functional is concave
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

        where :math:`\\nabla f(x)` is the element used to evaluated
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
        Note that ``Functional.proximal`` is a `Proximal factory`.
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

            f^*(x^*) = \\sup_{x} \{ \\langle x^*,y \\rangle - f(x)  \}.

        See, e.g., [Lue1969]_, [Roc1970]_.
        """
        raise NotImplementedError

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
        functional = self

        class DerivativeOperator(Functional):
            def __init__(self):
                super().__init__(functional.domain, linear=True)

            self.point = point

            def _call(self, x):
                return x.inner(functional.gradient(point))

        return DerivativeOperator()

    def translate(self, translation):
        """Creates a translation of the functional.

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
        if translation not in self.domain:
            raise TypeError('the translation {} is not in the domain of the'
                            'functional {!r}'.format(translation, self))

        return TranslatedFunctional(self, translation)

    def __mul__(self, other):
        """Return ``self * other``.

        If ``other`` is an operator, this corresponds to composition with the
        operator:

            ``func * op <==> (x --> func(op(x))``

        If ``other`` is a scalar, this corresponds to right multiplication of
        scalars with functionals:

            ``func * scalar <==> (x --> func(scalar * x))``

        If ``other`` is a vector, this corresponds to right multiplication of
        vectors with functionals:

            ``func * vector <==> (x --> func(vector * x))``

        Note that left and right multiplications are generally different.

        Parameters
        ----------
        other : {`Operator`, `LinearSpaceVector`, scalar}
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

        """
        if isinstance(other, Operator):
            return FunctionalComp(self, other)
        elif isinstance(other, Number):
            # Left multiplication is more efficient, so we can use this in the
            # case of linear operator.
            if self.is_linear:
                return FunctionalLeftScalarMult(self, other)
            else:
                return FunctionalRightScalarMult(self, other)
        else:
            super().__mul__(self, other)

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
        other : {`Operator`, `LinearSpaceVector`, scalar}
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
        rmul : {`Functional`, `Operator`}
            Multiplication result

            If ``other`` is an `Operator`, ``mul`` is a `OperatorComp`.

            If ``other`` is a scalar, ``mul`` is a
            `FunctionalLeftScalarMult`.

            If ``other`` is a vector, ``mul`` is a
            `OperatorLeftVectorMult`.
        """
        if other in self.domain.field:
            return FunctionalLeftScalarMult(self, other)
        else:
            super().__rmul__(self, other)

    def __add__(self, other):
        """Return ``self + other``.

        Parameters
        ----------
        other : {`Functional`, scalar}
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

        See `Functional.__add__`.
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
        """`True` if this operator is continuously differentiable."""
        return self._is_smooth

    @property
    def is_concave(self):
        """`True` if this operator is concave."""
        return self._is_concave

    @property
    def is_convex(self):
        """`True` if this operator is convex."""
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

        OperatorLeftScalarMult.__init__(self, op=func, scalar=scalar)

        self._func = func
        self._scalar = scalar

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

        functional = self

        class LeftScalarMultGradient(Operator):
            """The gradient operator for a `FunctionalLeftScalarMult`
            functional.
            """
            def __init__(self):
                """Initialize a `LeftScalarMultGradient` instance."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)

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
                return functional._scalar * functional._func.gradient(x)

        return LeftScalarMultGradient()

    @property
    def conjugate_functional(self):
        """Convex conjugate functional of the scaled functional."""
        # The helper function only allows positive scaling parameters.
        # Otherwise it gives an error.
        return ConvexConjugateFuncScaling(
                self._func.conjugate_functional, self._scalar)

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
        sigma = float(sigma)
        if sigma * self._scalar < 0:
            raise ValueError('The step lengt {} times the scalar {} needs to '
                             'be nonnegative.'.format(sigma, self._scalar))
        return self._func.proximal(sigma * self._scalar)

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
        return self._scalar * self._func.derivative(point)


class FunctionalRightScalarMult(Functional, OperatorRightScalarMult):
    """Scalar multiplication of the argument of functional.

    Given a functional ``f`` and a scalar ``scalar``, this represents the
    functional

        ``(f * scalar)(x) = f(scalar * x)``.
    """

    def __init__(self, func, scalar):
        """Initialize a FunctionalRightScalarMult instance.

        Parameters
        ----------
        scal : `Scalar`
            Scalar argument
        func : `Functional`
            The left ("outer") functional
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

        OperatorRightScalarMult.__init__(self, op=func, scalar=scalar)

        self._func = func
        self._scalar = scalar

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

        functional = self

        class RightScalarMultGradient(Operator):
            """The gradient operator for a `FunctionalRightScalarMult`
            functional.
            """

            def __init__(self):
                """Initialize a `LeftScalarMultGradient` instance."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)

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
                return (functional._scalar *
                        functional._func.gradient(functional._scalar * x))

        return RightScalarMultGradient()

    @property
    def conjugate_functional(self):
        """Convex conjugate functional of functional with scaled argument."""
        return ConvexConjugateArgScaling(self._func.conjugate_functional,
                                         self._scalar)

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
        return (proximal_arg_scaling(self._func.proximal, self._scalar))(sigma)

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
        return self._func.derivative(point) * self._scalar


class FunctionalComp(Functional, OperatorComp):
    """Composition of a functional with an operator."""

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

    # Should this one really take an x-argument? Isn't it suppose to return the
    # operator itself?
    def gradient(self, x, out=None):
        """Gradient of the compositon according to the chain rule.

        Parameters
        ----------
        x : domain element-like
            Point in which to evaluate the gradient
        out : domain element, optional
            Element into which the result is written

        Returns
        -------
        out : domain element
            Result of the gradient calcuation. If ``out`` was given,
            the returned object is a reference to it.
        """
        if out is None:
            return self._right.derivative(x).adjoint(
                self._left.gradient(self._right(x)))
        else:
            if self._tmp is not None:
                tmp_op_ran = self._right(x, out=self._tmp)
            else:
                tmp_op_ran = self._right(x)

            if self._tmp2 is not None:
                tmp_dom = self._left.gradient(tmp_op_ran, out=self._tmp2)
            else:
                tmp_dom = self._left.gradient(tmp_op_ran)

            self._right.derivative(x).adjoint(tmp_dom, out=out)

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
        return self._right.derivative(point).adjoint(
                self._left.derivative(self._right(point)))


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
                            'not match'.format(func1.range, func2.range))

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
        return self._op1.gradient + self._op2.gradient

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
        """Initialize a FunctionalScalarSum instance.

        Parameters
        ----------
        func : `Functional`
            Functional to which the scalar is added.
        scalar :
        tmp_dom : `LinearSpaceVector`, optional
            ...
        """

        # TODO: Update doc. What is the type of ``scalar``?
        # Also: what is tmp_dom used for?
        if not isinstance(func, Functional):
            raise TypeError('functional {!r} is no a Functional instance'
                            ''.format(func))
        if scalar not in func.range:
            raise TypeError('the scalar {} is not in the range of the'
                            'functional {!r}'.format(scalar, func))

        Functional.__init__(domain=func.domain, linear=func.is_linear,
                            smooth=func.is_smooth, concave=func.is_concave,
                            convex=func.is_convex,
                            grad_lipschitz=func.grad_lipschitz)

        OperatorSum.__init__(self, func,
                             ConstantOperator(vector=scalar,
                                              domain=func.domain,
                                              range=func.range),
                             tmp_ran=None, tmp_dom=tmp_dom)

        self.scalar = scalar

        @property
        def gradient(self):
            """Gradient operator of the FunctionalScalarSum.

            This is the same as the gradient of the original functional.
            """
            return self._op1.gradient

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
            return self.original_func.proximal(sigma)

        @property
        def conjugate_functional(self):
            """Convex conjugate functional of FunctionalScalarSum."""
            return self._op1.conjugate_functional - self.scalar

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
            return self._op1.derivative(point)


class TranslatedFunctional(Functional):
    """Implementation of the translated functional.

    Given a functional ``f`` and an element ``translation`` in the domain of
    ``f``, this corresponds to the functional ``f(. - translation)``.
    """
    # TODO: this will be linear if the "total" translation is zero...
    def __init__(self, func, translation):
        """Initialize a TranslatedFunctional instnace."""
        super().__init__(domain=func.domain, linear=False,
                         smooth=func.is_smooth,
                         concave=func.is_concave,
                         convex=func.is_convex,
                         grad_lipschitz=func.grad_lipschitz)
        self._original_func = func
        self._translation = translation

    def _call(self, x):
        """Evaluates the functional in a point ``x``.

        Parameters
        ----------
        x : `LinearSpaceVector`
            Element in the domain of the functional.

        Returns
        -------
        `self(x)` : `float`
            Evaluation of the functional, which is a constant.
        """
        return self._original_func(x - self._translation)

    @property
    def gradient(self):
        """Gradient operator of the translated.

        The operator is given by a transation of the gradient operator of the
        original functional.
        """
        class TranslatedGradientOperator(Operator):
            """The gradient operator for a translated functional."""
            def __init__(self, translated_func, translation):
                """Initialize a TranslatedGradientOperator instance."""
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
        out : `DerivativeOperator`
            The linear operator that maps ``x --> <x, grad_f(p)>``.
        """
        return self._original_func.derivative(point - self._translation)


class ConvexConjugateTranslation(Functional):
    """ The ``Functional`` representing (F( . - y))^*.

    Calculate the convex conjugate functional of the translated function
    F(x - y).

    This is calculated according to the rule

    (F( . - y))^* (x) = F^*(x) + <y, x>

    where ``y`` is the translation of the argument.

    Parameters
    ----------
    convex_conj_f : `Functional`
    Function corresponding to F^*.

    y : Element in domain of F^*.

    Notes
    -----
    For reference on the identity used, see [KP2015]_.
    """

    def __init__(self, convex_conj_f, y):
        """Initialize a ConvexConjugateTranslation instance.

        Parameters
        ----------
        convex_conj_f : `Functional`
            Function corresponding to F^*.

        y : Element in domain of F^*.
        """

        if y is not None and not isinstance(y, LinearSpaceVector):
            raise TypeError(
                'vector {!r} not None or a LinearSpaceVector instance.'
                ''.format(y))

        if y not in convex_conj_f.domain:
            raise TypeError(
                'vector {} not in the domain of the functional {}.'
                ''.format(y, convex_conj_f.domain))

        super().__init__(domain=convex_conj_f.domain,
                         linear=convex_conj_f.is_linear,
                         smooth=convex_conj_f.is_smooth,
                         concave=convex_conj_f.is_concave,
                         convex=convex_conj_f.is_convex)

        self.orig_convex_conj_f = convex_conj_f
        self.y = y

        # TODO:
        # The Lipschitz constant for the gradient can be bounded, by using
        # triangle inequality. However: is it the tightest bound?

    def _call(self, x):
        """Applies the functional to the given point.

        Returns
        -------
        `self(x)` : `element` in the `field`of the ``domain``.
            Evaluation of the functional.
        """
        return self.orig_convex_conj_f(x) + x.inner(self.y)

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
        return self.orig_convex_conj_f.gradient + ConstantOperator(self.y)

    # TODO: Add test for this proximal
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
            self.orig_convex_conj_f.proximal, a=0, u=self.y)(sigma)

    # TODO: Add this when convex conjugate of a linear perturbation has
    # been added. THIS WOULD ONLY BE VALIDE WHEN f IS PROPER, CONVEX AND
    # LSC AND THIS WOULD HAVE TO BE THE BIDUAL!
#        def conjugate_functional(self):
#            """Convex conjugate functional of the functional.
#
#            Parameters
#            ----------
#            none
#
#            Returns
#            -------
#            out : Functional
#                Domain equal to domain of functional
#            """
#            raise NotImplementedError


class ConvexConjugateFuncScaling(Functional):
    """ The ``Functional`` representing (scaling * F(.))^*.

    Calculate the convex conjugate functional of the scaled function
    sclaing * F(x).

    This is calculated according to the rule

        (scaling * F(.))^* (x) = scaling * F^*(x/scaling)

    where ``scaling`` is the scaling parameter. Note that this does not allow
    for scaling with nonpositive values.

    Parameters
    ----------
    convex_conj_f : `Functional`
        Function corresponding to F^*.

    scaling : `float`
        Positive scaling parameter.

    Notes
    -----
    For reference on the identity used, see [KP2015]_.
    """

    def __init__(self, convex_conj_f, scaling):
        """Initialize a ConvexConjugateFuncScaling instance.

        Parameters
        ----------
        convex_conj_f : `Functional`
            Function corresponding to F^*.

        scaling : 'float'
            The scaling parameter.
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

        self.orig_convex_conj_f = convex_conj_f
        self.scaling = scaling

    def _call(self, x):
        """Applies the functional to the given point.

        Returns
        -------
        `self(x)` : `float`
            Evaluation of the functional.
        """
        return self.scaling * self.orig_convex_conj_f(x * (1 / self.scaling))

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
        return self.orig_convex_conj_f.gradient * (1 / self.scaling)

    # TODO: Add test for this prox
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
                                    scaling=(1 / self.scaling)
                                    )(self.scaling * sigma)

    # TODO: Add this
    # THIS WOULD ONLY BE VALIDE WHEN f IS PROPER, CONVEX AND
    # LSC AND THIS WOULD HAVE TO BE THE BIDUAL!
#        def conjugate_functional(self):
#            """Convex conjugate functional of the functional.
#
#            Parameters
#            ----------
#            none
#
#            Returns
#            -------
#            out : Functional
#                Domain equal to domain of functional
#            """
#            raise NotImplementedError


class ConvexConjugateArgScaling(Functional):
    """ The ``Functional`` representing (F( . * scaling))^*.

    Calculate the convex conjugate of function F(x * scaling). This is
    calculated according to the rule

        (F( . * scaling))^* (x) = F^*(x/scaling)

    where ``scaling`` is the scaling parameter. Note that this does not allow
    for scaling with ``0``.

    Parameters
    ----------
    convex_conj_f : `Functional`
        Function corresponding to F^*.

    scaling : `float`
        Scaling parameter

    Notes
    -----
    For reference on the identity used, see [KP2015]_.
    """

    def __init__(self, convex_conj_f, scaling):
        """Initialize a ConvexConjugateArgScaling instance.

        Parameters
        ----------
        convex_conj_f : `Functional`
            Function corresponding to F^*.

        scaling : 'float'
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

        self.orig_convex_conj_f = convex_conj_f
        self.scaling = float(scaling)

    def _call(self, x):
        """Applies the functional to the given point.

        Returns
        -------
        `self(x)` : `float`
            Evaluation of the functional.
        """
        return self.orig_convex_conj_f(x * (1 / self.scaling))

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
        return ((1 / self.scaling) * self.orig_convex_conj_f.gradient *
                (1 / self.scaling))

    # TODO: Add test for this prox
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
                                    scaling=(1 / self.scaling))(sigma)

    # TODO: Add this
    # THIS WOULD ONLY BE VALIDE WHEN f IS PROPER, CONVEX AND
    # LSC AND THIS WOULD HAVE TO BE THE BIDUAL!
#        def conjugate_functional(self):
#            """Convex conjugate functional of the functional.
#
#            Parameters
#            ----------
#            none
#
#            Returns
#            -------
#            out : Functional
#                Domain equal to domain of functional
#            """
#            raise NotImplementedError


class ConvexConjugateLinearPerturb(Functional):
    """ The ``Functional`` representing (F(.) + <y,.>)^*.

    Calculate the convex conjugate functional perturbed function F(x) + <y,x>.

    This is calculated according to the rule

        (F(.) + <y,.>)^* (x) = F^*(x - y)

    where ``y`` is the linear perturbation.

    Parameters
    ----------
    convex_conj_f : `Functional`
        Function corresponding to F^*.

    y : Element in domain of F^*.

    Notes
    -----
    For reference on the identity used, see [KP2015]_. Note that this is only
    valide for functionals with a domain that is a Hilbert space.
    """

    def __init__(self, convex_conj_f, y):
        """Initialize a ConvexConjugateLinearPerturb instance.

        Parameters
        ----------
        convex_conj_f : `Functional`
            Function corresponding to F^*.

        y : Element in domain of F^*.
        """
        if y is not None and not isinstance(y, LinearSpaceVector):
            raise TypeError('vector {!r} not None or a LinearSpaceVector'
                            ' instance.'.format(y))

        if y not in convex_conj_f.domain:
            raise TypeError('vector {} not in the domain of the functional {}.'
                            ''.format(y, convex_conj_f.domain))

        super().__init__(domain=convex_conj_f.domain,
                         linear=convex_conj_f.is_linear,
                         smooth=convex_conj_f.is_smooth,
                         concave=convex_conj_f.is_concave,
                         convex=convex_conj_f.is_convex)

        self.orig_convex_conj_f = convex_conj_f
        self.y = y

    def _call(self, x):
        """Applies the functional to the given point.

        Returns
        -------
        `self(x)` : `float`
            Evaluation of the functional.
        """
        return self.orig_convex_conj_f(x - self.y)

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
        return (self.orig_convex_conj_f.gradient *
                ResidualOperator(IdentityOperator(self.domain), -self.y))

    # TODO: Add test for this prox
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

    # TODO: Add this
    # THIS WOULD ONLY BE VALIDE WHEN f IS PROPER, CONVEX AND
    # LSC AND THIS WOULD HAVE TO BE THE BIDUAL!
#        def conjugate_functional(self):
#            """Convex conjugate functional of the functional.
#
#            Parameters
#            ----------
#            none
#
#            Returns
#            -------
#            out : Functional
#                Domain equal to domain of functional
#            """
#            raise NotImplementedError
