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
                                  proximal_quadratic_perturbation,
                                  proximal_zero)


__all__ = ('Functional', 'FunctionalLeftScalarMult',
           'FunctionalRightScalarMult', 'FunctionalComp',
           'FunctionalRightVectorMult', 'FunctionalSum', 'FunctionalScalarSum',
           'TranslatedFunctional', 'ConvexConjugateTranslation',
           'ConvexConjugateFuncScaling', 'ConvexConjugateArgScaling',
           'ConvexConjugateLinearPerturb')


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
    this case. For more information, see
    http://odl.readthedocs.io/guide/in_depth/functional_guide.html.
    """

    def __init__(self, space, linear=False, grad_lipschitz=np.inf):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            The domain of this functional, i.e., the set of elements to
            which this functional can be applied.
        linear : bool, optional
            If `True`, the functional is considered as linear.
        grad_lipschitz : float, optional
            The Lipschitz constant of the gradient.
        """
        Operator.__init__(self, domain=space,
                          range=space.field, linear=linear)
        self.__grad_lipschitz = float(grad_lipschitz)

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
        """Return the proximal factory of the functional.

        Notes
        -----
        The nonsmooth solvers that make use of proximal operators to solve a
        given optimization problem take a `proximal factory` as input,
        i.e., a function returning a proximal operator. Note that
        ``Functional.proximal`` is in fact a `proximal factory`. See for
        example `forward_backward_pd`.
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
        raise NotImplementedError

    def derivative(self, point):
        """Return the derivative operator in the given point.

        This function returns the linear operator given by

            ``self.derivative(point)(x) == self.gradient(point).inner(x)``

        Parameters
        ----------
        point : `LinearSpaceVector`
            The point in which the gradient is evaluated.

        Returns
        -------
        out : `Operator`
        """
        return self.gradient(point).T

    def translated(self, shift):
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
            raise TypeError('`shift` {} is not in the domain of the '
                            'functional {!r}'.format(shift, self))

        return TranslatedFunctional(self, shift)

    def __mul__(self, other):
        """Return ``self * other``.

        If ``other`` is an `Operator`, this corresponds to composition with the
        operator:

            ``func * op == (x --> func(op(x)))``

        If ``other`` is a scalar, this corresponds to right multiplication of
        scalars with functionals:

            ``func * scalar == (x --> func(scalar * x))``

        If ``other`` is a vector, this corresponds to right multiplication of
        vectors with functionals:

            ``func * vector == (x --> func(vector * x))``

        Note that left and right multiplications are generally different.

        Parameters
        ----------
        other : `Operator`, `LinearSpaceVector` or scalar
            `Operator`:
            The `Operator.range` of ``other`` must match this functional's
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
            Multiplication result.

            If ``other`` is an `Operator`, ``mul`` is a
            `FunctionalComp`.

            If ``other`` is a scalar, ``mul`` is a
            `FunctionalRightScalarMult`.

            If ``other`` is a vector, ``mul`` is a
            `FunctionalRightVectorMult`.

        See Also
        --------
        Operator.__mul__ : implementation of __mul__ for Operators.
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
        elif isinstance(other, LinearSpaceVector):
            return FunctionalRightVectorMult(self, other)
        else:
            return super().__mul__(other)

    def __rmul__(self, other):
        """Return ``other * self``.

        If ``other`` is an `Operator`, since a functional is also an operator
        this corresponds to operator composition:

            ``op * func == (x --> op(func(x))``

        If ``other`` is a scalar, this corresponds to left multiplication of
        scalars with functionals:

            ``scalar * func == (x --> scalar * func(x))``

        If ``other`` is a vector,  since a functional is also an operator this
        corresponds to left multiplication of vectors with operators:

            ``vector * func == (x --> vector * func(x))``

        Note that left and right multiplications are generally
        different.

        Parameters
        ----------
        other : `Operator`, `LinearSpaceVector` or scalar
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
            Multiplication result.

            If ``other`` is an `Operator`, ``rmul`` is an `OperatorComp`.

            If ``other`` is a scalar, ``rmul`` is a
            `FunctionalLeftScalarMult`.

            If ``other`` is a vector, ``rmul`` is a
            `OperatorLeftVectorMult`.
        """
        if other in self.domain.field:
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

            ``func1 + func2 == (x --> func1(x) + func2(x))``

        If ``other`` is a scalar, this corresponds to adding a scalar to the
        value of the functional:

            ``func + scalar == (x --> func(x) + scalar)``

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
            Addition result.

            If ``other`` is in ``Functional.range``, ``add`` is a
            `FunctionalScalarSum`.

            If ``other`` is a `Functional`, ``add`` is a `FunctionalSum`.

        See Also
        --------
        Operator.__add__ : implementation of __add__ for Operators.
        """
        if other in self.domain.field:
            return FunctionalScalarSum(self, other)
        elif isinstance(other, Functional):
            return FunctionalSum(self, other)
        else:
            return NotImplemented

    # Since addition is commutative, right and left addition is the same
    __radd__ = __add__

    def __sub__(self, other):
        """Return ``self - other``."""
        return self.__add__(-1 * other)

    @property
    def grad_lipschitz(self):
        """Lipschitz constant for the gradient of the functional"""
        return self.__grad_lipschitz


class FunctionalLeftScalarMult(Functional, OperatorLeftScalarMult):

    """Scalar multiplication of functional from the left.

    Given a functional ``f`` and a scalar ``scalar``, this represents the
    functional

        ``(scalar * f)(x) == scalar * f(x)``.
    """

    def __init__(self, func, scalar):
        """Initialize a new instance.

        Parameters
        ----------
        func : `Functional`
            Functional to be scaled.
        scalar : `float`
            Number with which to scale the functional.
        """
        if not isinstance(func, Functional):
            raise TypeError('`func` {!r} is not a `Functional` instance.'
                            ''.format(func))

        scalar = func.range.element(scalar)

        if scalar == 0:
            Functional.__init__(self, space=func.domain, linear=True,
                                grad_lipschitz=0)

        else:
            Functional.__init__(self, space=func.domain,
                                linear=func.is_linear,
                                grad_lipschitz=(
                                    np.abs(scalar) * func.grad_lipschitz))

        OperatorLeftScalarMult.__init__(self, operator=func, scalar=scalar)

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
        """Convex conjugate functional of the scaled functional."""
        # The helper function only allows positive scaling parameters.
        # Otherwise it gives an error.
        return ConvexConjugateFuncScaling(
            self.functional.convex_conj, self.scalar)

    @property
    def proximal(self):
        """Return the proximal factory of the scaled functional.

        See Also
        --------
        proximal_zero
        """

        if self.scalar < 0:
            raise ValueError('Proximal operator of functional scaled with a '
                             'negative value {} is not well-defined'
                             ''.format(self.scalar))

        elif self.scalar == 0:
            return proximal_zero(self.domain)

        else:
            return lambda sigma: self.functional.proximal(sigma * self.scalar)


class FunctionalRightScalarMult(Functional, OperatorRightScalarMult):

    """Scalar multiplication of the argument of functional.

    Given a functional ``f`` and a scalar ``scalar``, this represents the
    functional

        ``(f * scalar)(x) == f(scalar * x)``.
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
            raise TypeError('`func` {!r} is not a `Functional` instance.'
                            ''.format(func))

        scalar = func.range.element(scalar)

        if scalar == 0:
            Functional.__init__(self, space=func.domain, linear=True,
                                grad_lipschitz=0)
        else:
            Functional.__init__(self, space=func.domain,
                                linear=func.is_linear,
                                grad_lipschitz=(np.abs(scalar) *
                                                func.grad_lipschitz))

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
        """Convex conjugate functional of functional with scaled argument."""
        return ConvexConjugateArgScaling(self.functional.convex_conj,
                                         self.scalar)

    @property
    def proximal(self, sigma=1.0):
        """Return the proximal factory of the functional.

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
        tmp1 : `element` of the range of ``op``, optional
            Used to avoid the creation of a temporary when applying ``op``
        tmp2 : `element` of the range of ``op``, optional
            Used to avoid the creation of a temporary when applying the
            gradient of ``func``
        """
        if not isinstance(func, Functional):
            raise TypeError('`fun` {!r} is not a `Functional` instance.'
                            ''.format(func))

        OperatorComp.__init__(self, left=func, right=op)

        Functional.__init__(self, space=func.domain,
                            linear=(func.is_linear and op.is_linear),
                            grad_lipschitz=np.infty)

    @property
    def gradient(self):
        """Gradient of the compositon according to the chain rule."""
        func = self.left
        op = self.right

        class FunctionalCompositionGradient(Operator):

            """Gradient of the compositon according to the chain rule."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(func.domain, func.domain, linear=False)

            def _call(self, x):
                """Apply the gradient operator to the given point."""
                return op.derivative(x).adjoint(func.gradient(op(x)))

        return FunctionalCompositionGradient()


class FunctionalRightVectorMult(Functional, OperatorRightVectorMult):

    """Expression type for the functional right vector multiplication.

    Given a functional ``func`` and a vector ``y`` in the domain of ``func``,
    this corresponds to the functional

        ``(func * y)(x) == func(y*x)``.
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
            raise TypeError('`fun` {!r} is not a `Functional` instance.'
                            ''.format(func))

        OperatorRightVectorMult.__init__(self, operator=func, vector=vector)

        Functional.__init__(self, space=func.domain)

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        return self.vector * self.operator.gradient * self.vector

    @property
    def convex_conj(self):
        """Convex conjugate functional of the functional."""
        return self.operator.convex_conj * (1.0 / self.vector)


class FunctionalSum(Functional, OperatorSum):

    """Expression type for the sum of functionals.

    ``FunctionalSum(func1, func2) == (x --> func1(x) + func2(x))``.
    """

    def __init__(self, func1, func2):
        """Initialize a new instance.

        Parameters
        ----------
        func1, func2 : `Functional`
            The summands of the functional sum. Their `Operator.domain`
            and `Operator.range` must coincide.
        """
        if not isinstance(func1, Functional):
            raise TypeError('`func1` {!r} is not a `Functional` instance.'
                            ''.format(func1))
        if not isinstance(func2, Functional):
            raise TypeError('`func2` {!r} is not a `Functional` instance.'
                            ''.format(func2))

        OperatorSum.__init__(self, func1, func2)

        Functional.__init__(self, space=func1.domain,
                            linear=(func1.is_linear and func2.is_linear),
                            grad_lipschitz=(func1.grad_lipschitz +
                                            func2.grad_lipschitz))

    @property
    def left_func(self):
        """The left functional in the sum."""
        return self.left

    @property
    def right_func(self):
        """The right functional in the sum."""
        return self.right

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
            raise TypeError('`fun` {!r} is no a `Functional` instance'
                            ''.format(func))
        if scalar not in func.range:
            raise TypeError('`scalar` {} is not in the range of '
                            '`func` {!r}'.format(scalar, func))

        FunctionalSum.__init__(self, func1=func,
                               func2=ConstantFunctional(space=func.domain,
                                                        constant=scalar))

    @property
    def scalar(self):
        """The scalar that is added to the functional"""
        return self.right_func.constant

    @property
    def proximal(self):
        """Proximal factory of the FunctionalScalarSum."""
        return self.left_func.proximal

    @property
    def convex_conj(self):
        """Convex conjugate functional of FunctionalScalarSum."""
        return self.left.convex_conj - self.scalar


class TranslatedFunctional(Functional):

    """Implementation of the translated functional.

    Given a functional ``f`` and an element ``translation`` in the domain of
    ``f``, this corresponds to the functional ``f(. - translation)``.
    """

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

        super().__init__(space=func.domain, linear=False,
                         grad_lipschitz=func.grad_lipschitz)

        # TODO: Add case if we have translation -> scaling -> translation?
        if isinstance(func, TranslatedFunctional):
            self.__original_func = func.original_func
            self.__translation = func.translation + translation

        else:
            self.__original_func = func
            self.__translation = translation

    @property
    def original_func(self):
        """The original functional that has been translated."""
        return self.__original_func

    @property
    def translation(self):
        """The translation."""
        return self.__translation

    def _call(self, x):
        """Evaluate the functional in a point ``x``."""
        return self.original_func(x - self.translation)

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        return (self.original_func.gradient *
                (IdentityOperator(self.domain) - self.translation))

    @property
    def proximal(self):
        """Return the proximal factory of the translated functional.

        See Also
        --------
        proximal_translation
        """
        return proximal_translation(self.original_func.proximal,
                                    self.translation)

    @property
    def convex_conj(self):
        """Convex conjugate functional of the translated functional."""
        return ConvexConjugateTranslation(
            self.original_func.convex_conj,
            self.translation)


class ConvexConjugateTranslation(Functional):

    """ The ``Functional`` representing ``(F( . - translation))^*``.

    This is a functional representing the conjugate functional of the
    translated function ``F(. - translation)``. It is calculated according to
    the rule

        ``(F( . - translation))^* (x) == F^*(x) + <translation, x>``,

    where ``translation`` is the translation of the argument.

    Notes
    -----
    The implementation assumes that the underlying  functional ``F`` is proper,
    convex, and lower semi-continuous. Otherwise the convex conjugate of the
    convex conjugate is not necessarily the functional itself.

    For reference on the identity used, see [KP2015]_.
    """

    def __init__(self, cconj_f, translation):
        """Initialize a new instance.

        Parameters
        ----------
        cconj_f : `Functional`
            Function corresponding to F^*.

        translation : `LinearSpaceVector`
            Element in domain of ``F^*``.
        """
        if not isinstance(cconj_f, Functional):
            raise TypeError('`cconj_f` {} is not a `Functional` instance'
                            ''.format(cconj_f))

        if translation not in cconj_f.domain:
            raise TypeError(
                '`translation` {} not in the domain of `cconj_f` {!r}.'
                ''.format(translation, cconj_f.domain))

        super().__init__(space=cconj_f.domain,
                         linear=cconj_f.is_linear)

        # Only compute the grad_lipschitz if it is not inf
        if not cconj_f.grad_lipschitz == np.inf:
            self.__grad_lipschitz = (cconj_f.grad_lipschitz +
                                     translation.norm())

        self.__orig_cconj_f = cconj_f
        self.__translation = translation

    @property
    def orig_cconj_f(self):
        """The original convex conjugate functional."""
        return self.__orig_cconj_f

    @property
    def translation(self):
        """The translation."""
        return self.__translation

    def _call(self, x):
        """Apply the functional to the given point."""
        return self.orig_cconj_f(x) + x.inner(self.translation)

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        return self.orig_cconj_f.gradient + ConstantOperator(self.translation)

    @property
    def proximal(self):
        """Return the proximal factory of the ConvexConjugateTranslation.

        See Also
        --------
        proximal_quadratic_perturbation
        """
        return proximal_quadratic_perturbation(
            self.orig_cconj_f.proximal, a=0, u=self.translation)

    @property
    def convex_conj(self):
        """Convex conjugate functional of the functional.

        By the Fenchel-Moreau theorem this a translation of the original
        convex conjugate functional.
        """
        return self.orig_cconj_f.convex_conj.traslated(
            self.translation)


class ConvexConjugateFuncScaling(Functional):

    """ The ``Functional`` representing ``(scaling * F(.))^*``.

    This is a functional representing the conjugate functional of the scaled
    function ``scaling * F(.)``. This is calculated according to the rule

        ``(scaling * F(.))^* (x) == scaling * F^*(x/scaling)``,

    where ``scaling`` is the scaling parameter. Note that scaling is only
    allowed with strictly positive scaling parameters.

    Notes
    -----
    The implementation assumes that the underlying  functional ``F`` is proper,
    convex, and lower semi-continuous. Otherwise the convex conjugate of the
    convex conjugate is not necessarily the functional itself.

    For reference on the identity used, see [KP2015]_.
    """

    def __init__(self, cconj_f, scaling):
        """Initialize a new instance.

        Parameters
        ----------
        cconj_f : `Functional`
            Functional corresponding to ``F^*``.
        scaling : `float`, positive
            Positive scaling parameter.
        """
        if not isinstance(cconj_f, Functional):
            raise TypeError('`cconj_f` {} is not a `Functional` instance'
                            ''.format(cconj_f))

        scaling = float(scaling)

        # The case scaling = 0 is handeled in Functional.__rmul__
        if scaling <= 0:
            raise ValueError(
                'Scaling with nonpositive values is not allowed. Current '
                'value: {}.'.format(scaling))

        super().__init__(space=cconj_f.domain, linear=cconj_f.is_linear,
                         grad_lipschitz=(np.abs(scaling) *
                                         cconj_f.grad_lipschitz))

        self.__orig_cconj_f = cconj_f
        self.__scaling = scaling

    @property
    def orig_cconj_f(self):
        """The original convex conjugate functional."""
        return self.__orig_cconj_f

    @property
    def scaling(self):
        """The scaling."""
        return self.__scaling

    def _call(self, x):
        """Apply the functional to the given point."""
        return self.scaling * self.orig_cconj_f(x * (1.0 / self.scaling))

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        return self.orig_cconj_f.gradient * (1.0 / self.scaling)

    @property
    def proximal(self):
        """Return the proximal factory of the ConvexConjugateFuncScaling.

        See Also
        --------
        proximal_arg_scaling
        """
        return lambda sigma: proximal_arg_scaling(self.orig_cconj_f.proximal,
                                                  scaling=(1.0 / self.scaling)
                                                  )(self.scaling * sigma)

    @property
    def convex_conj(self):
        """Convex conjugate functional of the functional.

        By the Fenchel-Moreau theorem this a scaling of the original
        convex conjugate functional.
        """
        return self.scaling * self.orig_cconj_f.convex_conj


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
    The implementation assumes that the underlying  functional ``F`` is proper,
    convex, and lower semi-continuous. Otherwise the convex conjugate of the
    convex conjugate is not necessarily the functional itself.

    For reference on the identity used, see [KP2015]_.
    """

    def __init__(self, cconj_f, scaling):
        """Initialize a new instance.

        Parameters
        ----------
        convex_conj_f : `Functional`
            Functional corresponding to ``F^*``.

        scaling : 'float', nonzero
            The scaling parameter.
        """
        if not isinstance(cconj_f, Functional):
            raise TypeError('`cconj_f` {} is not a `Functional` instance'
                            ''.format(cconj_f))

        scaling = float(scaling)
        if scaling == 0:
            raise ValueError('Scaling with 0 is not allowed. Current value:'
                             ' {}.'.format(scaling))

        super().__init__(space=cconj_f.domain,
                         linear=cconj_f.is_linear)

        self.__orig_cconj_f = cconj_f
        self.__scaling = scaling

    @property
    def orig_cconj_f(self):
        """The original convex conjugate functional."""
        return self.__orig_cconj_f

    @property
    def scaling(self):
        """The scaling."""
        return self.__scaling

    def _call(self, x):
        """Apply the functional to the given point."""
        return self.orig_cconj_f(x * (1.0 / self.scaling))

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        return ((1.0 / self.scaling) * self.orig_cconj_f.gradient *
                (1.0 / self.scaling))

    @property
    def proximal(self):
        """Return the proximal factory of the ConvexConjugateArgScaling.

        See Also
        --------
        proximal_arg_scaling
        """
        return proximal_arg_scaling(self.orig_cconj_f.proximal,
                                    scaling=(1.0 / self.scaling))

    @property
    def convex_conj(self):
        """Convex conjugate functional of the functional.

        By the Fenchel-Moreau theorem this a scaling of the argument of the
        original convex conjugate functional.
        """
        return self.orig_cconj_f.convex_conj * self.scaling


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
    The implementation assumes that the underlying  functional ``F`` is proper,
    convex, and lower semi-continuous. Otherwise the convex conjugate of the
    convex conjugate is not necessarily the functional itself.

    For reference on the identity used, see [KP2015]_. Note that this is only
    valide for functionals with a domain that is a Hilbert space.
    """

    def __init__(self, cconj_f, y):
        """Initialize a new instance.

        Parameters
        ----------
        convex_conj_f : `Functional`
            Functional corresponding to ``F^*``.

        y : `LinearSpaceVector`
            Element in domain of ``F^*``.
        """
        if not isinstance(cconj_f, Functional):
            raise TypeError('`cconj_f` {} is not a `Functional` instance'
                            ''.format(cconj_f))

        if not isinstance(y, LinearSpaceVector):
            raise TypeError('`y` {!r} not a `LinearSpaceVector` instance.'
                            ''.format(y))

        if y not in cconj_f.domain:
            raise TypeError('`y` {!r} not in the domain of `cconj_f` '
                            '{!r}.'.format(y, cconj_f.domain))

        super().__init__(space=cconj_f.domain,
                         linear=cconj_f.is_linear)

        self.__orig_cconj_f = cconj_f
        self.__y = y

    @property
    def orig_cconj_f(self):
        """The original convex conjugate functional."""
        return self.__orig_cconj_f

    @property
    def y(self):
        """The linear perturbation."""
        return self.__y

    def _call(self, x):
        """Apply the functional to the given point."""
        return self.orig_cconj_f(x - self.y)

    @property
    def gradient(self):
        """Gradient operator of the functional."""
        return (self.orig_cconj_f.gradient *
                ResidualOperator(IdentityOperator(self.domain), self.y))

    @property
    def proximal(self):
        """Return the proximal factory of the ConvexConjugateLinearPerturb.

        See Also
        --------
        proximal_translation
        """
        return proximal_translation(self.orig_cconj_f.proximal, self.y)

    @property
    def convex_conj(self):
        """Convex conjugate functional of the functional.

        By the Fenchel-Moreau theorem this a linear perturbation of the
        original convex conjugate functional.
        """
        return self.orig_cconj_f.convex_conj + self.y.T
