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

"""Utility functions for computing convex conjugate functionals of a functional
in which
1) the arguemnt is shifted
2) the argument is scalled
3) the functional is scalled
4) a linear functional is added to it

For more details on convex conjugate functionals, see for example [Lue1969]_.
"""


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

from odl.solvers.functional import Functional
from odl import LinearSpaceVector, Operator, ResidualOperator, IdentityOperator

__all__ = ('convex_conjugate_translation', 'convex_conjugate_arg_scaling',
           'convex_conjugate_functional_scaling',
           'convex_conjugate_linear_perturbation')


def convex_conjugate_translation(convex_conj_f, y):
    """Calculate the convex conjugate functional of the translated function
    F(x - y).

    This is calculated according to the rule

        (F( . - y))^* (x) = F^*(x) + <y, x>

    where ``y`` is the translation of the argument.

    Parameters
    ----------
    convex_conj_f : `Functional`
        Function corresponding to F^*.

    y : Element in domain of F^*.

    Returns
    -------
    ConvexConjugateTranslation : `Functional`
        Functional corresponding to (F( . - y))^*

    Notes
    -----
    For reference on the identity used, see [KP2015]_.
    """

    if y is not None and not isinstance(y, LinearSpaceVector):
        raise TypeError('vector {!r} not None or a LinearSpaceVector instance.'
                        ''.format(y))

    if y not in convex_conj_f.domain:
        raise TypeError('vector {} not in the domain of the functional {}.'
                        ''.format(y, convex_conj_f.domain))

    class ConvexConjugateTranslation(Functional):
        """ The ``Functional`` representing (F( . - y))^*.
        """

        def __init__(self, convex_conj_f, y):
            """Initialize a ConvexConjugateTranslation instance.

            Parameters
            ----------
            convex_conj_f : `Functional`
                Function corresponding to F^*.

            y : Element in domain of F^*.
            """
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
            `self(x)` : `float`
                Evaluation of the functional.
            """
            return convex_conj_f(x) + x.inner(self.y)

        @property
        def gradient(self):
            """Gradient operator of the functional.

            Returns the operator that corresponds to the mapping

                x -> grad_f(x)

            where ``grad_f(x)`` is the element used to evaluated derivatives in
            a direction ``d`` by <grad_f(x), d>.

            Returns
            -------
            ConvexConjugateTranslationGradient : `Operator`
                Operator to evaluated grad_f in a point.
            """

            tmp = self

            class ConvexConjugateTranslationGradient(Operator):
                """Operator corresponding to the gradient of the convex
                conjugate of a translated function.
                """

                def __init__(self):
                    """Initialize a ConvexConjugateTranslationGradient
                    instance.
                    """
                    super().__init__(tmp.domain, tmp.domain)
                    self.original_gradient = \
                        tmp.orig_convex_conj_f.gradient

                def _call(self, x):
                    """Applies the operator to the given point.

                    Returns
                    -------
                    `self(x)` : `LinearSpaceVector`
                        Evaluation of the gradient in the point ´x´.
                    """
                    return self.original_gradient(x) + tmp.y

            return ConvexConjugateTranslationGradient()

        # TODO: Add this when the proximal is added to the functional
#        def proximal(self, sigma=1.0):
#            """Return the proximal operator of the functional.
#
#            Parameters
#            ----------
#            sigma : positive float, optional
#                Regularization parameter of the proximal operator
#
#            Returns
#            -------
#            out : Operator
#                Domain and range equal to domain of functional
#            """
#            raise NotImplementedError

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

    return ConvexConjugateTranslation(convex_conj_f, y)


def convex_conjugate_arg_scaling(convex_conj_f, scaling):
    """Calculate the convex conjugate of function F(x * scaling).

    This is calculated according to the rule

        (F( . * scaling))^* (x) = F^*(x/scaling)

    where ``scaling`` is the scaling parameter. Note that this does not allow
    for scaling with ``0``.

    Parameters
    ----------
    convex_conj_f : `Functional`
        Function corresponding to F^*.

    scaling : `float`
        Scaling parameter

    Returns
    -------
    ConvexConjugateArgScaling : `Functional`
        Functional corresponding to (F( . * scaling))^*

    Notes
    -----
    For reference on the identity used, see [KP2015]_.
    """

    scaling = float(scaling)
    if scaling == 0:
        raise ValueError('Scaling with 0 is not allowed. Current value: {}.'
                         ''.format(scaling))

    class ConvexConjugateArgScaling(Functional):
        """ The ``Functional`` representing (F( . * scaling))^*.
        """

        def __init__(self, convex_conj_f, scaling):
            super().__init__(domain=convex_conj_f.domain,
                             linear=convex_conj_f.is_linear,
                             smooth=convex_conj_f.is_smooth,
                             concave=convex_conj_f.is_concave,
                             convex=convex_conj_f.is_convex)

            self.orig_convex_conj_f = convex_conj_f
            self.scaling = scaling

        def _call(self, x):
            return convex_conj_f(x * (1/self.scaling))

        @property
        def gradient(self):
            return ((1/self.scaling) * self.orig_convex_conj_f.gradient *
                    (1/self.scaling))

        # TODO: Add when the proximal frame-work is added to the functional
#        def proximal(self, sigma=1.0):
#            """Return the proximal operator of the functional.
#
#            Parameters
#            ----------
#            sigma : positive float, optional
#                Regularization parameter of the proximal operator
#
#            Returns
#            -------
#            out : Operator
#                Domain and range equal to domain of functional
#            """
#            raise NotImplementedError

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

    return ConvexConjugateArgScaling(convex_conj_f, scaling)


def convex_conjugate_functional_scaling(convex_conj_f, scaling):
    """Calculate the convex conjugate functional of the scaled function
    sclaing * F(x).

    This is calculated according to the rule

        (scaling * F(.))^* (x) = scaling * F^*(x/scaling)

    where ``scaling`` is the scaling parameter. Note that this does not allow
    for scaling with ``0``.

    Parameters
    ----------
    convex_conj_f : `Functional`
        Function corresponding to F^*.

    scaling : `float`
        Scaling parameter

    Returns
    -------
    ConvexConjugateFuncScaling : `Functional`
        Functional corresponding to (scaling * F(.))^*

    Notes
    -----
    For reference on the identity used, see [KP2015]_.
    """

    # TODO: scaling with zero gives the zero-functional. Should this be
    # allowed?

    scaling = float(scaling)
    if scaling == 0:
        raise ValueError('Scaling with 0 is not allowed. Current value: {}.'
                         ''.format(scaling))

    class ConvexConjugateFuncScaling(Functional):
        """ The ``Functional`` representing (scaling * F(.))^*.
        """

        def __init__(self, convex_conj_f, scaling):
            super().__init__(domain=convex_conj_f.domain,
                             linear=convex_conj_f.is_linear,
                             smooth=convex_conj_f.is_smooth,
                             concave=convex_conj_f.is_concave,
                             convex=convex_conj_f.is_convex)

            self.orig_convex_conj_f = convex_conj_f
            self.scaling = scaling

        def _call(self, x):
            return self.scaling * convex_conj_f(x * (1/self.scaling))

        @property
        def gradient(self):
            return self.orig_convex_conj_f.gradient * (1/self.scaling)

        # TODO: Add when the proximal frame-work is added to the functional
#        def proximal(self, sigma=1.0):
#            """Return the proximal operator of the functional.
#
#            Parameters
#            ----------
#            sigma : positive float, optional
#                Regularization parameter of the proximal operator
#
#            Returns
#            -------
#            out : Operator
#                Domain and range equal to domain of functional
#            """
#            raise NotImplementedError

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

    return ConvexConjugateFuncScaling(convex_conj_f, scaling)


def convex_conjugate_linear_perturbation(convex_conj_f, y):
    """Calculate the convex conjugate functional perturbed function F(x) +
    <y,x>.

    This is calculated according to the rule

        (F(.) + <y,.>)^* (x) = F^*(x - y)

    where ``y`` is the linear perturbation.

    Parameters
    ----------
    convex_conj_f : `Functional`
        Function corresponding to F^*.

    y : Element in domain of F^*.

    Returns
    -------
    ConvexConjugateLinearPerturb : `Functional`
        Functional corresponding to (F(.) + <y,.>)^*

    Notes
    -----
    For reference on the identity used, see [KP2015]_. Note that this is only
    valide for functionals with a domain that is a Hilbert space.
    """

    if y is not None and not isinstance(y, LinearSpaceVector):
        raise TypeError('vector {!r} not None or a LinearSpaceVector instance.'
                        ''.format(y))

    if y not in convex_conj_f.domain:
        raise TypeError('vector {} not in the domain of the functional {}.'
                        ''.format(y, convex_conj_f.domain))

    class ConvexConjugateLinearPerturb(Functional):
        """ The ``Functional`` representing (F(.) + <y,.>)^*.
        """

        def __init__(self, convex_conj_f, y):
            super().__init__(domain=convex_conj_f.domain,
                             linear=convex_conj_f.is_linear,
                             smooth=convex_conj_f.is_smooth,
                             concave=convex_conj_f.is_concave,
                             convex=convex_conj_f.is_convex)

            self.orig_convex_conj_f = convex_conj_f
            self.y = y

        def _call(self, x):
            return convex_conj_f(x - self.y)

        @property
        def gradient(self):
            return (self.orig_convex_conj_f.gradient *
                    ResidualOperator(IdentityOperator(self.domain), -self.y))

        # TODO: Add when the proximal frame-work is added to the functional
#        def proximal(self, sigma=1.0):
#            """Return the proximal operator of the functional.
#
#            Parameters
#            ----------
#            sigma : positive float, optional
#                Regularization parameter of the proximal operator
#
#            Returns
#            -------
#            out : Operator
#                Domain and range equal to domain of functional
#            """
#            raise NotImplementedError

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

    return ConvexConjugateLinearPerturb(convex_conj_f, y)
