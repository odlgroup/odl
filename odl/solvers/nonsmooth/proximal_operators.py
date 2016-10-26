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
# along with ODL. If not, see <http://www.gnu.org/licenses/>.

"""Factory functions for creating proximal operators.

Functions with ``cconj`` mean the proximal of the convex conjugate and are
provided for convenience.

For more details see :ref:`proximal_operators` and references therein. For
more details on proximal operators including how to evaluate the proximal
operator of a variety of functions see [PB2014]_. """

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np
import scipy.special

from odl.operator import (Operator, IdentityOperator, ScalingOperator,
                          ConstantOperator, DiagonalOperator)
from odl.space import ProductSpace
from odl.set import LinearSpaceElement


__all__ = ('combine_proximals', 'proximal_cconj', 'proximal_translation',
           'proximal_arg_scaling', 'proximal_quadratic_perturbation',
           'proximal_composition', 'proximal_const_func',
           'proximal_box_constraint', 'proximal_nonnegativity',
           'proximal_l1', 'proximal_cconj_l1',
           'proximal_l2', 'proximal_cconj_l2',
           'proximal_l2_squared', 'proximal_cconj_l2_squared',
           'proximal_cconj_kl', 'proximal_cconj_kl_cross_entropy')


def combine_proximals(*factory_list):
    """Combine proximal operators into a diagonal product space operator.

    This assumes the functional to be separable across variables in order to
    make use of the separable sum property of proximal operators.

    Parameters
    ----------
    factory_list : sequence of callables
        Proximal operator factories to be combined.

    Returns
    -------
    diag_op : function
        Returns a diagonal product space operator factory to be initialized
        with the same step size parameter

    Notes
    -----
    That two functionals :math:`F` and :math:`G` are separable across variables
    means that :math:`F((x, y)) = F(x)` and :math:`G((x, y)) = G(y)`, and in
    this case the proximal operator of the sum is given by

    .. math::
        \mathrm{prox}_{\\sigma (F(x) + G(y))}(x, y) =
        (\mathrm{prox}_{\\sigma F}(x), \mathrm{prox}_{\\sigma G}(y)).
    """

    def diag_op_factory(sigma):
        """Diagonal matrix of operators.

        Parameters
        ----------
        sigma : positive float
            Step size parameter

        Returns
        -------
        diag_op : `DiagonalOperator`
        """
        return DiagonalOperator(
            *[factory(sigma) for factory in factory_list])

    return diag_op_factory


def proximal_cconj(prox_factory):
    """Calculate the proximal of the dual using Moreau decomposition.

    Parameters
    ----------
    prox_factory : callable
        A factory function that, when called with a step size, returns the
        proximal operator of ``F``

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized

    Notes
    -----
    The Moreau identity states that for any convex function :math:`F` with
    convex conjugate :math:`F^*`, the proximals satisfy

    .. math::

        \mathrm{prox}_{\\sigma F^*}(x) +\\sigma \,
        \mathrm{prox}_{F / \\sigma}(x / \\sigma) = x

    where :math:`\\sigma` is a scalar step size. Using this, the proximal of
    the convex conjugate is given by

    .. math::

        \mathrm{prox}_{\\sigma F^*}(x) =
        x - \\sigma \, \mathrm{prox}_{F / \\sigma}(x / \\sigma)

    Note that since :math:`(F^*)^* = F`, this can be used to get the proximal
    of the original function from the proximal of the convex conjugate.

    For reference on the Moreau identity, see [CP2011c]_.
    """

    def cconj_prox_factory(sigma):
        """Create proximal for the dual with a given sigma.

        Parameters
        ----------
        sigma : positive float
            Step size parameter

        Returns
        -------
        proximal : `Operator`
            The proximal operator of ``s * F^*`` where ``s`` is the step size
        """
        sigma = float(sigma)
        prox_other = (sigma * prox_factory(1.0 / sigma) * (1.0 / sigma))
        return IdentityOperator(prox_other.domain) - prox_other

    return cconj_prox_factory


def proximal_translation(prox_factory, y):
    """Calculate the proximal of the translated function F(x - y).

    Parameters
    ----------
    prox_factory : callable
        A factory function that, when called with a step size, returns the
        proximal operator of ``F``.
    y : Element in domain of ``F``.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized

    Notes
    -----
    Given a functional :math:`F`, this is calculated according to the rule

    .. math::

        \mathrm{prox}_{\\sigma  F( \cdot - y)}(x) =
        y + \mathrm{prox}_{\\sigma F}(x - y)

    where :math:`y` is the translation, and :math:`\\sigma` is the step size.

    For reference on the identity used, see [CP2011c]_.
    """

    def translation_prox_factory(sigma):
        """Create proximal for the translation with a given sigma.

        Parameters
        ----------
        sigma : positive float
            Step size parameter

        Returns
        -------
        proximal : `Operator`
            The proximal operator of ``s * F( . - y)`` where ``s`` is the
            step size
        """

        return (ConstantOperator(y) + prox_factory(sigma) *
                (IdentityOperator(y.space) - ConstantOperator(y)))

    return translation_prox_factory


def proximal_arg_scaling(prox_factory, scaling):
    """Calculate the proximal of function F(x * scaling).

    Parameters
    ----------
    prox_factory : callable
        A factory function that, when called with a step size, returns the
        proximal operator of ``F``
    scaling : float
        Scaling parameter

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized

    Notes
    -----
    Given a functional :math:`F`, this is calculated according to the rule

    .. math::

        \mathrm{prox}_{\\sigma F( \cdot scal)}(x) =
        \\frac{1}{scal} \mathrm{prox}_{\\sigma \, scal^2 \, F }(x \, scal)

    where :math:`scal` is the scaling parameter, and :math:`\\sigma` is the
    step size.

    For reference on the identity used, see [CP2011c]_.
    """

    scaling = float(scaling)
    if scaling == 0:
        return proximal_const_func(prox_factory(1.0).domain)

    def arg_scaling_prox_factory(sigma):
        """Create proximal for the translation with a given sigma.

        Parameters
        ----------
        sigma : positive float
            Step size parameter

        Returns
        -------
        proximal : `Operator`
            The proximal operator of ``sigma * F( . * a)`` where ``sigma`` is
            the step size
        """
        prox = prox_factory(sigma * scaling ** 2)
        return (1 / scaling) * prox * scaling

    return arg_scaling_prox_factory


def proximal_quadratic_perturbation(prox_factory, a, u=None):
    """Calculate the proximal of function F(x) + a * \|x\|^2 + <u,x>.

    Parameters
    ----------
    prox_factory : callable
        A factory function that, when called with a step size, returns the
        proximal operator of ``F``
    a : non-negative float
        Scaling of the quadratic term
    u : Element in domain of F, optional
        Defines the linear functional. For ``None``, the zero element
        is taken.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized

    Notes
    -----
    Given a functional :math:`F`, this is calculated according to the rule

    .. math::

        \mathrm{prox}_{\\sigma \left(F( \cdot ) + a \| \cdot \|^2 +
        <u, \cdot >\\right)}(x) =
        c \; \mathrm{prox}_{\\sigma F( \cdot \, c)}((x - \\sigma u) c)

    where :math:`c` is the constant

    .. math::

        c = \\frac{1}{\\sqrt{2 \\sigma a + 1}},

    :math:`a` is the scaling parameter belonging to the quadratic term,
    :math:`u` is the space element defining the linear functional, and
    :math:`\\sigma` is the step size.

    For reference on the identity used, see [CP2011c]_. Note that this identity
    is not the exact one given in the reference, but was recalculated for
    arbitrary step lengths.
    """

    a = float(a)
    if a < 0:
        raise ValueError('scaling parameter muts be non-negative, got {}'
                         ''.format(a))

    if u is not None and not isinstance(u, LinearSpaceElement):
        raise TypeError('`u` must be `None` or a `LinearSpaceElement` '
                        'instance, got {!r}.'.format(u))

    def quadratic_perturbation_prox_factory(sigma):
        """Create proximal for the quadratic perturbation with a given sigma.

        Parameters
        ----------
        sigma : positive float
            Step size parameter

        Returns
        -------
        proximal : `Operator`
            The proximal operator of ``sigma * (F(x) + a * \|x\|^2 + <u,x>)``,
            where ``sigma`` is the step size
        """
        const = 1.0 / np.sqrt(sigma * 2.0 * a + 1)
        prox = proximal_arg_scaling(prox_factory, const)(sigma)
        if u is not None:
            return (const * prox *
                    (ScalingOperator(u.space, const) - sigma * const * u))
        else:
            return const * prox * const

    return quadratic_perturbation_prox_factory


def proximal_composition(proximal, operator, mu):
    """Proximal operator factory of functional composed with unitary operator.

    For a functional ``F`` and a linear unitary `Operator` ``L`` this is the
    factory for the proximal operator of ``F * L``.

    Parameters
    ----------
    prox_factory : callable
        A factory function that, when called with a step size returns the
        proximal operator of ``F``
    operator : `Operator`
        The operator to compose the functional with
    mu : ``operator.field`` element
        Scalar such that ``(operator.adjoint * operator)(x) = mu * x``

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized

    Notes
    -----
    Given a linear operator :math:`L` with the property that for a scalar
    :math:`\\mu`

    .. math::

        L^*(L(x)) = \\mu * x

    and a convex function :math:`F`, the following identity holds

    .. math::

        \mathrm{prox}_{\\sigma F \circ L}(x) = x + \\frac{1}{\\mu}
        L^* \left( \mathrm{prox}_{\\mu \\sigma F}(Lx) - Lx \\right)

    This factory function implements this functionality.

    There is no simple formula for more general operators.

    The function cannot verify that the operator is unitary, the user needs
    to verify this.

    For reference on the identity used, see [CP2011c]_.
    """

    def proximal_composition_factory(sigma):
        """Create proximal for the dual with a given sigma

        Parameters
        ----------
        sigma : positive float
            Step size parameter

        Returns
        -------
        proximal : `Operator`
            The proximal operator of ``prox[sigma * F * L](x)``
        """
        Id = IdentityOperator(operator.domain)
        Ir = IdentityOperator(operator.range)
        prox_muf = proximal(mu * sigma)
        return (Id +
                (1.0 / mu) * operator.adjoint * ((prox_muf - Ir) * operator))

    return proximal_composition_factory


def proximal_const_func(space):
    """Proximal operator factory of the constant functional.

    Function to initialize the proximal operator of the constant functional
    defined on ``space``.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of the functional G=constant

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized

    Notes
    -----
    The constant functional :math:`G` is defind as :math:`G(x) = constant`
    for all values of :math:`x`. The proximal operator of this functional is
    the identity operator

    .. math::

        \mathrm{prox}_{\\sigma G}(x) = x

    Note that it is independent of :math:`\\sigma`.
    """

    def identity_factory(sigma):
        """Return an instance of the proximal operator.

        Parameters
        ----------
        sigma : positive float
            Unused step size parameter. Introduced to provide a unified
            interface.

        Returns
        -------
        id : `IdentityOperator`
            The proximal operator instance of G = 0 which is the
            identity operator
        """

        return IdentityOperator(space)

    return identity_factory


def proximal_box_constraint(space, lower=None, upper=None):
    """Proximal operator factory for ``G(x) = ind(a <= x <= b)``.

    If P is the set of elements with a <= x <= b, the indicator function of
    which is defined as::

        ind(a <= x <= b) = {0 if x in P, infinity if x is not in P}

    with x being an element in ``space``.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of the functional G(x)
    lower : ``space.field`` element or ``space`` `element-like`, optional
        The lower bound.
        Default: ``None``, interpreted as -infinity
    upper : ``space.field`` element or ``space`` `element-like`, optional
        The upper bound.
        Default: ``None``, interpreted as +infinity

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized

    Notes
    -----
    If :math:`P` is an interval :math:`[a,b]`, the indicator function is
    defined as

    .. math::

        I_{P}(x) = \\begin{cases}
        0 & \\text{if } x \\in P, \\\\
        \\infty & \\text{if } x \\not \\in P
        \\end{cases}

    For a step size :math:`\\sigma`, the proximal operator of
    :math:`\\sigma I_{P}` is given by the projection onto the interval

    .. math::

         \mathrm{prox}_{\\sigma I_{P}}(x) = \\begin{cases}
         a & \\text{if } x < a, \\\\
         x & \\text{if } x \\in [a,b], \\\\
         b & \\text{if } x > b.
         \\end{cases}

    The proximal operator is independent of :math:`\\sigma` and invariant under
    a positive rescaling of :math:`I_{P}(x)`, since that leaves the indicator
    function unchanged.

    For spaces of the form :math:`R^n`, the definition extends naturally
    in each component.

    See Also
    --------
    proximal_nonnegativity : Special case with ``lower=0, upper=infty``
    """

    # Convert element-likes if needed, also does some space checking
    if lower is not None and lower not in space and lower not in space.field:
        lower = space.element(lower)
    if upper is not None and upper not in space and upper not in space.field:
        upper = space.element(upper)

    if lower in space.field and upper in space.field:
        if lower > upper:
            raise ValueError('invalid values, `lower` ({}) > `upper` ({})'
                             ''.format(lower, upper))

    class ProxOpBoxConstraint(Operator):

        """Proximal operator for G(x) = ind(a <= x <= b)."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float
                Step size parameter, not used.
            """
            super().__init__(domain=space, range=space, linear=False)

        def _call(self, x, out):
            """Apply the operator to ``x`` and store the result in ``out``."""

            if lower is not None and upper is None:
                x.ufunc.maximum(lower, out=out)
            elif lower is None and upper is not None:
                x.ufunc.minimum(upper, out=out)
            elif lower is not None and upper is not None:
                x.ufunc.maximum(lower, out=out)
                out.ufunc.minimum(upper, out=out)
            else:
                out.assign(x)

    return ProxOpBoxConstraint


def proximal_nonnegativity(space):
    """Function to create the proximal operator of ``G(x) = ind(x >= 0)``.

    Function for the proximal operator of the functional ``G(x)=ind(x >= 0)``
    to be initialized.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of the functional G(x)

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized

    See Also
    --------
    proximal_box_constraint
    """

    return proximal_box_constraint(space, lower=0)


def proximal_cconj_l2(space, lam=1, g=None):
    """Proximal operator factory of the convex conj of the l2-norm/distance.

    Function for the proximal operator of the convex conjugate of the
    functional F where F is the l2-norm (or distance to g, if given)::

        F(x) =  lam ||x - g||_2

    with x and g elements in ``space``, scaling factor lam, and given data g.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of F(x). Needs to be a Hilbert space.
        That is, have an inner product (`LinearSpace.inner`).
    g : ``space`` element
        An element in ``space``
    lam : positive float
        Scaling factor or regularization parameter

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized

    Notes
    -----
    Most problems are forumlated for the squared norm/distance, in that case
    use the `proximal_cconj_l2_squared` instead.

    The :math:`L_2`-norm/distance :math:`F` is given by is given by

    .. math::

        F(x) = \\lambda \|x - g\|_2

    The convex conjugate :math:`F^*` of :math:`F` is given by

    .. math::

        F^*(y) = \\begin{cases}
        0 & \\text{if } \|y-g\|_2 \leq \\lambda, \\\\
        \\infty & \\text{else.}
        \\end{cases}

    For a step size :math:`\\sigma`, the proximal operator of
    :math:`\\sigma F^*` is given by the projection onto the set of :math:`y`
    satisfying :math:`\|y-g\|_2 \leq \\lambda`, i.e., by

    .. math::

        \mathrm{prox}_{\\sigma F^*}(y) = \\begin{cases}
        \\lambda \\frac{y - g}{\|y - g\|}
        & \\text{if } \|y-g\|_2 > \\lambda, \\\\
        y & \\text{if } \|y-g\|_2 \leq \\lambda
        \\end{cases}

    Note that the expression is independent of :math:`\sigma`.

    See Also
    --------
    proximal_l2 : proximal without convex conjugate
    proximal_cconj_l2_squared : proximal for squared norm/distance
    """
    prox_l2 = proximal_l2(space, lam=lam, g=g)
    return proximal_cconj(prox_l2)


def proximal_l2(space, lam=1, g=None):
    """Proximal operator factory of the l2-norm/distance.

    Function for the proximal operator of the functional ``F`` where ``F``
    is the l2-norm (or distance to g, if given)::

        ``F(x) =  lam ||x - g||_2``

    Parameters
    ----------
    space : `LinearSpace`
        Domain of F(x). Needs to be a Hilbert space.
        That is, have an inner product (`LinearSpace.inner`).
    g : ``space`` element
        An element in ``space``.
    lam : positive float
        Scaling factor or regularization parameter.

    Returns
    -------
    prox_factory : callable
        Factory for the proximal operator to be initialized.

    Notes
    -----
    Most problems are forumlated for the squared norm/distance, in that case
    use `proximal_l2_squared` instead.

    The :math:`L_2`-norm/distance :math:`F` is given by

    .. math::

        F(x) = \\lambda \|x - g\|_2

    For a step size :math:`\\sigma`, the proximal operator of :math:`\\sigma F`
    is given by

    .. math::

        \mathrm{prox}_{\\sigma F}(y) = \\begin{cases}
        \\frac{1 - c}{\|y-g\|} \\cdot y  + c \cdot g
        & \\text{if } c < g, \\\\
        g & \\text{else},
        \\end{cases}

    where :math:`c = \\sigma \\frac{\\lambda}{\|y - g\|_2}`.

    See Also
    --------
    proximal_l2_squared : proximal for squared norm/distance
    proximal_cconj_l2 : proximal for convex conjugate
    """
    lam = float(lam)

    if g is not None and g not in space:
        raise TypeError('{!r} is not an element of {!r}'.format(g, space))

    class ProximalL2(Operator):

        """Proximal operator of the l2-norm/distance."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float
                Step size parameter
            """
            self.sigma = float(sigma)
            super().__init__(domain=space, range=space, linear=False)

        def _call(self, x, out):
            """Apply the operator to ``x`` and stores the result in ``out``."""

            dtype = getattr(self.domain, 'dtype', float)
            eps = np.finfo(dtype).resolution * 10

            if g is None:
                x_norm = x.norm() * (1 + eps)
                if x_norm > 0:
                    step = self.sigma * lam / x_norm
                else:
                    step = np.infty

                if step < 1.0:
                    out.lincomb(1.0 - step, x)
                else:
                    out.set_zero()

            else:
                x_norm = (x - g).norm() * (1 + eps)
                if x_norm > 0:
                    step = self.sigma * lam / x_norm
                else:
                    step = np.infty

                if step < 1.0:
                    out.lincomb(1.0 - step, x, step, g)
                else:
                    out.assign(g)

    return ProximalL2


def proximal_cconj_l2_squared(space, lam=1, g=None):
    """Proximal operator factory of the convex conj of the squared l2-norm/dist

    Function for the proximal operator of the convex conjugate of the
    functional F where F is the l2-norm (or distance to g, if given)::

        F(x) =  lam ||x - g||_2^2

    with x and g elements in ``space``, scaling factor lam, and given data g.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of F(x). Needs to be a Hilbert space.
        That is, have an inner product (`LinearSpace.inner`).
    g : ``space`` element
        An element in ``space``
    lam : positive float
        Scaling factor or regularization parameter

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized

    Notes
    -----
    The squared :math:`L_2`-norm/distance :math:`F` is given by

    .. math::

        F(x) =  \\lambda \|x - g\|_2^2.

    The convex conjugate :math:`F^*` of :math:`F` is given by

    .. math::

        F^*(y) = \\frac{1}{4\\lambda} \left( \|
        y\|_2^2 + \langle y, g \\rangle \\right)

    For a step size :math:`\\sigma`, the proximal operator of
    :math:`\\sigma F^*` is given by

    .. math::

        \mathrm{prox}_{\\sigma F^*}(y) = \\frac{y - \\sigma g}{1 +
        \\sigma/(2 \\lambda)}

    See Also
    --------
    proximal_cconj_l2 : proximal without square
    proximal_l2_squared : proximal without convex conjugate
    """
    lam = float(lam)

    if g is not None and g not in space:
        raise TypeError('{!r} is not an element of {!r}'.format(g, space))

    class ProximalCConjL2Squared(Operator):

        """Proximal operator of the convex conj of the squared l2-norm/dist."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float
                Step size parameter
            """
            self.sigma = float(sigma)
            super().__init__(domain=space, range=space, linear=g is None)

        def _call(self, x, out):
            """Apply the operator to ``x`` and stores the result in ``out``"""

            # (x - sig*g) / (1 + sig/(2 lam))

            sig = self.sigma
            if g is None:
                out.lincomb(1.0 / (1 + 0.5 * sig / lam), x)
            else:
                out.lincomb(1.0 / (1 + 0.5 * sig / lam), x,
                            -sig / (1 + 0.5 * sig / lam), g)

    return ProximalCConjL2Squared


def proximal_l2_squared(space, lam=1, g=None):
    """Proximal operator factory of the squared l2-norm/distance.

    Function for the proximal operator of the convex conjugate of the
    functional F where F is the l2-norm (or distance to g, if given)::

        F(x) =  lam ||x - g||_2^2

    with x and g elements in ``space``, scaling factor lam, and given data g.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of F(x). Needs to be a Hilbert space.
        That is, have an inner product (`LinearSpace.inner`).
    g : ``space`` element
        An element in ``space``
    lam : positive float
        Scaling factor or regularization parameter

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized

    Notes
    -----
    The squared :math:`L_2`-norm/distance :math:`F` is given by

    .. math::

        F(x) =  \\lambda \|x - g\|_2^2.

    For a step size :math:`\\sigma`, the proximal operator of :math:`\\sigma F`
    is given by

    .. math::

        \mathrm{prox}_{\\sigma F}(x) = \\frac{x + 2 \\sigma \\lambda g}
        {1 + 2 \\sigma \\lambda}.

    See Also
    --------
    proximal_l2 : proximal without square
    proximal_cconj_l2_squared : proximal for convex conjugate
    """
    # TODO: optimize
    prox_cc_l2_squared = proximal_cconj_l2_squared(space, lam=lam, g=g)
    return proximal_cconj(prox_cc_l2_squared)


def proximal_cconj_l1(space, lam=1, g=None, isotropic=False):
    """Proximal operator factory of the convex conj of the l1-norm/distance.

    Function for the proximal operator of the convex conjugate of the
    functional ``F`` where ``F`` is an l1-norm (or distance to g, if given)::

        F(x) = lam ||x - g||_1

    with x and g elements in ``space`` and scaling factor lam.

    Parameters
    ----------
    space : `LinearSpace` or `ProductSpace` of `LinearSpace` spaces
        Domain of the functional F
    g : ``space`` element
        An element in ``space``
    lam : positive float
        Scaling factor or regularization parameter
    isotropic : bool
        If ``True``, take the vectorial 2-norm point-wise. Otherwise,
        use the vectorial 1-norm. Only available if ``space`` is a
        `ProductSpace`.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized

    Notes
    -----
    The :math:`L_1`-norm/distance :math:`F` is the functional

    .. math::

        F(x) = \\lambda \|x - g\|_1.

    The convex conjugate :math:`F^*` of :math:`F` is given by the indicator
    function of the set :math:`Q_\\lambda`

    .. math::

        F^*(y) = I_{Q_\\lambda} \left(
        \left| \\frac{y}{\\lambda} \\right| +
        \left\\langle \\frac{y}{\\lambda}, g \\right\\rangle
        \\right)

    where :math:`Q_\\lambda` is a hypercube centered at the origin with
    width :math:`2 \\lambda`.

    For a step size :math:`\\sigma`, the proximal operator of
    :math:`\\sigma F^*` is given by

    .. math::

        \mathrm{prox}_{\\sigma F^*}(y) = \\frac{\\lambda (y - \\sigma g)}{
        \\max(\\lambda, |y - \\sigma g|)}

    An alternative formulation is available for `ProductSpace`'s, in which
    case the ``isotropic`` parameter can be used, giving

    .. math::

        F(x) = \\lambda \| \|x - g\|_2 \|_1

    In this case, the dual is

    .. math::

        F^*(y) = \\lambda I_{Q_\\lambda} \left(
        \left|\left| \\frac{y}{\\lambda} \\right|\\right|_2 +
        \left\langle \\frac{y}{\\lambda}, g \\right\\rangle
        \\right)

    For a step size :math:`\\sigma`, the proximal operator of
    :math:`\\sigma F^*` is given by

    .. math::

        \mathrm{prox}_{\\sigma F^*}(y) = \\frac{\\lambda (y - \\sigma g)}{
        \\max(\\lambda, \|y - \\sigma g\|_2)}

    where :math:`\\max( \cdot , \cdot)` thresholds the lower bound of
    :math:`\|y\|_2` point-wise.

    See Also
    --------
    proximal_l1 : proximal without convex conjugate conjugate
    """
    lam = float(lam)

    if g is not None and g not in space:
        raise TypeError('{!r} is not an element of {!r}'.format(g, space))

    if isotropic and not isinstance(space, ProductSpace):
        raise TypeError('`isotropic` given without productspace `space`({})'
                        ''.format(space))
    if (isotropic and isinstance(space, ProductSpace) and
            not space.is_power_space):
        raise TypeError('`isotropic` given with non-powerspace `space`({})'
                        ''.format(space))

    class ProximalCConjL1(Operator):

        """Proximal operator of the convex conj of the l1-norm/distance."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float
                Step size parameter
            """
            # sigma is not used
            self.sigma = float(sigma)
            super().__init__(domain=space, range=space, linear=False)

        def _call(self, x, out):
            """Apply the operator to ``x`` and stores the result in ``out``."""

            # lam * (x - sigma * g) / max(lam, |x - sigma * g|)

            if g is not None:
                diff = x - self.sigma * g
            else:
                diff = x

            if isotropic:
                # Calculate |x| = pointwise 2-norm of x

                tmp = diff[0] ** 2
                sq_tmp = x[0].space.element()
                for x_i in diff[1:]:
                    x_i.multiply(x_i, out=sq_tmp)
                    tmp += sq_tmp
                tmp.ufunc.sqrt(out=tmp)

                # Pointwise maximum of |x| and lambda
                tmp.ufunc.maximum(lam, out=tmp)

                # Global scaling
                tmp /= lam

                # Pointwise division
                for out_i, x_i in zip(out, diff):
                    x_i.divide(tmp, out=out_i)

            else:
                # Calculate |x| = pointwise 2-norm of x
                diff.ufunc.absolute(out=out)

                # Pointwise maximum of |x| and lambda
                out.ufunc.maximum(lam, out=out)

                # Global scaling
                out /= lam

                # Pointwise division
                diff.divide(out, out=out)

    return ProximalCConjL1


def proximal_l1(space, lam=1, g=None, isotropic=False):
    """Proximal operator factory of the l1-norm/distance.

    Function for the proximal operator of the functional F where F is an
    l1-norm (or distance to g, if given)::

        F(x) = lam ||x - g||_1

    with x and g elements in ``space``, and scaling factor lam.

    Parameters
    ----------
    space : `LinearSpace` or `ProductSpace`
        Domain of the functional.
    g : ``space`` element
        An element in ``space``.
    lam : positive float
        Scaling factor or regularization parameter.
    isotropic : bool
        If ``True``, take the vectorial 2-norm point-wise. Otherwise,
        use the vectorial 1-norm. Only available if ``space`` is a
        `ProductSpace`.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized

    Notes
    -----
    The :math:`L_1`-norm/distance :math:`F` is the functional

    .. math::

        F(x) = \\lambda \|x - g\|_1.

    For a step size :math:`\\sigma`, the proximal operator of :math:`\\sigma F`
    is

    .. math::

        \mathrm{prox}_{\\sigma F}(y) = \\begin{cases}
        y - \\sigma \\lambda
        & \\text{if } y > g + \\sigma \\lambda, \\\\
        0
        & \\text{if } g - \\sigma \\lambda \\leq y \\leq g +
        \\sigma \\lambda \\\\
        y + \\sigma \\lambda
        & \\text{if } y < g - \\sigma \\lambda,
        \\end{cases}

    An alternative formulation is available for `ProductSpace`'s, where the
    the ``isotropic`` parameter can be used, giving

    .. math::

        F(x) = \\lambda \| \|x - g\|_2 \|_1

    The proximal can be calculated using the Moreau equality (also known as
    Moreau decomposition or Moreau identity). See for example [BC2011]_.

    See Also
    --------
    proximal_cconj_l1 : proximal for convex conjugate
    """
    # TODO: optimize
    prox_cc_l1 = proximal_cconj_l1(space, lam=lam, g=g, isotropic=isotropic)
    return proximal_cconj(prox_cc_l1)


def proximal_cconj_kl(space, lam=1, g=None):
    """Proximal operator factory of the convex conjugate of the KL divergence.

    Function returning the proximal operator of the convex conjugate of the
    functional F where F is the entropy-type Kullback-Leibler (KL) divergence::

        F(x) = sum_i (x_i - g_i + g_i ln(g_i) - g_i ln(pos(x_i))) + ind_P(x)

    with ``x`` and ``g`` elements in the linear space ``X``, and ``g``
    non-negative. Here, ``pos`` denotes the nonnegative part, and ``ind_P`` is
    the indicator function for nonnegativity.

    Parameters
    ----------
    space : `FnBase`
        Space X which is the domain of the functional F
    g : ``space`` element, optional
        Data term, positive. If None it is take as the one-element.
    lam : positive float
        Scaling factor.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized.

    See Also
    --------
    proximal_cconj_kl_cross_entropy : proximal for releated functional

    Notes
    -----
    The functional is given by the expression

    .. math::

        F(x) = \\sum_i (x_i - g_i + g_i \\ln(g_i) - g_i \\ln(pos(x_i))) +
        I_{x \\geq 0}(x)

    The indicator function :math:`I_{x \geq 0}(x)` is used to restrict the
    domain of :math:`F` such that :math:`F` is defined over whole space
    :math:`X`. The non-negativity thresholding :math:`pos` is used to define
    :math:`F` in the real numbers.

    Note that the functional is not well-defined without a prior g. Hence, if g
    is omitted this will be interpreted as if g is equal to the one-element.

    The convex conjugate :math:`F^*` of :math:`F` is

    .. math::

        F^*(p) = \\sum_i (-g_i \\ln(pos({1_X}_i - p_i))) +
        I_{1_X - p \geq 0}(p)

    where :math:`p` is the variable dual to :math:`x`, and :math:`1_X` is an
    element of the space :math:`X` with all components set to 1.

    The proximal operator of the convex conjugate of F is

    .. math::

        \mathrm{prox}_{\\sigma (\\lambda F)^*}(x) =
        \\frac{\\lambda 1_X + x - \\sqrt{(x -  \\lambda 1_X)^2 +
        4 \\lambda \\sigma g}}{2}

    where :math:`\\sigma` is the step size-like parameter, and :math:`\\lambda`
    is the weighting in front of the function :math:`F`.

    KL based objectives are common in MLEM optimization problems and are often
    used when data noise governed by a multivariate Poisson probability
    distribution is significant.

    The intermediate image estimates can have negative values even though
    the converged solution will be non-negative. Non-negative intermediate
    image estimates can be enforced by adding an indicator function ind_P
    the primal objective.

    This functional :math:`F`, described above, is related to the
    Kullback-Leibler cross entropy functional. The KL cross entropy is the one
    diescribed in `this Wikipedia article
    <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_, and
    the functional :math:`F` is obtained by switching place of the prior and
    the varialbe in the KL cross entropy functional. See the See Also section.
    """
    lam = float(lam)

    if g is not None and g not in space:
        raise TypeError('{} is not an element of {}'.format(g, space))

    class ProximalCConjKL(Operator):

        """Proximal operator of the convex conjugate of the KL divergence."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float
            """
            self.sigma = float(sigma)
            super().__init__(domain=space, range=space, linear=False)

        def _call(self, x, out):
            """Apply the operator to ``x`` and stores the result in ``out``."""

            # 1 / 2 (lam_X + x - sqrt((x - lam_X) ^ 2 + 4; lam sigma g)

            # out = x - lam_X
            out.assign(x)
            out -= lam

            # (out)^2
            out.ufunc.square(out=out)

            # out = out + 4 lam sigma g
            # If g is None, it is taken as the one element
            if g is None:
                out += 4.0 * lam * self.sigma
            else:
                out.lincomb(1, out, 4.0 * lam * self.sigma, g)

            # out = sqrt(out)
            out.ufunc.sqrt(out=out)

            # out = x - out
            out.lincomb(1, x, -1, out)

            # out = lam_X + out
            out.lincomb(lam, space.one(), 1, out)

            # out = 1/2 * out
            out /= 2

    return ProximalCConjKL


def proximal_cconj_kl_cross_entropy(space, lam=1, g=None):
    """Proximal factory of the convex conjugate of cross entropy KL divergence.

    Function returning the proximal facotry of the convex conjugate of the
    functional F, where F is the corss entorpy Kullback-Leibler (KL)
    divergence given by::

        F(x) = sum_i (x_i ln(pos(x_i)) - x_i ln(g_i) + g_i - x_i) + ind_P(x)

    with ``x`` and ``g`` in the linear space ``X``, and ``g`` non-negative.
    Here, ``pos`` denotes the nonnegative part, and ``ind_P`` is the indicator
    function for nonnegativity.

    Parameters
    ----------
    space : `FnBase`
        Space X which is the domain of the functional F
    g : ``space`` element, optional
        Data term, positive. If None it is take as the one-element.
    lam : positive float
        Scaling factor.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized.


    See Also
    --------
    proximal_cconj_kl : proximal for related functional

    Notes
    -----
    The functional is given by the expression

    .. math::

        F(x) = \\sum_i (x_i \\ln(pos(x_i)) - x_i \\ln(g_i) + g_i - x_i) +
        I_{x \\geq 0}(x)

    The indicator function :math:`I_{x \geq 0}(x)` is used to restrict the
    domain of :math:`F` such that :math:`F` is defined over whole space
    :math:`X`. The non-negativity thresholding :math:`pos` is used to define
    :math:`F` in the real numbers.

    Note that the functional is not well-defined without a prior g. Hence, if g
    is omitted this will be interpreted as if g is equal to the one-element.

    The convex conjugate :math:`F^*` of :math:`F` is

    .. math::

        F^*(p) = \\sum_i g_i (exp(p_i) - 1)

    where :math:`p` is the variable dual to :math:`x`.

    The proximal operator of the convex conjugate of :math:`F` is

    .. math::

        \mathrm{prox}_{\\sigma (\\lambda F)^*}(x) = x - \\lambda
        W(\\frac{\\sigma}{\\lambda} g e^{x/\\lambda})

    where :math:`\\sigma` is the step size-like parameter, :math:`\\lambda` is
    the weighting in front of the function :math:`F`, and :math:`W` is the
    Lambert W function (see, for example, the
    `Wikipedia article <https://en.wikipedia.org/wiki/Lambert_W_function>`_).

    For real-valued input x, the Lambert :math:`W` function is defined only for
    :math:`x \\geq -1/e`, and it has two branches for values
    :math:`-1/e \\leq x < 0`. However, for inteneded use-cases, where
    :math:`\\lambda` and :math:`g` are positive, the argument of :math:`W`
    will always be positive.

    `Wikipedia article on Kullback Leibler divergence
    <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_.
    For further information about the functional, see for example `this article
    <http://ieeexplore.ieee.org/document/1056144/?arnumber=1056144>`_.

    The KL cross entropy functional :math:`F`, described above, is related to
    another functional functional also know as KL divergence. This functional
    is often used as data discrepancy term in inverse problems, when data is
    corrupted with Poisson noise. This functional is obtained by changing place
    of the prior and the variable. See the See Also section.
    """
    lam = float(lam)

    if g is not None and g not in space:
        raise TypeError('{} is not an element of {}'.format(g, space))

    class ProximalCConjKLCrossEntropy(Operator):

        """Proximal operator of conjugate of cross entropy KL divergence."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float
            """
            self.sigma = float(sigma)
            super().__init__(domain=space, range=space, linear=False)

        def _call(self, x, out):
            """Apply the operator to ``x`` and stores the result in ``out``."""

            if g is None:
                # If g is None, it is taken as the one element
                # Different branches of lambertw is not an issue, see Notes
                out.lincomb(1, x, -lam, scipy.special.lambertw(
                    (self.sigma / lam) * np.exp(x / lam)))
            else:
                # Different branches of lambertw is not an issue, see Notes
                out.lincomb(1, x,
                            -lam, scipy.special.lambertw(
                                (self.sigma / lam) * g * np.exp(x / lam)))

    return ProximalCConjKLCrossEntropy


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
