# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Factory functions for creating proximal operators.

Functions with ``convex_conj`` mean the proximal of the convex conjugate and
are provided for convenience.

For more details see :ref:`proximal_operators` and references therein. For
more details on proximal operators including how to evaluate the proximal
operator of a variety of functions see [PB2014].

References
----------
[PB2014] Parikh, N, and Boyd, S. *Proximal Algorithms*.
Foundations and Trends in Optimization, 1 (2014), pp 127-239.
"""

from __future__ import print_function, division, absolute_import
import numpy as np

from odl.operator import (
    Operator, IdentityOperator, ConstantOperator, DiagonalOperator,
    PointwiseNorm, MultiplyOperator)
from odl.space import ProductSpace
from odl.set.space import LinearSpaceElement


__all__ = ('combine_proximals', 'proximal_convex_conj', 'proximal_translation',
           'proximal_arg_scaling', 'proximal_quadratic_perturbation',
           'proximal_composition', 'proximal_const_func',
           'proximal_box_constraint', 'proximal_nonnegativity',
           'proximal_l1', 'proximal_convex_conj_l1',
           'proximal_l2', 'proximal_convex_conj_l2',
           'proximal_linfty', 'proximal_convex_conj_linfty',
           'proj_simplex', 'proj_l1',
           'proximal_l2_squared', 'proximal_convex_conj_l2_squared',
           'proximal_l1_l2', 'proximal_convex_conj_l1_l2',
           'proximal_convex_conj_kl', 'proximal_convex_conj_kl_cross_entropy',
           'proximal_huber')


def combine_proximals(*factory_list):
    r"""Combine proximal operators into a diagonal product space operator.

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
        \mathrm{prox}_{\sigma (F(x) + G(y))}(x, y) =
        (\mathrm{prox}_{\sigma F}(x), \mathrm{prox}_{\sigma G}(y)).
    """
    def diag_op_factory(sigma):
        """Diagonal matrix of operators.

        Parameters
        ----------
        sigma : positive float or sequence of positive floats
            Step size parameter(s), if a sequence, the length must match
            the length of the ``factory_list``.

        Returns
        -------
        diag_op : `DiagonalOperator`
        """
        if np.isscalar(sigma):
            sigma = [sigma] * len(factory_list)

        return DiagonalOperator(
            *[factory(sigmai)
              for sigmai, factory in zip(sigma, factory_list)])

    return diag_op_factory


def proximal_convex_conj(prox_factory):
    r"""Calculate the proximal of the dual using Moreau decomposition.

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
        \mathrm{prox}_{\sigma F^*}(x) +\sigma \,
        \mathrm{prox}_{F / \sigma}(x / \sigma) = x

    where :math:`\sigma` is a scalar step size. Using this, the proximal of
    the convex conjugate is given by

    .. math::
        \mathrm{prox}_{\sigma F^*}(x) =
        x - \sigma \, \mathrm{prox}_{F / \sigma}(x / \sigma)

    Note that since :math:`(F^*)^* = F`, this can be used to get the proximal
    of the original function from the proximal of the convex conjugate.

    For reference on the Moreau identity, see [CP2011c].

    References
    ----------
    [CP2011c] Combettes, P L, and Pesquet, J-C. *Proximal splitting
    methods in signal processing.* In:  Bauschke, H H, Burachik, R S,
    Combettes, P L, Elser, V, Luke, D R, and Wolkowicz, H. Fixed-point
    algorithms for inverse problems in science and engineering, Springer,
    2011.
    """
    def convex_conj_prox_factory(sigma):
        """Create proximal for the dual with a given sigma.

        Parameters
        ----------
        sigma : positive float or array-like
            Step size parameter. Can be a pointwise positive space element or
            a sequence of positive floats if `prox_factory` supports that.

        Returns
        -------
        proximal : `Operator`
            The proximal operator of ``s * F^*`` where ``s`` is the step size
        """

        # Get the underlying space. At the same time, check if the given
        # prox_factory accepts stepsize objects of the type given by sigma.
        space = prox_factory(sigma).domain

        mult_inner = MultiplyOperator(1.0 / sigma, domain=space, range=space)
        mult_outer = MultiplyOperator(sigma, domain=space, range=space)
        result = (IdentityOperator(space) -
                  mult_outer * prox_factory(1.0 / sigma) * mult_inner)
        return result

    return convex_conj_prox_factory


def proximal_translation(prox_factory, y):
    r"""Calculate the proximal of the translated function F(x - y).

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
        \mathrm{prox}_{\sigma F( \cdot - y)}(x) =
        y + \mathrm{prox}_{\sigma F}(x - y)

    where :math:`y` is the translation, and :math:`\sigma` is the step size.

    For reference on the identity used, see [CP2011c].

    References
    ----------
    [CP2011c] Combettes, P L, and Pesquet, J-C. *Proximal splitting
    methods in signal processing.* In:  Bauschke, H H, Burachik, R S,
    Combettes, P L, Elser, V, Luke, D R, and Wolkowicz, H. Fixed-point
    algorithms for inverse problems in science and engineering, Springer,
    2011.
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
    r"""Calculate the proximal of function F(x * scaling).

    Parameters
    ----------
    prox_factory : callable
        A factory function that, when called with a step size, returns the
        proximal operator of ``F``
    scaling : float or sequence of floats or space element
        Scaling parameter. The permissible types depent on the stepsizes
        accepted by prox_factory. It may not contain any nonzero imaginary
        parts. If it is a scalar, it may be zero, in which case the
        resulting proxmial operator is the identity. If not a scalar,
        it may not contain any zero components.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized

    Notes
    -----
    Given a functional :math:`F`, and scaling factor :math:`\alpha` this is
    calculated according to the rule

    .. math::
        \mathrm{prox}_{\sigma F(\alpha \, \cdot)}(x) =
        \frac{1}{\alpha}
        \mathrm{prox}_{\sigma \alpha^2 F(\cdot) }(\alpha x)

    where :math:`\sigma` is the step size.

    For reference on the identity used, see [CP2011c].

    References
    ----------
    [CP2011c] Combettes, P L, and Pesquet, J-C. *Proximal splitting
    methods in signal processing.* In:  Bauschke, H H, Burachik, R S,
    Combettes, P L, Elser, V, Luke, D R, and Wolkowicz, H. Fixed-point
    algorithms for inverse problems in science and engineering, Springer,
    2011.
    """

    # To begin, we could check for two things:
    # * Currently, we do not support complex scaling. We could therefore catch
    #   nonempty imaginary parts.
    # * If some components of scaling are zero, then the following routine will
    #   crash with a division-by-zero error. The correct solution would be to
    #   just keep these components and do the following computations only for
    #   the others.
    # Since these checks are computationally expensive, we do not execute them
    # unconditionally, but only if the scaling factor is a scalar:
    if np.isscalar(scaling):
        if scaling == 0:
            return proximal_const_func(prox_factory(1.0).domain)
        elif scaling.imag != 0:
            raise ValueError("Complex scaling not supported.")
        else:
            scaling = float(scaling.real)
    else:
        scaling = np.asarray(scaling)

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
        scaling_square = scaling * scaling
        prox = prox_factory(sigma * scaling_square)
        space = prox.domain
        mult_inner = MultiplyOperator(scaling, domain=space, range=space)
        mult_outer = MultiplyOperator(1 / scaling, domain=space, range=space)
        return mult_outer * prox * mult_inner

    return arg_scaling_prox_factory


def proximal_quadratic_perturbation(prox_factory, a, u=None):
    r"""Calculate the proximal of function F(x) + a * \|x\|^2 + <u,x>.

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
        \mathrm{prox}_{\sigma \left(F( \cdot ) + a \| \cdot \|^2 +
        <u, \cdot >\right)}(x) =
        c \; \mathrm{prox}_{\sigma F( \cdot \, c)}((x - \sigma u) c)

    where :math:`c` is the constant

    .. math::
        c = \frac{1}{\sqrt{2 \sigma a + 1}},

    :math:`a` is the scaling parameter belonging to the quadratic term,
    :math:`u` is the space element defining the linear functional, and
    :math:`\sigma` is the step size.

    For reference on the identity used, see [CP2011c]. Note that this identity
    is not the exact one given in the reference, but was recalculated for
    arbitrary step lengths.

    References
    ----------
    [CP2011c] Combettes, P L, and Pesquet, J-C. *Proximal splitting
    methods in signal processing.* In:  Bauschke, H H, Burachik, R S,
    Combettes, P L, Elser, V, Luke, D R, and Wolkowicz, H. Fixed-point
    algorithms for inverse problems in science and engineering, Springer,
    2011.
    """
    a = float(a)
    if a < 0:
        raise ValueError('scaling parameter muts be non-negative, got {}'
                         ''.format(a))

    if u is not None and not isinstance(u, LinearSpaceElement):
        raise TypeError('`u` must be `None` or a `LinearSpaceElement` '
                        'instance, got {!r}.'.format(u))

    def quadratic_perturbation_prox_factory(sigma):
        r"""Create proximal for the quadratic perturbation with a given sigma.

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
        if np.isscalar(sigma):
            sigma = float(sigma)
        else:
            sigma = np.asarray(sigma)

        const = 1.0 / np.sqrt(sigma * 2.0 * a + 1)
        prox = proximal_arg_scaling(prox_factory, const)(sigma)
        if u is not None:
            return (MultiplyOperator(const, domain=u.space, range=u.space) *
                    prox *
                    (MultiplyOperator(const, domain=u.space, range=u.space) -
                     sigma * const * u))
        else:
            space = prox.domain
            return (MultiplyOperator(const, domain=space, range=space) *
                    prox * MultiplyOperator(const, domain=space, range=space))

    return quadratic_perturbation_prox_factory


def proximal_composition(proximal, operator, mu):
    r"""Proximal operator factory of functional composed with unitary operator.

    For a functional ``F`` and a linear unitary `Operator` ``L`` this is the
    factory for the proximal operator of ``F * L``.

    Parameters
    ----------
    proximal : callable
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
    :math:`\mu`

    .. math::
        L^*(L(x)) = \mu * x

    and a convex function :math:`F`, the following identity holds

    .. math::
        \mathrm{prox}_{\sigma F \circ L}(x) = x + \frac{1}{\mu}
        L^* \left( \mathrm{prox}_{\mu \sigma F}(Lx) - Lx \right)

    This factory function implements this functionality.

    There is no simple formula for more general operators.

    The function cannot verify that the operator is unitary, the user needs
    to verify this.

    For reference on the identity used, see [CP2011c].

    References
    ----------
    [CP2011c] Combettes, P L, and Pesquet, J-C. *Proximal splitting
    methods in signal processing.* In:  Bauschke, H H, Burachik, R S,
    Combettes, P L, Elser, V, Luke, D R, and Wolkowicz, H. Fixed-point
    algorithms for inverse problems in science and engineering, Springer,
    2011.
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
    r"""Proximal operator factory of the constant functional.

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
        \mathrm{prox}_{\sigma G}(x) = x

    Note that it is independent of :math:`\sigma`.
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
    r"""Proximal operator factory for ``G(x) = ind(a <= x <= b)``.

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
        I_{P}(x) = \begin{cases}
        0 & \text{if } x \in P, \\
        \infty & \text{if } x \not \in P
        \end{cases}

    For a step size :math:`\sigma`, the proximal operator of
    :math:`\sigma I_{P}` is given by the projection onto the interval

    .. math::
         \mathrm{prox}_{\sigma I_{P}}(x) = \begin{cases}
         a & \text{if } x < a, \\
         x & \text{if } x \in [a,b], \\
         b & \text{if } x > b.
         \end{cases}

    The proximal operator is independent of :math:`\sigma` and invariant under
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
            super(ProxOpBoxConstraint, self).__init__(
                domain=space, range=space, linear=False)

        def _call(self, x, out):
            """Apply the operator to ``x`` and store the result in ``out``."""
            if lower is not None and upper is None:
                x.ufuncs.maximum(lower, out=out)
            elif lower is None and upper is not None:
                x.ufuncs.minimum(upper, out=out)
            elif lower is not None and upper is not None:
                x.ufuncs.maximum(lower, out=out)
                out.ufuncs.minimum(upper, out=out)
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


def proximal_convex_conj_l2(space, lam=1, g=None):
    r"""Proximal operator factory of the convex conj of the l2-norm/distance.

    Function for the proximal operator of the convex conjugate of the
    functional F where F is the l2-norm (or distance to g, if given)::

        F(x) =  lam ||x - g||_2

    with x and g elements in ``space``, scaling factor lam, and given data g.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of F(x). Needs to be a Hilbert space.
        That is, have an inner product (`LinearSpace.inner`).
    lam : positive float, optional
        Scaling factor or regularization parameter.
    g : ``space`` element, optional
        An element in ``space``. Default: ``space.zero``.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized

    Notes
    -----
    Most problems are forumlated for the squared norm/distance, in that case
    use the `proximal_convex_conj_l2_squared` instead.

    The :math:`L_2`-norm/distance :math:`F` is given by is given by

    .. math::
        F(x) = \lambda \|x - g\|_2

    The convex conjugate :math:`F^*` of :math:`F` is given by

    .. math::
        F^*(y) = \begin{cases}
        0 & \text{if } \|y-g\|_2 \leq \lambda, \\
        \infty & \text{else.}
        \end{cases}

    For a step size :math:`\sigma`, the proximal operator of
    :math:`\sigma F^*` is given by the projection onto the set of :math:`y`
    satisfying :math:`\|y-g\|_2 \leq \lambda`, i.e., by

    .. math::
        \mathrm{prox}_{\sigma F^*}(y) = \begin{cases}
        \lambda \frac{y - g}{\|y - g\|}
        & \text{if } \|y-g\|_2 > \lambda, \\
        y & \text{if } \|y-g\|_2 \leq \lambda
        \end{cases}

    Note that the expression is independent of :math:`\sigma`.

    See Also
    --------
    proximal_l2 : proximal without convex conjugate
    proximal_convex_conj_l2_squared : proximal for squared norm/distance
    """
    prox_l2 = proximal_l2(space, lam=lam, g=g)
    return proximal_convex_conj(prox_l2)


def proximal_l2(space, lam=1, g=None):
    r"""Proximal operator factory of the l2-norm/distance.

    Function for the proximal operator of the functional ``F`` where ``F``
    is the l2-norm (or distance to g, if given)::

        ``F(x) =  lam ||x - g||_2``

    Parameters
    ----------
    space : `LinearSpace`
        Domain of F(x). Needs to be a Hilbert space.
        That is, have an inner product (`LinearSpace.inner`).
    lam : positive float, optional
        Scaling factor or regularization parameter.
    g : ``space`` element, optional
        An element in ``space``. Default: ``space.zero``.

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
        F(x) = \lambda \|x - g\|_2

    For a step size :math:`\sigma`, the proximal operator of :math:`\sigma F`
    is given by

    .. math::
        \mathrm{prox}_{\sigma F}(y) = \begin{cases}
        \frac{1 - c}{\|y-g\|} \cdot y  + c \cdot g
        & \text{if } c < g, \\
        g & \text{else},
        \end{cases}

    where :math:`c = \sigma \frac{\lambda}{\|y - g\|_2}`.

    See Also
    --------
    proximal_l2_squared : proximal for squared norm/distance
    proximal_convex_conj_l2 : proximal for convex conjugate
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
            super(ProximalL2, self).__init__(
                domain=space, range=space, linear=False)
            self.sigma = float(sigma)

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


def proximal_convex_conj_l2_squared(space, lam=1, g=None):
    r"""Proximal operator factory of the convex conj of the squared l2-dist

    Function for the proximal operator of the convex conjugate of the
    functional F where F is the l2-norm (or distance to g, if given)::

        F(x) =  lam ||x - g||_2^2

    with x and g elements in ``space``, scaling factor lam, and given data g.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of F(x). Needs to be a Hilbert space.
        That is, have an inner product (`LinearSpace.inner`).
    lam : positive float, optional
        Scaling factor or regularization parameter.
    g : ``space`` element, optional
        An element in ``space``. Default: ``space.zero``.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized

    Notes
    -----
    The squared :math:`L_2`-norm/distance :math:`F` is given by

    .. math::
        F(x) =  \lambda \|x - g\|_2^2.

    The convex conjugate :math:`F^*` of :math:`F` is given by

    .. math::
        F^*(y) = \frac{1}{4\lambda} \left( \|
        y\|_2^2 + \langle y, g \rangle \right)

    For a step size :math:`\sigma`, the proximal operator of
    :math:`\sigma F^*` is given by

    .. math::
        \mathrm{prox}_{\sigma F^*}(y) = \frac{y - \sigma g}{1 +
        \sigma/(2 \lambda)}

    See Also
    --------
    proximal_convex_conj_l2 : proximal without square
    proximal_l2_squared : proximal without convex conjugate
    """
    lam = float(lam)

    if g is not None and g not in space:
        raise TypeError('{!r} is not an element of {!r}'.format(g, space))

    class ProximalConvexConjL2Squared(Operator):

        """Proximal operator of the convex conj of the squared l2-norm/dist."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float or pointwise positive space.element
                Step size parameter. If scalar, it contains a global stepsize,
                otherwise the space.element defines a stepsize for each point.
            """
            super(ProximalConvexConjL2Squared, self).__init__(
                domain=space, range=space, linear=g is None)
            if np.isscalar(sigma):
                self.sigma = float(sigma)
            else:
                self.sigma = space.element(sigma)

        def _call(self, x, out):
            """Apply the operator to ``x`` and store the result in ``out``"""
            # (x - sig*g) / (1 + sig/(2 lam))
            sig = self.sigma
            if np.isscalar(sig):
                if g is None:
                    out.lincomb(1 / (1 + 0.5 * sig / lam), x)
                else:
                    out.lincomb(1 / (1 + 0.5 * sig / lam), x,
                                -sig / (1 + 0.5 * sig / lam), g)
            elif sig in space:
                if g is None:
                    x.divide(1 + 0.5 / lam * sig, out=out)
                else:
                    if x is out:
                        # Can't write to `out` since old `x` is still needed
                        tmp = sig.multiply(g)
                        out.lincomb(1, x, -1, tmp)
                    else:
                        sig.multiply(g, out=out)
                        out.lincomb(1, x, -1, out)
                    out.divide(1 + 0.5 / lam * sig, out=out)
            else:
                raise RuntimeError(
                    '`sigma` is neither a scalar nor a space element.'
                )

    return ProximalConvexConjL2Squared


def proximal_l2_squared(space, lam=1, g=None):
    r"""Proximal operator factory of the squared l2-norm/distance.

    Function for the proximal operator of the convex conjugate of the
    functional F where F is the l2-norm (or distance to g, if given)::

        F(x) =  lam ||x - g||_2^2

    with x and g elements in ``space``, scaling factor lam, and given data g.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of F(x). Needs to be a Hilbert space.
        That is, have an inner product (`LinearSpace.inner`).
    lam : positive float, optional
        Scaling factor or regularization parameter.
    g : ``space`` element, optional
        An element in ``space``. Default: ``space.zero``.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized

    Notes
    -----
    The squared :math:`L_2`-norm/distance :math:`F` is given by

    .. math::
        F(x) =  \lambda \|x - g\|_2^2.

    For a step size :math:`\sigma`, the proximal operator of :math:`\sigma F`
    is given by

    .. math::
        \mathrm{prox}_{\sigma F}(x) = \frac{x + 2 \sigma \lambda g}
        {1 + 2 \sigma \lambda}.

    See Also
    --------
    proximal_l2 : proximal without square
    proximal_convex_conj_l2_squared : proximal for convex conjugate
    """
    class ProximalL2Squared(Operator):

        """Proximal operator of the squared l2-norm/dist."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float or pointwise positive space.element
                Step size parameter. If scalar, it contains a global stepsize,
                otherwise the space.element defines a stepsize for each point.
            """
            super(ProximalL2Squared, self).__init__(
                domain=space, range=space, linear=g is None)
            if np.isscalar(sigma):
                self.sigma = float(sigma)
            else:
                self.sigma = space.element(sigma)

        def _call(self, x, out):
            """Apply the operator to ``x`` and store the result in ``out``"""
            # (x + 2*sig*lam*g) / (1 + 2*sig*lam))
            sig = self.sigma
            if np.isscalar(sig):
                if g is None:
                    out.lincomb(1 / (1 + 2 * sig * lam), x)
                else:
                    out.lincomb(1 / (1 + 2 * sig * lam), x,
                                2 * sig * lam / (1 + 2 * sig * lam), g)
            else:   # sig in space
                if g is None:
                    x.divide(1 + 2 * sig * lam, out=out)
                else:
                    if x is out:
                        # Can't write to `out` since old `x` is still needed
                        tmp = sig.multiply(2 * lam * g)
                        out.lincomb(1, x, 1, tmp)
                    else:
                        sig.multiply(2 * lam * g, out=out)
                        out.lincomb(1, x, 1, out)
                    out.divide(1 + 2 * sig * lam, out=out)

    return ProximalL2Squared


def proximal_convex_conj_l1(space, lam=1, g=None):
    r"""Proximal operator factory of the L1 norm/distance convex conjugate.

    Implements the proximal operator of the convex conjugate of the
    functional ::

        F(x) = lam ||x - g||_1

    with ``x`` and ``g`` elements in ``space``, and scaling factor ``lam``.

    Parameters
    ----------
    space : `LinearSpace` or `ProductSpace` of `LinearSpace` spaces
        Domain of the functional F
    lam : positive float, optional
        Scaling factor or regularization parameter.
    g : ``space`` element, optional
        Element to which the L1 distance is taken.
        Default: ``space.zero``.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized.

    Notes
    -----
    The convex conjugate :math:`F^*` of the functional

    .. math::
        F(x) = \lambda \|x - g\|_1.

    is in the case of scalar-valued functions given by

    .. math::
        F^*(y) = \iota_{B_\infty} \big( \lambda^{-1}\, y \big) +
        \left\langle \lambda^{-1}\, y,\: g \right\rangle,

    where :math:`\iota_{B_\infty}` is the indicator function of the
    unit ball with respect to :math:`\|\cdot\|_\infty`.
    For vector-valued functions, the convex conjugate is

    .. math::
        F^*(y) = \sum_{k=1}^d F^*(y_k)

    due to separability of the (non-isotropic) 1-norm.

    For a step size :math:`\sigma`, the proximal operator of
    :math:`\sigma F^*` is given by

    .. math::
        \mathrm{prox}_{\sigma F^*}(y) = \frac{\lambda (y - \sigma g)}{
        \max(\lambda, |y - \sigma g|)}

    Here, all operations are to be read pointwise.

    For vector-valued :math:`x` and :math:`g`, the (non-isotropic) proximal
    operator is the component-wise scalar proximal:

    .. math::
        \mathrm{prox}_{\sigma F^*}(x) = \left(
            \mathrm{prox}_{\sigma F^*}(x_1), \dots,
            \mathrm{prox}_{\sigma F^*}(x_d)
            \right),

    where :math:`d` is the number of components of :math:`x`.

    See Also
    --------
    proximal_convex_conj_l1_l2 : isotropic variant for vector-valued functions
    proximal_l1 : proximal without convex conjugate
    """
    # Fix for rounding errors
    dtype = getattr(space, 'dtype', float)
    eps = np.finfo(dtype).resolution * 10
    lam = float(lam * (1 - eps))

    if g is not None and g not in space:
        raise TypeError('{!r} is not an element of {!r}'.format(g, space))

    class ProximalConvexConjL1(Operator):

        """Proximal operator of the L1 norm/distance convex conjugate."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float or pointwise positive space.element
                Step size parameter. If scalar, it contains a global stepsize,
                otherwise the space.element defines a stepsize for each point.
            """
            super(ProximalConvexConjL1, self).__init__(
                domain=space, range=space, linear=False)
            if np.isscalar(sigma):
                self.sigma = float(sigma)
            else:
                self.sigma = space.element(sigma)

        def _call(self, x, out):
            """Return ``self(x, out=out)``."""
            # lam * (x - sig * g) / max(lam, |x - sig * g|)

            # diff = x - sig * g
            if g is not None:
                diff = self.domain.element()
                diff.lincomb(1, x, -self.sigma, g)
            else:
                if x is out:
                    # Handle aliased `x` and `out`
                    # This is necessary since we write to both `diff` and
                    # `out`.
                    diff = x.copy()
                else:
                    diff = x

            # out = max( |x-sig*g|, lam ) / lam
            diff.ufuncs.absolute(out=out)
            out.ufuncs.maximum(lam, out=out)
            out /= lam

            # out = diff / ...
            diff.divide(out, out=out)

    return ProximalConvexConjL1


def proximal_convex_conj_l1_l2(space, lam=1, g=None):
    r"""Proximal operator factory of the L1-L2 norm/distance convex conjugate.

    Implements the proximal operator of the convex conjugate of the
    functional ::

        F(x) = lam || |x - g|_2 ||_1

    with ``x`` and ``g`` elements in ``space``, and scaling factor ``lam``.
    Here, ``|.|_2`` is the pointwise Euclidean norm of a vector-valued
    function.

    Parameters
    ----------
    space : `LinearSpace` or `ProductSpace` of `LinearSpace` spaces
        Domain of the functional F
    lam : positive float, optional
        Scaling factor or regularization parameter.
    g : ``space`` element, optional
        Element to which the L1 distance is taken.
        Default: ``space.zero``.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized.

    Notes
    -----
    The convex conjugate :math:`F^*` of the functional

    .. math::
        F(x) = \lambda \| |x - g|_2 \|_1.

    is given by

    .. math::
        F^*(y) = \iota_{B_\infty} \big( \lambda^{-1}\, |y|_2 \big) +
        \left\langle \lambda^{-1}\, y,\: g \right\rangle,

    where :math:`\iota_{B_\infty}` is the indicator function of the
    unit ball with respect to :math:`\|\cdot\|_\infty`.

    For a step size :math:`\sigma`, the proximal operator of
    :math:`\sigma F^*` is given by

    .. math::
        \mathrm{prox}_{\sigma F^*}(y) = \frac{\lambda (y - \sigma g)}{
        \max(\lambda, |y - \sigma g|_2)}

    Here, all operations are to be read pointwise.

    See Also
    --------
    proximal_convex_conj_l1 : Scalar or non-isotropic vectorial variant
    """
    # Fix for rounding errors
    dtype = getattr(space, 'dtype', float)
    eps = np.finfo(dtype).resolution * 10
    lam = float(lam * (1 - eps))

    if g is not None and g not in space:
        raise TypeError('{!r} is not an element of {!r}'.format(g, space))

    class ProximalConvexConjL1L2(Operator):

        """Proximal operator of the convex conj of the l1-norm/distance."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float
                Step size parameter
            """
            super(ProximalConvexConjL1L2, self).__init__(
                domain=space, range=space, linear=False)
            self.sigma = float(sigma)

        def _call(self, x, out):
            """Return ``self(x, out=out)``."""
            # lam * (x - sig * g) / max(lam, |x - sig * g|)

            # diff = x - sig * g
            if g is not None:
                diff = self.domain.element()
                diff.lincomb(1, x, -self.sigma, g)
            else:
                diff = x

            # denom = max( |x-sig*g|_2, lam ) / lam  (|.|_2 pointwise)
            pwnorm = PointwiseNorm(self.domain, exponent=2)
            denom = pwnorm(diff)
            denom.ufuncs.maximum(lam, out=denom)
            denom /= lam

            # Pointwise division
            for out_i, diff_i in zip(out, diff):
                diff_i.divide(denom, out=out_i)

    return ProximalConvexConjL1L2


def proximal_l1(space, lam=1, g=None):
    r"""Proximal operator factory of the L1 norm/distance.

    Implements the proximal operator of the functional ::

        F(x) = lam ||x - g||_1

    with ``x`` and ``g`` elements in ``space``, and scaling factor ``lam``.

    Parameters
    ----------
    space : `LinearSpace` or `ProductSpace`
        Domain of the functional.
    lam : positive float, optional
        Scaling factor or regularization parameter.
    g : ``space`` element, optional
        Element to which the L1 distance is taken.
        Default: ``space.zero``.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized

    Notes
    -----
    For the functional

    .. math::
        F(x) = \lambda \|x - g\|_1,

    and a step size :math:`\sigma`, the proximal operator of
    :math:`\sigma F` is given as the "soft-shrinkage" operator

    .. math::
        \mathrm{prox}_{\sigma F}(x) =
        \begin{cases}
            g, & \text{where } |x - g| \leq \sigma\lambda, \\
            x - \sigma\lambda \mathrm{sign}(x - g), & \text{elsewhere.}
        \end{cases}

    Here, all operations are to be read pointwise.

    For vector-valued :math:`x` and :math:`g`, the (non-isotropic) proximal
    operator is the component-wise scalar proximal:

    .. math::
        \mathrm{prox}_{\sigma F}(x) = \left(
            \mathrm{prox}_{\sigma F}(x_1), \dots,
            \mathrm{prox}_{\sigma F}(x_d)
            \right),

    where :math:`d` is the number of components of :math:`x`.

    See Also
    --------
    proximal_convex_conj_l1 : proximal for convex conjugate
    proximal_l1_l2 : isotropic variant of the group L1 norm proximal
    """
    lam = float(lam)

    if g is not None and g not in space:
        raise TypeError('{!r} is not an element of {!r}'.format(g, space))

    class ProximalL1(Operator):

        """Proximal operator of the L1 norm/distance."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float or pointwise positive space.element
                Step size parameter. If scalar, it contains a global stepsize,
                otherwise the space.element defines a stepsize for each point.
            """
            super(ProximalL1, self).__init__(
                domain=space, range=space, linear=False)
            if np.isscalar(sigma):
                self.sigma = float(sigma)
            else:
                self.sigma = space.element(sigma)

        def _call(self, x, out):
            """Return ``self(x, out=out)``."""
            # diff = x - g
            if g is not None:
                diff = x - g
            else:
                if x is out:
                    # Handle aliased `x` and `out` (original `x` needed later)
                    diff = x.copy()
                else:
                    diff = x

            # We write the operator as
            # x - (x - g) / max(|x - g| / sig*lam, 1)
            denom = diff.ufuncs.absolute()
            denom /= self.sigma * lam
            denom.ufuncs.maximum(1, out=denom)

            # out = (x - g) / denom
            diff.ufuncs.divide(denom, out=out)

            # out = x - ...
            out.lincomb(1, x, -1, out)

    return ProximalL1


def proximal_l1_l2(space, lam=1, g=None):
    r"""Proximal operator factory of the group-L1-L2 norm/distance.

    Implements the proximal operator of the functional ::

        F(x) = lam || |x - g|_2 ||_1

    with ``x`` and ``g`` elements in ``space``, and scaling factor ``lam``.
    Here, ``|.|_2`` is the pointwise Euclidean norm of a vector-valued
    function.

    Parameters
    ----------
    space : `LinearSpace` or `ProductSpace`
        Domain of the functional.
    lam : positive float, optional
        Scaling factor or regularization parameter.
    g : ``space`` element, optional
        Element to which the L1-L2 distance is taken.
        Default: ``space.zero``.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized

    Notes
    -----
    For the functional

    .. math::
        F(x) = \lambda \| |x - g|_2 \|_1,

    and a step size :math:`\sigma`, the proximal operator of
    :math:`\sigma F` is given as the "soft-shrinkage" operator

    .. math::
        \mathrm{prox}_{\sigma F}(x) =
        \begin{cases}
            g, & \text{where } |x - g|_2 \leq \sigma\lambda, \\
            x - \sigma\lambda \frac{x - g}{|x - g|_2}, & \text{elsewhere.}
        \end{cases}

    Here, all operations are to be read pointwise.

    See Also
    --------
    proximal_l1 : Scalar or non-isotropic vectorial variant
    """
    lam = float(lam)

    if g is not None and g not in space:
        raise TypeError('{!r} is not an element of {!r}'.format(g, space))

    class ProximalL1L2(Operator):

        """Proximal operator of the group-L1-L2 norm/distance."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float
                Step size parameter.
            """
            super(ProximalL1L2, self).__init__(
                domain=space, range=space, linear=False)
            self.sigma = float(sigma)

        def _call(self, x, out):
            """Return ``self(x, out=out)``."""
            # diff = x - g
            if g is not None:
                diff = x - g
            else:
                if x is out:
                    # Handle aliased `x` and `out` (original `x` needed later)
                    diff = x.copy()
                else:
                    diff = x

            # We write the operator as
            # x - (x - g) / max(|x - g|_2 / sig*lam, 1)
            pwnorm = PointwiseNorm(self.domain, exponent=2)
            denom = pwnorm(diff)
            denom /= self.sigma * lam
            denom.ufuncs.maximum(1, out=denom)

            # out = (x - g) / denom
            for out_i, diff_i in zip(out, diff):
                diff_i.divide(denom, out=out_i)

            # out = x - ...
            out.lincomb(1, x, -1, out)

    return ProximalL1L2


def proximal_linfty(space):
    r"""Proximal operator factory of the ``l_\infty``-norm.

    Function for the proximal operator of the functional ``F`` where ``F``
    is the ``l_\infty``-norm::

        ``F(x) =  \sup_i |x_i|``

    Parameters
    ----------
    space : `LinearSpace`
        Domain of ``F``.

    Returns
    -------
    prox_factory : callable
        Factory for the proximal operator to be initialized.

    Notes
    -----
    The proximal is computed by the Moreau identity and a projection onto an
    l1-ball [PB2014].

    See Also
    --------
    proj_l1 : projection onto l1-ball
    """
    class ProximalLInfty(Operator):

        """Proximal operator of the linf-norm."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float
                Step size parameter
            """
            super(ProximalLInfty, self).__init__(
                domain=space, range=space, linear=False)
            self.sigma = float(sigma)

        def _call(self, x, out):
            """Return ``self(x)``."""

            radius = 1

            if x is out:
                x = x.copy()

            proj_l1(x, radius, out)
            out.lincomb(-1, out, 1, x)

    return ProximalLInfty


def proximal_convex_conj_linfty(space):
    r"""Proximal operator factory of the Linfty norm/distance convex conjugate.

    Implements the proximal operator of the convex conjugate of the
    functional ::

        F(x) = \|x\|_\infty

    with ``x`` in ``space``.

    Parameters
    ----------
    space : `LinearSpace` or `ProductSpace` of `LinearSpace` spaces
        Domain of the functional F

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized.

    Notes
    -----
    The convex conjugate :math:`F^*` of the functional

    .. math::
        F(x) = \|x\|_\infty.

    is in the case of scalar-valued functions given by the indicator function
    of the unit 1-norm ball

    .. math::
        F^*(y) = \iota_{B_1} \big( y \big).

    See Also
    --------
    proj_l1 : orthogonal projection onto balls in the 1-norm
    """

    class ProximalConvexConjLinfty(Operator):

        """Proximal operator of the Linfty norm/distance convex conjugate."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float or pointwise positive space.element
                Step size parameter. If scalar, it contains a global stepsize,
                otherwise the space.element defines a stepsize for each point.
            """
            super(ProximalConvexConjLinfty, self).__init__(
                domain=space, range=space, linear=False)

        def _call(self, x, out):
            """Return ``self(x, out=out)``."""
            proj_l1(x, radius=1, out=out)

    return ProximalConvexConjLinfty


def proj_l1(x, radius=1, out=None):
    r"""Projection onto l1-ball.

    Projection onto::

        ``{ x \in X | ||x||_1 \leq r}``

    with ``r`` being the radius.

    Parameters
    ----------
    space : `LinearSpace`
        Space / domain ``X``.
    radius : positive float, optional
        Radius ``r`` of the ball.

    Returns
    -------
    prox_factory : callable
        Factory for the proximal operator to be initialized.

    Notes
    -----
    The projection onto an l1-ball can be computed by projection onto a
    simplex, see [D+2008] for details.

    References
    ----------
    [D+2008] Duchi, J., Shalev-Shwartz, S., Singer, Y., and Chandra, T.
    *Efficient Projections onto the L1-ball for Learning in High dimensions*.
    ICML 2008, pp. 272-279. http://doi.org/10.1145/1390156.1390191

    See Also
    --------
    proximal_linfty : proximal for l-infinity norm
    proj_simplex : projection onto simplex
    """

    if out is None:
        out = x.space.element()

    u = x.ufuncs.absolute()
    v = x.ufuncs.sign()
    proj_simplex(u, radius, out)
    out *= v

    return out


def proj_simplex(x, diameter=1, out=None):
    r"""Projection onto simplex.

    Projection onto::

        ``{ x \in X | x_i \geq 0, \sum_i x_i = r}``

    with :math:`r` being the diameter. It is computed by the formula proposed
    in [D+2008].

    Parameters
    ----------
    space : `LinearSpace`
        Space / domain ``X``.
    diameter : positive float, optional
        Diameter of the simplex.

    Returns
    -------
    prox_factory : callable
        Factory for the proximal operator to be initialized.

    Notes
    -----
    The projection onto a simplex is not of closed-form but can be solved by a
    non-iterative algorithm, see [D+2008] for details.

    References
    ----------
    [D+2008] Duchi, J., Shalev-Shwartz, S., Singer, Y., and Chandra, T.
    *Efficient Projections onto the L1-ball for Learning in High dimensions*.
    ICML 2008, pp. 272-279. http://doi.org/10.1145/1390156.1390191

    See Also
    --------
    proj_l1 : projection onto l1-norm ball
    """
    if out is None:
        out = x.space.element()

    # sort values in descending order
    x_sor = x.asarray().flatten()
    x_sor.sort()
    x_sor = x_sor[::-1]

    # find critical index
    j = np.arange(1, x.size + 1)
    x_avrg = (1 / j) * (np.cumsum(x_sor) - diameter)
    crit = x_sor - x_avrg
    i = np.argwhere(crit >= 0).flatten().max()

    # output is a shifted and thresholded version of the input
    out[:] = np.maximum(x - x_avrg[i], 0)

    return out


def proximal_convex_conj_kl(space, lam=1, g=None):
    r"""Proximal operator factory of the convex conjugate of the KL divergence.

    Function returning the proximal operator of the convex conjugate of the
    functional F where F is the entropy-type Kullback-Leibler (KL) divergence::

        F(x) = sum_i (x_i - g_i + g_i ln(g_i) - g_i ln(pos(x_i))) + ind_P(x)

    with ``x`` and ``g`` elements in the linear space ``X``, and ``g``
    non-negative. Here, ``pos`` denotes the nonnegative part, and ``ind_P`` is
    the indicator function for nonnegativity.

    Parameters
    ----------
    space : `TensorSpace`
        Space X which is the domain of the functional F
    lam : positive float, optional
        Scaling factor.
    g : ``space`` element, optional
        Data term, positive. If None it is take as the one-element.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized.

    See Also
    --------
    proximal_convex_conj_kl_cross_entropy : proximal for releated functional

    Notes
    -----
    The functional is given by the expression

    .. math::
        F(x) = \sum_i (x_i - g_i + g_i \ln(g_i) - g_i \ln(pos(x_i))) +
        I_{x \geq 0}(x)

    The indicator function :math:`I_{x \geq 0}(x)` is used to restrict the
    domain of :math:`F` such that :math:`F` is defined over whole space
    :math:`X`. The non-negativity thresholding :math:`pos` is used to define
    :math:`F` in the real numbers.

    Note that the functional is not well-defined without a prior g. Hence, if g
    is omitted this will be interpreted as if g is equal to the one-element.

    The convex conjugate :math:`F^*` of :math:`F` is

    .. math::
        F^*(p) = \sum_i (-g_i \ln(\text{pos}({1_X}_i - p_i))) +
        I_{1_X - p \geq 0}(p)

    where :math:`p` is the variable dual to :math:`x`, and :math:`1_X` is an
    element of the space :math:`X` with all components set to 1.

    The proximal operator of the convex conjugate of F is

    .. math::
        \mathrm{prox}_{\sigma (\lambda F)^*}(x) =
        \frac{\lambda 1_X + x - \sqrt{(x -  \lambda 1_X)^2 +
        4 \lambda \sigma g}}{2}

    where :math:`\sigma` is the step size-like parameter, and :math:`\lambda`
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
    described in `this Wikipedia article
    <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_, and
    the functional :math:`F` is obtained by switching place of the prior and
    the varialbe in the KL cross entropy functional. See the See Also section.
    """
    lam = float(lam)

    if g is not None and g not in space:
        raise TypeError('{} is not an element of {}'.format(g, space))

    class ProximalConvexConjKL(Operator):

        """Proximal operator of the convex conjugate of the KL divergence."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float
            """
            super(ProximalConvexConjKL, self).__init__(
                domain=space, range=space, linear=False)
            self.sigma = float(sigma)

        def _call(self, x, out):
            """Return ``self(x, out=out)``."""
            # (x + lam - sqrt((x - lam)^2 + 4*lam*sig*g)) / 2

            # out = (x - lam)^2
            if x is out:
                # Handle aliased `x` and `out` (need original `x` later on)
                x = x.copy()
            else:
                out.assign(x)
            out -= lam
            out.ufuncs.square(out=out)

            # out = ... + 4*lam*sigma*g
            # If g is None, it is taken as the one element
            if g is None:
                out += 4.0 * lam * self.sigma
            else:
                out.lincomb(1, out, 4.0 * lam * self.sigma, g)

            # out = x - sqrt(...) + lam
            out.ufuncs.sqrt(out=out)
            out.lincomb(1, x, -1, out)
            out += lam

            # out = 1/2 * ...
            out /= 2

    return ProximalConvexConjKL


def proximal_convex_conj_kl_cross_entropy(space, lam=1, g=None):
    r"""Proximal factory of the convex conj of cross entropy KL divergence.

    Function returning the proximal factory of the convex conjugate of the
    functional F, where F is the cross entropy Kullback-Leibler (KL)
    divergence given by::

        F(x) = sum_i (x_i ln(pos(x_i)) - x_i ln(g_i) + g_i - x_i) + ind_P(x)

    with ``x`` and ``g`` in the linear space ``X``, and ``g`` non-negative.
    Here, ``pos`` denotes the nonnegative part, and ``ind_P`` is the indicator
    function for nonnegativity.

    Parameters
    ----------
    space : `TensorSpace`
        Space X which is the domain of the functional F
    lam : positive float, optional
        Scaling factor.
    g : ``space`` element, optional
        Data term, positive. If None it is take as the one-element.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized.

    See Also
    --------
    proximal_convex_conj_kl : proximal for related functional

    Notes
    -----
    The functional is given by the expression

    .. math::
        F(x) = \sum_i (x_i \ln(pos(x_i)) - x_i \ln(g_i) + g_i - x_i) +
        I_{x \geq 0}(x)

    The indicator function :math:`I_{x \geq 0}(x)` is used to restrict the
    domain of :math:`F` such that :math:`F` is defined over whole space
    :math:`X`. The non-negativity thresholding :math:`pos` is used to define
    :math:`F` in the real numbers.

    Note that the functional is not well-defined without a prior g. Hence, if g
    is omitted this will be interpreted as if g is equal to the one-element.

    The convex conjugate :math:`F^*` of :math:`F` is

    .. math::
        F^*(p) = \sum_i g_i (exp(p_i) - 1)

    where :math:`p` is the variable dual to :math:`x`.

    The proximal operator of the convex conjugate of :math:`F` is

    .. math::
        \mathrm{prox}_{\sigma (\lambda F)^*}(x) = x - \lambda
        W(\frac{\sigma}{\lambda} g e^{x/\lambda})

    where :math:`\sigma` is the step size-like parameter, :math:`\lambda` is
    the weighting in front of the function :math:`F`, and :math:`W` is the
    Lambert W function (see, for example, the
    `Wikipedia article <https://en.wikipedia.org/wiki/Lambert_W_function>`_).

    For real-valued input x, the Lambert :math:`W` function is defined only for
    :math:`x \geq -1/e`, and it has two branches for values
    :math:`-1/e \leq x < 0`. However, for inteneded use-cases, where
    :math:`\lambda` and :math:`g` are positive, the argument of :math:`W`
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

    class ProximalConvexConjKLCrossEntropy(Operator):

        """Proximal operator of conjugate of cross entropy KL divergence."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float
            """
            self.sigma = float(sigma)
            super(ProximalConvexConjKLCrossEntropy, self).__init__(
                domain=space, range=space, linear=False)

        def _call(self, x, out):
            """Return ``self(x, out=out)``."""
            # Lazy import to improve `import odl` time
            import scipy.special

            if g is None:
                # If g is None, it is taken as the one element
                # Different branches of lambertw is not an issue, see Notes
                lambw = scipy.special.lambertw(
                    (self.sigma / lam) * np.exp(x / lam))
            else:
                # Different branches of lambertw is not an issue, see Notes
                lambw = scipy.special.lambertw(
                    (self.sigma / lam) * g * np.exp(x / lam))

            if not np.issubsctype(self.domain.dtype, np.complexfloating):
                lambw = lambw.real

            lambw = x.space.element(lambw)

            out.lincomb(1, x, -lam, lambw)

    return ProximalConvexConjKLCrossEntropy


def proximal_huber(space, gamma):
    """Proximal factory of the Huber norm.

    Parameters
    ----------
    space : `TensorSpace`
        The domain of the functional
    gamma : float
        The smoothing parameter of the Huber norm functional.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator to be initialized.

    See Also
    --------
    odl.solvers.default_functionals.Huber : the Huber norm functional

    Notes
    -----
    The proximal operator is given by given by the proximal operator of
    ``1/(2*gamma) * L2 norm`` in points that are ``<= gamma``, and by the
    proximal operator of the l1 norm in points that are ``> gamma``.
    """

    gamma = float(gamma)

    class ProximalHuber(Operator):

        """Proximal operator of Huber norm."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float
            """
            self.sigma = float(sigma)
            super(ProximalHuber, self).__init__(domain=space, range=space,
                                                linear=False)

        def _call(self, x, out):
            """Return ``self(x, out=out)``."""
            if isinstance(self.domain, ProductSpace):
                norm = PointwiseNorm(self.domain, 2)(x)
            else:
                norm = x.ufuncs.absolute()

            mask = norm.ufuncs.less_equal(gamma + self.sigma)
            out[mask] = gamma / (gamma + self.sigma) * x[mask]

            mask.ufuncs.logical_not(out=mask)
            sign_x = x.ufuncs.sign()
            out[mask] = x[mask] - self.sigma * sign_x[mask]

            return out

    return ProximalHuber


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
