# Copyright 2014-2018 The ODL contributors
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
operator of a variety of functions see `[PB2014]
<https://web.stanford.edu/~boyd/papers/prox_algs.html>`_.

References
----------
[PB2014] Parikh, N, and Boyd, S. *Proximal Algorithms*.
Foundations and Trends in Optimization, 1 (2014), pp 127-239.
"""

from __future__ import absolute_import, division, print_function

import numpy as np

from odl.operator import (
    ConstantOperator, DiagonalOperator, IdentityOperator, MultiplyOperator,
    Operator, PointwiseNorm)
from odl.set.space import LinearSpaceElement
from odl.space import ProductSpace
from odl.util import (
    signature_string_parts, repr_string, npy_printoptions, REPR_PRECISION,
    method_repr_string, array_str)

__all__ = ('proximal_separable_sum', 'proximal_convex_conj',
           'proximal_translation', 'proximal_arg_scaling',
           'proximal_quadratic_perturbation', 'proximal_composition',
           'proximal_const_func', 'proximal_indicator_box',
           'proximal_l1', 'proximal_indicator_linf_unit_ball',
           'proximal_l2', 'proximal_indicator_l2_unit_ball',
           'proximal_l1_l2', 'proximal_convex_conj_l1_l2',
           'proximal_convex_conj_kl', 'proximal_convex_conj_kl_cross_entropy',
           'proximal_huber')


def proximal_separable_sum(*factory_funcs):
    r"""Return the proximal factory for the separable sum of functionals.

    Parameters
    ----------
    factory_func1, ..., factory_funcN : callable
        Proximal operator factories, one for each of the functionals in
        the separable sum.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the separable sum of functionals.

    Notes
    -----
    That two functionals :math:`F` and :math:`G` are separable across variables
    means that :math:`F((x, y)) = F(x)` and :math:`G((x, y)) = G(y)`. In
    this case, the proximal operator of the sum is given by

    .. math::
        \mathrm{prox}_{\sigma (F(x) + G(y))}(x, y) =
        (\mathrm{prox}_{\sigma F}(x), \mathrm{prox}_{\sigma G}(y)).
    """

    def separable_sum_prox_factory(sigma):
        """Proximal factory for a separable sum of functionals ``F_i``.

        Parameters
        ----------
        sigma : positive float or sequence of positive floats
            Step size parameter(s). If a sequence, the length must match
            the length of the ``factory_list``. Furthermore, each of the
            sequence entries can be sequences or arrays, depending on
            what the used proximal factories support.

        Returns
        -------
        diag_op : `DiagonalOperator`
            The operator ``(prox[sigma_1](F_i), ..., prox[sigma_n](F_n))``.
        """
        if np.isscalar(sigma):
            sigma = [sigma] * len(factory_funcs)

        return DiagonalOperator(
            *[factory(sigma_i)
              for sigma_i, factory in zip(sigma, factory_funcs)])

    return separable_sum_prox_factory


def proximal_convex_conj(prox_factory):
    r"""Return the proximal factory for the convex conjugate of a functional.

    Parameters
    ----------
    prox_factory : callable
        A factory function that, when called with a step size, returns a
        proximal operator.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the convex conjugate.

    Notes
    -----
    The Moreau identity states that for any convex function :math:`F` with
    convex conjugate :math:`F^*`, the proximals satisfy

    .. math::
        \mathrm{prox}_{\sigma F^*}(x) + \sigma \,
        \mathrm{prox}_{F / \sigma}(x / \sigma) = x

    where :math:`\sigma` is a scalar step size. Using this, the proximal of
    the convex conjugate is given by

    .. math::
        \mathrm{prox}_{\sigma F^*}(x) =
        x - \sigma \, \mathrm{prox}_{F / \sigma}(x / \sigma)

    Note that since :math:`(F^*)^* = F`, this can be used to get the proximal
    of the original function from the proximal of the convex conjugate.

    For reference see `[CP2011c]
    <https://link.springer.com/chapter/10.1007%2F978-1-4419-9569-8_10>`_.

    References
    ----------
    [CP2011c] Combettes, P L, and Pesquet, J-C. *Proximal splitting
    methods in signal processing.* In:  Bauschke, H H, Burachik, R S,
    Combettes, P L, Elser, V, Luke, D R, and Wolkowicz, H. Fixed-point
    algorithms for inverse problems in science and engineering, Springer,
    2011.
    """
    def convex_conj_prox_factory(sigma):
        """Proximal factory for the convex conjugate of a functional ``F``.

        Parameters
        ----------
        sigma : positive float or array-like
            Step size parameter. Can be a scalar, a pointwise positive space
            element or a sequence of positive floats if the provided
            ``prox_factory`` supports that.

        Returns
        -------
        proximal : `Operator`
            The proximal operator of ``sigma * F^*``.
        """
        # Get the underlying space. At the same time, check if the given
        # prox_factory accepts `sigma` of the given type.
        space = prox_factory(sigma).domain

        mult_right = MultiplyOperator(1 / sigma, domain=space, range=space)
        mult_left = MultiplyOperator(sigma, domain=space, range=space)
        result = (IdentityOperator(space) -
                  mult_left * prox_factory(1 / sigma) * mult_right)
        return result

    return convex_conj_prox_factory


def proximal_translation(prox_factory, y):
    r"""Return the proximal factory for a translated functional.

    The returned `proximal factory` is associated with the translated
    functional ::

        x --> F(x - y)

    given the proximal factory of the original functional ``F``.

    Parameters
    ----------
    prox_factory : callable
        A factory function that, when called with a step size, returns a
        proximal operator.
    y : Element in domain of the functional by which should be translated.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the translated functional.

    Notes
    -----
    Given a functional :math:`F`, this is calculated according to the rule

    .. math::
        \mathrm{prox}_{\sigma F(\cdot - y)}(x) =
        y + \mathrm{prox}_{\sigma F}(x - y)

    where :math:`y` is the translation, and :math:`\sigma` is the step size.

    For reference see `[CP2011c]
    <https://link.springer.com/chapter/10.1007%2F978-1-4419-9569-8_10>`_.

    References
    ----------
    [CP2011c] Combettes, P L, and Pesquet, J-C. *Proximal splitting
    methods in signal processing.* In:  Bauschke, H H, Burachik, R S,
    Combettes, P L, Elser, V, Luke, D R, and Wolkowicz, H. Fixed-point
    algorithms for inverse problems in science and engineering, Springer,
    2011.
    """

    def translation_prox_factory(sigma):
        """Proximal factory for the translated functional.

        Parameters
        ----------
        sigma : positive float or array-like
            Step size parameter. Can be a scalar, a pointwise positive space
            element or a sequence of positive floats if the provided
            ``prox_factory`` supports that.

        Returns
        -------
        proximal : `Operator`
            The proximal operator of ``s * F( . - y)`` where ``s`` is the
            step size.
        """
        return (ConstantOperator(y) + prox_factory(sigma) *
                (IdentityOperator(y.space) - ConstantOperator(y)))

    return translation_prox_factory


def proximal_arg_scaling(prox_factory, scaling):
    r"""Return the proximal factory for a right-scaled functional.

    The returned `proximal factory` is associated with the functional whose
    argument is scaled by a factor, ::

        x --> F(a * x)

    given the proximal factory of the original functional ``F``.

    Parameters
    ----------
    prox_factory : callable
        A factory function that, when called with a step size, returns a
        proximal operator.
    scaling : float or array-like
        Scaling parameter. Can be a scalar, a pointwise positive space
        element or a sequence of positive floats if the provided
        ``prox_factory`` support such types as step size parameters.

        .. note::
            - A scalar 0 is valid, but arrays may not contain zeros since
              they lead to division by 0.
            - Complex factors with nonzero imaginary parts are not supported
              yet. For such scalars, an exception will be raised.
            - For arrays, these conditions are not checked for efficiency
              reasons.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the right-scaled functional.

    Notes
    -----
    Given a functional :math:`F` and a scaling factor :math:`\alpha`,
    the proximal calculated here is

    .. math::
        \mathrm{prox}_{\sigma F(\alpha \, \cdot)}(x) =
        \frac{1}{\alpha}
        \mathrm{prox}_{\sigma \alpha^2 F(\cdot) }(\alpha x),

    where :math:`\sigma` is the step size.

    For reference see `[CP2011c]
    <https://link.springer.com/chapter/10.1007%2F978-1-4419-9569-8_10>`_.

    References
    ----------
    [CP2011c] Combettes, P L, and Pesquet, J-C. *Proximal splitting
    methods in signal processing.* In:  Bauschke, H H, Burachik, R S,
    Combettes, P L, Elser, V, Luke, D R, and Wolkowicz, H. Fixed-point
    algorithms for inverse problems in science and engineering, Springer,
    2011.
    """
    # TODO: Implement the correct proximal for arrays with zero entries.
    # This proximal maps the components where the factor is zero to
    # themselves.

    # TODO(kohr-h): Implement the full complex version of this?
    if np.isscalar(scaling):
        # We run these checks only for scalars, since they can potentially
        # be computationally expensive for arrays.
        if scaling == 0:
            # Special case
            return proximal_const_func(prox_factory(1.0).domain)
        elif scaling.imag != 0:
            raise NotImplementedError('complex scaling not supported.')
        else:
            scaling = float(scaling)
    else:
        scaling = np.asarray(scaling)

    def arg_scaling_prox_factory(sigma):
        """Proximal factory for the right-scaled functional.

        Parameters
        ----------
        sigma : positive float or array-like
            Step size parameter. Can be a scalar, a pointwise positive space
            element or a sequence of positive floats if the provided
            ``prox_factory`` supports that, and if the product
            ``sigma * scaling`` makes sense.

        Returns
        -------
        proximal : `Operator`
            The proximal operator of ``sigma * F( . * a)``.
        """
        scaling_square = scaling * scaling
        prox = prox_factory(sigma * scaling_square)
        space = prox.domain
        mult_inner = MultiplyOperator(scaling, domain=space, range=space)
        mult_outer = MultiplyOperator(1 / scaling, domain=space, range=space)
        return mult_outer * prox * mult_inner

    return arg_scaling_prox_factory


def proximal_quadratic_perturbation(prox_factory, a, u=None):
    r"""Return the proximal factory for a quadratically perturbed functional.

    The returned `proximal factory` is associated with the functional ::

        x --> F(x) + <x, a * x + u>

    given the proximal factory of the original functional ``F``, where
    ``a`` is a scalar and ``u`` a vector.

    Parameters
    ----------
    prox_factory : callable
        A factory function that, when called with a step size, returns a
        proximal operator.
    a : non-negative float
        Coefficient of the quadratic term.
    u : array-like, optional
        Element of the functional domain that defines the linear term.
        The default ``None`` means zero.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the perturbed functional.

    Notes
    -----
    Given a functional :math:`F`, this proximal is calculated according to
    the rule

    .. math::
        \mathrm{prox}_{\sigma \left(F( \cdot ) + a \| \cdot \|^2 +
        <u, \cdot >\right)}(x) =
        c \, \mathrm{prox}_{\sigma F( \cdot \, c)}
        \big((x - \sigma u)\cdot c\big),

    where :math:`c` is the constant

    .. math::
        c = \frac{1}{\sqrt{2 \sigma a + 1}},

    :math:`a` is the scaling parameter belonging to the quadratic term,
    :math:`u` is the space element defining the linear functional, and
    :math:`\sigma` is the step size.

    For reference see `[CP2011c]
    <https://link.springer.com/chapter/10.1007%2F978-1-4419-9569-8_10>`_.

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
        """Proximal factory for the right-scaled functional.

        Parameters
        ----------
        sigma : positive float or array-like
            Step size parameter. Can be a scalar, a pointwise positive space
            element or a sequence of positive floats if the provided
            ``prox_factory`` supports that.

        Returns
        -------
        proximal : `Operator`
            The proximal operator of ``x --> sigma * (F(x) + <x, a * x + u>)``.
        """
        if np.isscalar(sigma):
            sigma = float(sigma)
        else:
            sigma = np.asarray(sigma)

        const = 1.0 / np.sqrt(sigma * 2.0 * a + 1)
        prox = proximal_arg_scaling(prox_factory, const)(sigma)
        if a != 0:
            space = prox.domain
            mult_op = MultiplyOperator(const, domain=space)

        if u is None and a == 0:
            return prox
        elif u is None and a != 0:
            return mult_op * prox * mult_op
        elif u is not None and a == 0:
            return prox - sigma * u
        else:
            return mult_op * prox * (mult_op - (sigma * const) * u)

    return quadratic_perturbation_prox_factory


def proximal_composition(prox_factory, operator, mu):
    r"""Return the proximal factory for a functional composed with an operator.

    The returned `proximal factory` is associated with the functional ::

        x --> F(L x)

    given the proximal factory of the original functional ``F``, where
    ``L`` is an operator.

    .. note::
        The explicit formula for the proximal used by this function only
        holds for operators :math:`L` that satisfy

        .. math::
            L^* L = \mu\, I_X,

        with the identity operator :math:`I_X` on the domain of :math:`L`
        and a positive constant :math:`\mu`.

        This property is not checked; it is up to the user to ensure that
        passed-in operators are valid in this sense.

    Parameters
    ----------
    prox_factory : callable
        A factory function that, when called with a step size, returns a
        proximal operator.
    operator : `Operator`
        The operator to be composed with the functional.
    mu : float
        Scalar such that ``(operator.adjoint * operator)(x) = mu * x``.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the composed functional.

    Notes
    -----
    Given a linear operator :math:`L` with :math:`L^*L x = \mu\, x`, and a
    convex functional :math:`F`, the following identity holds:

    .. math::
        \mathrm{prox}_{\sigma F \circ L}(x) = \frac{1}{\mu}
        L^* \left( \mathrm{prox}_{\mu \sigma F}(Lx) \right)

    There is no simple formula for more general operators.

    For reference see `[CP2011c]
    <https://link.springer.com/chapter/10.1007%2F978-1-4419-9569-8_10>`_.

    References
    ----------
    [CP2011c] Combettes, P L, and Pesquet, J-C. *Proximal splitting
    methods in signal processing.* In:  Bauschke, H H, Burachik, R S,
    Combettes, P L, Elser, V, Luke, D R, and Wolkowicz, H. Fixed-point
    algorithms for inverse problems in science and engineering, Springer,
    2011.
    """
    mu = float(mu)

    def proximal_composition_factory(sigma):
        """Proximal factory for the composed functional.

        Parameters
        ----------
        sigma : positive float or array-like
            Step size parameter. Can be a scalar, a pointwise positive space
            element or a sequence of positive floats if the provided
            ``prox_factory`` supports that.

        Returns
        -------
        proximal : `Operator`
            The proximal operator of ``x --> prox[sigma * F * L](x)``.
        """
        prox_sig_mu = prox_factory(sigma * mu)
        return (1 / mu) * operator.adjoint * prox_sig_mu * operator

    return proximal_composition_factory


def proximal_const_func(space):
    r"""Return the proximal factory for a constant functional.

    The returned `proximal factory` is associated with the functional ::

        x --> const

    It always returns the `IdentityOperator` on the space of ``x``.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of the constant functional.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of a constant functional.
        It always returns the identity operator, independently of its
        input parameter.
    """

    def identity_factory(sigma):
        """Proximal factory for the identity functional.

        Parameters
        ----------
        sigma : positive float or array-like
            Step size parameter (unused but kept to maintain a uniform
            interface).

        Returns
        -------
        proximal : `IdentityOperator`
            The proximal operator of a constant functional.
        """
        return IdentityOperator(space)

    return identity_factory


def proximal_indicator_box(space, lower=None, upper=None):
    r"""Return the proximal factory for a box indicator functional.

    The box indicator function assigns the value ``+inf`` to all points
    outside the box, and ``0`` to points inside. Its proximal operator
    is the projection onto that box.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of the functional.
    lower : float or ``space`` `element-like`, optional
        The (pointwise) lower bound. The default ``None`` means no lower
        bound, i.e., ``-inf``.
    upper : float or ``space`` `element-like`, optional
        The (pointwise) upper bound. The default ``None`` means no upper
        bound, i.e., ``+inf``.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of a box indicator functional.
        It always returns the box projection operator, independently of its
        input parameter.

    Notes
    -----
    The box indicator with lower bound :math:`a` and upper bound :math:`b`
    (can be scalars, vectors or functions) is
    defined as

    .. math::
        \iota_{[a,b]}(x) =
        \begin{cases}
            0      & \text{if } a \leq x \leq b \text{ everywhere}, \\
            \infty & \text{otherwise}.
        \end{cases}

    Its proximal operator is (independently of :math:`\sigma`) given by
    the projection onto the box:

    .. math::
         \mathrm{prox}_{\sigma \iota_{[a,b]}}(x) =
         \begin{cases}
         a & \text{where } x < a, \\
         x & \text{where } a \leq x \leq b, \\
         b & \text{where } x > b.
         \end{cases}
    """
    if lower is not None:
        if np.isscalar(lower):
            lower = float(lower)
        else:
            lower = space.element(lower)
    if upper is not None:
        if np.isscalar(upper):
            upper = float(upper)
        else:
            upper = space.element(upper)

    if np.isscalar(lower) and np.isscalar(upper) and lower > upper:
        raise ValueError('`lower` may not be larger than `upper`, but '
                         '{} > {}'.format(lower, upper))

    class ProximalIndicatorBox(Operator):

        """Proximal operator for a box indicator function."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float or array-like
                Step size parameter (unused but kept to maintain a uniform
                interface).
            """
            super(ProximalIndicatorBox, self).__init__(
                domain=space, range=space, linear=False)
            self.sigma = sigma

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

        def __repr__(self):
            """Return ``repr(self)``.

            Examples
            --------
            >>> space = odl.rn(2)
            >>> lower = 0
            >>> upper = space.one()
            >>> indicator = odl.solvers.IndicatorBox(space, lower, upper)
            >>> indicator.proximal(2)
            IndicatorBox(
                rn(2), lower=0.0, upper=rn(2).element([ 1.,  1.])
            ).proximal(1.0)
            """
            posargs = [space]
            optargs = [('lower', lower, None),
                       ('upper', upper, None)]
            with npy_printoptions(precision=REPR_PRECISION):
                inner_parts = signature_string_parts(posargs, optargs)
            callee_repr = repr_string('IndicatorBox', inner_parts)
            return method_repr_string(callee_repr, 'proximal', ['1.0'])

    return ProximalIndicatorBox


def proximal_l1(space):
    r"""Return the proximal factory for the L1 norm functional.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of the functional.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the L1 norm functional.

    Notes
    -----
    For a step size :math:`\sigma`, the proximal operator of
    :math:`\sigma \|\cdot\|_1` is given by

    .. math::
        \mathrm{prox}_{\sigma \|\cdot\|_1}(x) =
        \max\big\{|x| - \sigma,\, 0\big\}\ \mathrm{sign}(x),

    where all operations are to be read pointwise.

    For vector-valued :math:`\mathbf{x}`, the (non-isotropic) proximal
    operator is the component-wise scalar proximal:

    .. math::
        \mathrm{prox}_{\sigma \|\cdot\|_1}(\mathbf{x}) = \left(
            \mathrm{prox}_{\sigma F}(x_1), \dots,
            \mathrm{prox}_{\sigma F}(x_d)
            \right).

    See Also
    --------
    proximal_convex_conj_l1
    proximal_l1_l2 : isotropic variant of the group L1 norm proximal
    """

    class ProximalL1(Operator):

        """Proximal operator of the L1 norm/distance."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float or array-like
                Step size parameter. A scalar defines a global step size,
                and arrays follow the broadcasting rules.
            """
            super(ProximalL1, self).__init__(
                domain=space, range=space, linear=False)

            if isinstance(space, ProductSpace) and not space.is_power_space:
                dtype = float
            else:
                dtype = space.dtype

            self.sigma = np.asarray(sigma, dtype=dtype)

        def _call(self, x, out):
            """Return ``self(x, out=out)``."""
            # Assign here in case `x` and `out` are aliased
            sign_x = x.ufuncs.sign()

            x.ufuncs.absolute(out=out)
            out -= self.sigma
            out.ufuncs.maximum(0, out=out)
            out *= sign_x

        def __repr__(self):
            """Return ``repr(self)``.

            Examples
            --------
            >>> space = odl.rn(2)
            >>> l1norm = odl.solvers.L1Norm(space)
            >>> l1norm.proximal(2)
            L1Norm(rn(2)).proximal(2.0)
            """
            posargs = [space]
            inner_parts = signature_string_parts(posargs, [])
            callee_repr = repr_string('L1Norm', inner_parts)
            with npy_printoptions(precision=REPR_PRECISION):
                prox_arg_str = array_str(self.sigma)
            return method_repr_string(callee_repr, 'proximal', [prox_arg_str])

    return ProximalL1


def proximal_indicator_linf_unit_ball(space):
    r"""Return the proximal factory for the L^inf unit ball indicator.

    The L^inf unit ball indicator function assigns the value ``+inf`` to all
    points outside the unit ball with respect to the inf-norm, and ``0`` to
    points inside. Its proximal operator is the projection onto that ball.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of the functional.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the ball indicator functional.
        It always returns the unit ball projection operator, independently
        of its input parameter.

    Notes
    -----
    The :math:`L^\infty` unit ball indicator is defined as

    .. math::
        \iota_{B_\infty}(x) =
        \begin{cases}
            0      & \text{if } \|x\|_\infty \leq 1, \\
            \infty & \text{otherwise}.
        \end{cases}

    Its proximal operator is (independently of :math:`\sigma`) given by
    the projection onto the ball:

    .. math::
        \mathrm{prox}_{\sigma \iota_{B_\infty}}(x) =
        \mathrm{sign}(x)\, \min\{|x|,\, 1\},

    where all operations are to be understood pointwise.

    For vector-valued functions, since the :math:`\infty`-norm is separable
    across components, the proximal is given as

    .. math::
        \mathrm{prox}_{\sigma \iota_{B_\infty}}(\mathbf{x}) = \left(
            \mathrm{prox}_{\sigma \iota_{B_\infty}}(x_1), \dots,
            \mathrm{prox}_{\sigma \iota_{B_\infty}}(x_d)
            \right).

    See Also
    --------
    proximal_l1 : proximal of the convex conjugate
    proximal_convex_conj_l1_l2 :
        proximal of the isotropic variant for vector-valued functions
    """

    class ProximalIndicatorLinfUnitBall(Operator):

        """Proximal operator for the L^inf unit ball indicator."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float or array-like
                Step size parameter (unused but kept to maintain a uniform
                interface).
            """
            super(ProximalIndicatorLinfUnitBall, self).__init__(
                domain=space, range=space, linear=False)

        def _call(self, x, out):
            """Return ``self(x, out=out)``."""
            # Take abs first due to possible aliasing of `x` and `out`
            abs_x = x.ufuncs.absolute()
            abs_x.ufuncs.minimum(1, out=abs_x)
            x.ufuncs.sign(out=out)
            out *= abs_x
            return out

        def __repr__(self):
            """Return ``repr(self)``.

            Examples
            --------
            >>> space = odl.rn(2)
            >>> l1norm = odl.solvers.L1Norm(space)
            >>> l1norm.convex_conj.proximal(2)
            IndicatorLpUnitBall(rn(2), exponent='inf').proximal(1.0)
            """
            posargs = [space]
            optargs = [('exponent', float('inf'), None)]
            inner_parts = signature_string_parts(posargs, optargs)
            callee_repr = repr_string('IndicatorLpUnitBall', inner_parts)
            return method_repr_string(callee_repr, 'proximal', ['1.0'])

    return ProximalIndicatorLinfUnitBall


def proximal_l2(space):
    r"""Return the proximal factory for the L2 norm functional.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of the functional.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the L2 norm functional.

    Notes
    -----
    For a step size :math:`\sigma`, the proximal operator of
    :math:`\sigma \|\cdot\|_2` is given by

    .. math::
        \mathrm{prox}_{\sigma \|\cdot\|_2}(y) =
        \max\left\{1 - \frac{\sigma}{\|y\|_2},\ 0\right\}\ y.

    See Also
    --------
    proximal_l2_squared : proximal for squared norm/distance
    proximal_indicator_l2_unit_ball : proximal of the convex conjugate
    """

    class ProximalL2(Operator):

        """Proximal operator of the L2 norm."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float
                Step size parameter.
            """
            super(ProximalL2, self).__init__(
                domain=space, range=space, linear=False)
            self.sigma = float(sigma)

        def _call(self, x, out):
            """Implement ``self(x, out)``."""
            dtype = getattr(self.domain, 'dtype', float)
            eps = np.finfo(dtype).resolution * 10

            x_norm = x.norm() * (1 + eps)
            if x_norm == 0:
                out.set_zero()
            else:
                out.lincomb(1 - self.sigma / x_norm, x)

        def __repr__(self):
            """Return ``repr(self)``.

            Examples
            --------
            >>> space = odl.rn(2)
            >>> l2norm = odl.solvers.L2Norm(space)
            >>> l2norm.proximal(2)
            L2Norm(rn(2)).proximal(2.0)
            """
            posargs = [space]
            inner_parts = signature_string_parts(posargs, [])
            callee_repr = repr_string('L2Norm', inner_parts)
            with npy_printoptions(precision=REPR_PRECISION):
                prox_arg_str = array_str(self.sigma)
            return method_repr_string(callee_repr, 'proximal', [prox_arg_str])

    return ProximalL2


def proximal_indicator_l2_unit_ball(space):
    r"""Return the proximal factory for the L2 unit ball indicator functional.

    The L2 unit ball indicator function assigns the value ``+inf`` to all
    points outside the unit ball, and ``0`` to points inside. Its proximal
    operator is the projection onto that ball.

    Parameters
    ----------
    space : `LinearSpace`
        Domain of the functional.

    Returns
    -------
    prox_factory : function
        Factory for the proximal operator of the ball indicator functional.
        It always returns the unit ball projection operator, independently
        of its input parameter.

    Notes
    -----
    The :math:`L^2` unit ball indicator is defined as

    .. math::
        \iota_{B_2}(x) =
        \begin{cases}
            0      & \text{if } \|x\|_2 \leq 1, \\
            \infty & \text{otherwise}.
        \end{cases}

    Its proximal operator is (independently of :math:`\sigma`) given by
    the projection onto the ball:

    .. math::
         \mathrm{prox}_{\sigma \iota_{B_2}}(x) =
         \begin{cases}
         \frac{x}{\|x\|_2} & \text{if } \|x\|_2 > 1, \\
         \ x & \text{otherwise.}
         \end{cases}

    See Also
    --------
    proximal_l2
    proximal_convex_conj_l2_squared
    """

    class ProximalIndicatorL2UnitBall(Operator):

        """Proximal operator for the L2 unit ball indicator."""

        def __init__(self, sigma):
            """Initialize a new instance.

            Parameters
            ----------
            sigma : positive float or array-like
                Step size parameter (unused but kept to maintain a uniform
                interface).
            """
            super(ProximalIndicatorL2UnitBall, self).__init__(
                domain=space, range=space, linear=False)

        def _call(self, x, out):
            """Implement ``self(x, out)``."""
            dtype = getattr(self.domain, 'dtype', float)
            eps = np.finfo(dtype).resolution * 10

            x_norm = x.norm() * (1 + eps)
            if x_norm > 1:
                out.lincomb(1 / x_norm, x)
            else:
                out.assign(x)

        def __repr__(self):
            """Return ``repr(self)``.

            Examples
            --------
            >>> space = odl.rn(2)
            >>> l2norm = odl.solvers.L2Norm(space)
            >>> l2norm.convex_conj.proximal(2)
            IndicatorLpUnitBall(rn(2), exponent=2.0).proximal(1.0)
            """
            posargs = [space]
            optargs = [('exponent', 2.0, None)]
            with npy_printoptions(precision=REPR_PRECISION):
                inner_parts = signature_string_parts(posargs, optargs)
            callee_repr = repr_string('IndicatorLpUnitBall', inner_parts)
            return method_repr_string(callee_repr, 'proximal', ['1.0'])

    return ProximalIndicatorL2UnitBall


#TODO: continue here

def proximal_l1_l2(space, lam=1, g=None):
    """Proximal operator factory of the group-L1-L2 norm/distance.

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
        F(x) = \\lambda \| |x - g|_2 \|_1,

    and a step size :math:`\\sigma`, the proximal operator of
    :math:`\\sigma F` is given as the "soft-shrinkage" operator

    .. math::
        \mathrm{prox}_{\\sigma F}(x) =
        \\begin{cases}
            g, & \\text{where } |x - g|_2 \\leq \sigma\\lambda, \\\\
            x - \sigma\\lambda \\frac{x - g}{|x - g|_2}, & \\text{elsewhere.}
        \\end{cases}

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


def proximal_convex_conj_l1_l2(space, lam=1, g=None):
    """Proximal operator factory of the L1-L2 norm/distance convex conjugate.

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
        F(x) = \\lambda \| |x - g|_2 \|_1.

    is given by

    .. math::
        F^*(y) = \iota_{B_\infty} \\big( \\lambda^{-1}\, |y|_2 \\big) +
        \\left\\langle \\lambda^{-1}\, y,\: g \\right\\rangle,

    where :math:`\iota_{B_\infty}` is the indicator function of the
    unit ball with respect to :math:`\|\cdot\|_\infty`.

    For a step size :math:`\\sigma`, the proximal operator of
    :math:`\\sigma F^*` is given by

    .. math::
        \mathrm{prox}_{\\sigma F^*}(y) = \\frac{\\lambda (y - \\sigma g)}{
        \\max(\\lambda, |y - \\sigma g|_2)}

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


# TODO(kohr-h): implement KL prox


def proximal_convex_conj_kl(space, lam=1, g=None):
    """Proximal operator factory of the convex conjugate of the KL divergence.

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
        F^*(p) = \\sum_i (-g_i \\ln(\\text{pos}({1_X}_i - p_i))) +
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
    """Proximal factory of the convex conjugate of cross entropy KL divergence.

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
    odl.solvers.Huber : the Huber norm functional

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

            idx = norm.ufuncs.less_equal(gamma + self.sigma)
            out[idx] = gamma / (gamma + self.sigma) * x[idx]

            idx.ufuncs.logical_not(out=idx)
            sign_x = x.ufuncs.sign()
            out[idx] = x[idx] - self.sigma * sign_x[idx]

            return out

    return ProximalHuber


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
