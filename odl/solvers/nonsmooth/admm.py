"""Alternating Direction method of Multipliers (ADMM) method variants."""

from __future__ import division
from odl.operator import Operator, OpDomainError, power_method_opnorm


__all__ = ('admm_linearized', 'admm_precon_nonlinear')


def admm_linearized(x, f, g, L, tau, sigma, niter, **kwargs):
    """Generic linearized ADMM method for convex problems.

    ADMM stands for "Alternating Direction Method of Multipliers" and
    is a popular convex optimization method. This variant solves problems
    of the form ::

        min_x [ f(x) + g(Lx) ]

    with convex ``f`` and ``g``, and a linear operator ``L``. See Section
    4.4 of `[PB2014] <http://web.stanford.edu/~boyd/papers/prox_algs.html>`_
    and the Notes for more mathematical details.

    Parameters
    ----------
    x : ``L.domain`` element
        Starting point of the iteration, updated in-place.
    f, g : `Functional`
        The functions ``f`` and ``g`` in the problem definition. They
        need to implement the ``proximal`` method.
    L : linear `Operator`
        Linear operator composed with ``g`` in the problem definition.
        It must implement ``L.adjoint`` and fulfill ``L.domain == f.domain``
        and ``L.range == g.domain``.
    tau, sigma : positive float
        Step size parameters for the update of the variables.
    niter : non-negative int
        Number of iterations.

    Other Parameters
    ----------------
    callback : callable, optional
        Function called with the current iterate after each iteration.

    Notes
    -----
    Given :math:`x^{(0)}` (the provided ``x``) and
    :math:`u^{(0)} = z^{(0)} = 0`, linearized ADMM applies the following
    iteration:

    .. math::
        x^{(k+1)} &= \mathrm{prox}_{\\tau f} \\left[
            x^{(k)} - \sigma^{-1}\\tau L^*\\big(
                L x^{(k)} - z^{(k)} + u^{(k)}
            \\big)
        \\right]

        z^{(k+1)} &= \mathrm{prox}_{\sigma g}\\left(
            L x^{(k+1)} + u^{(k)}
        \\right)

        u^{(k+1)} &= u^{(k)} + L x^{(k+1)} - z^{(k+1)}

    The step size parameters :math:`\\tau` and :math:`\sigma` must satisfy

    .. math::
        0 < \\tau < \\frac{\sigma}{\|L\|^2}

    to guarantee convergence.

    The name "linearized ADMM" comes from the fact that in the
    minimization subproblem for the :math:`x` variable, this variant
    uses a linearization of a quadratic term in the augmented Lagrangian
    of the generic ADMM, in order to make the step expressible with
    the proximal operator of :math:`f`. See
    `[PB2014] <http://web.stanford.edu/~boyd/papers/prox_algs.html>`_
    for details.

    Another name for this algorithm is *split inexact Uzawa method*.

    References
    ----------
    [PB2014] Parikh, N and Boyd, S. *Proximal Algorithms*. Foundations and
    Trends in Optimization, 1(3) (2014), pp 123-231.
    """
    if not isinstance(L, Operator):
        raise TypeError('`op` {!r} is not an `Operator` instance'
                        ''.format(L))

    if x not in L.domain:
        raise OpDomainError('`x` {!r} is not in the domain of `op` {!r}'
                            ''.format(x, L.domain))

    tau, tau_in = float(tau), tau
    if tau <= 0:
        raise ValueError('`tau` must be positive, got {}'.format(tau_in))

    sigma, sigma_in = float(sigma), sigma
    if sigma <= 0:
        raise ValueError('`sigma` must be positive, got {}'.format(sigma_in))

    niter, niter_in = int(niter), niter
    if niter < 0 or niter != niter_in:
        raise ValueError('`niter` must be a non-negative integer, got {}'
                         ''.format(niter_in))

    # Callback object
    callback = kwargs.pop('callback', None)
    if callback is not None and not callable(callback):
        raise TypeError('`callback` {} is not callable'.format(callback))

    # Initialize range variables
    z = L.range.zero()
    u = L.range.zero()

    # Temporary for Lx + u [- z]
    tmp_ran = L(x)
    # Temporary for L^*(Lx + u - z)
    tmp_dom = L.domain.element()

    # Store proximals since their initialization may involve computation
    prox_tau_f = f.proximal(tau)
    prox_sigma_g = g.proximal(sigma)

    for _ in range(niter):
        # tmp_ran has value Lx^k here
        # tmp_dom <- L^*(Lx^k + u^k - z^k)
        tmp_ran += u
        tmp_ran -= z
        L.adjoint(tmp_ran, out=tmp_dom)

        # x <- x^k - (tau/sigma) L^*(Lx^k + u^k - z^k)
        x.lincomb(1, x, -tau / sigma, tmp_dom)
        # x^(k+1) <- prox[tau*f](x)
        prox_tau_f(x, out=x)

        # tmp_ran <- Lx^(k+1)
        L(x, out=tmp_ran)
        # z^(k+1) <- prox[sigma*g](Lx^(k+1) + u^k)
        prox_sigma_g(tmp_ran + u, out=z)  # 1 copy here

        # u^(k+1) = u^k + Lx^(k+1) - z^(k+1)
        u += tmp_ran
        u -= z

        if callback is not None:
            callback(x)


def admm_linearized_simple(x, f, g, L, tau, sigma, niter, **kwargs):
    """Non-optimized version of ``admm_linearized``.

    This function is intended for debugging. It makes a lot of copies and
    performs no error checking.
    """
    callback = kwargs.pop('callback', None)
    z = L.range.zero()
    u = L.range.zero()
    for _ in range(niter):
        x[:] = f.proximal(tau)(x - tau / sigma * L.adjoint(L(x) + u - z))
        z = g.proximal(sigma)(L(x) + u)
        u = L(x) + u - z
        if callback is not None:
            callback(x)


def admm_precon_nonlinear(x, f, g, L, delta, niter, sigma=None, **kwargs):
    """Preconditioned nonlinear ADMM.

    ADMM stands for "Alternating Direction Method of Multipliers" and
    is a popular convex optimization method. This variant solves problems
    of the form ::

        min_x [ f(x) + g(L(x)) ]

    with convex ``f`` and ``g``, and a (possibly nonlinear) operator ``L``.
    See `[BKS+2015] <https://doi.org/10.1007/978-3-319-55795-3_10>`_
    and the Notes for more mathematical details.

    Parameters
    ----------
    x : ``L.domain`` element
        Starting point, updated in-place.
    f, g : `Functional`
        The functions ``f`` and ``g`` in the problem definition. They
        need to implement the ``proximal`` method and satisfy
        ``f.domain == L.domain`` and ``g.domain == L.range``,
        respectively.
    L : `Operator`
        Possibly nonlinear operator composed with ``g`` in the problem
        definition. It must implement ``L.derivative(x).adjoint`` and
        fulfill ``L.domain == f.domain`` and ``L.range == g.domain``.
    delta : positive float
        Step size parameter for the update of the constraint variable.
    sigma : positive float, optional
        Step size parameter for ``g.proximal``.
        Default: ``0.5 / delta``.
    niter : non-negative int
        Number of iterations.

    Other Parameters
    ----------------
    opnorm_maxiter : nonnegative int
        Maximum number of iterations to be used for the operator norm
        estimation in each step of the optimization loop.
        Default: 2
    opnorm_factor : float between 0 and 1
        Multiply this factor with the upper bound for the ``tau`` step
        size determination, see Notes. Smaller values can be used to
        compensate for a bad operator norm estimate caused by a small
        ``opnorm_maxiter``.
        Default: 0.1
    callback : callable, optional
        Function called with the current iterate after each iteration.

    Notes
    -----
    The preconditioned nonlinear ADMM solves the problem

    .. math::
        \min_{x} \\left[ f(x) + g\\big(L(x)\\big) \\right].

    It starts with initial values :math:`x^{(0)}` as given,
    :math:`y^{(0)} = 0` and :math:`\mu^{(0)} = \overline{\mu}^{(0)} = 0`,
    and applies the following iteration:

    .. math::
        A^{(k)} &= \partial L(x^{(k)}),

        \\text{choose } \\tau^{(k)} &< \\frac{1}{\delta \|A^{(k)}\|^2}

        x^{(k+1)} &= \mathrm{prox}_{\\tau^{(k)} f} \\left[
            x^{(k)} - \\tau^{(k)} \\big(A^{(k)}\\big)^*\, \overline{\mu}^{(k)}
        \\right]

        y^{(k+1)} &= \mathrm{prox}_{\sigma g}\\left[
            y^{(k)} + \sigma \\Big(
                \mu^{(k)} + \delta \\big(L(x^{(k+1)} - y^{(k)}\\big)
            \\Big)
        \\right]

        \mu^{(k+1)} &= \mu^{(k)} + \delta \\big(L(x^{(k+1)}) - y^{(k+1)}\\big)

        \overline{\mu}^{(k+1)} &= 2\mu^{(k+1)} - \mu^{(k)}

    For (local) convergence it is required that :math:`\sigma < 1 / \delta`.

    References
    ----------
    [BKS+2015] Benning, M, Knoll, F, Schönlieb, C-B., and Valkonen, T.
    *Preconditioned ADMM with Nonlinear Operator Constraint*.
    System Modeling and Optimization, 2015, pp. 117–126.
    """
    if not isinstance(L, Operator):
        raise TypeError('`op` {!r} is not an `Operator` instance'
                        ''.format(L))

    if x not in L.domain:
        raise OpDomainError('`x` {!r} is not in the domain of `op` {!r}'
                            ''.format(x, L.domain))

    delta, delta_in = float(delta), delta
    if delta <= 0:
        raise ValueError('`delta` must be positive, got {}'.format(delta_in))

    niter, niter_in = int(niter), niter
    if niter < 0 or niter != niter_in:
        raise ValueError('`niter` must be a non-negative integer, got {}'
                         ''.format(niter_in))

    if sigma is None:
        sigma = 0.5 / delta
    else:
        sigma, sigma_in = float(sigma), sigma
        if not 0 < sigma < 1 / delta:
            raise ValueError(
                '`sigma` must lie strictly between 0 and 1/delta`, got '
                'sigma={:.4} and (1/delta)={:.4}'.format(sigma_in, 1 / delta))

    opnorm_maxiter = kwargs.pop('opnorm_maxiter', 2)
    opnorm_factor = kwargs.pop('opnorm_factor', 0.1)
    opnorm_factor, opnorm_factor_in = float(opnorm_factor), opnorm_factor
    if not 0 < opnorm_factor < 1:
        raise ValueError('`opnorm_factor` must lie strictly between 0 and 1, '
                         'got {}'.format(opnorm_factor_in))

    callback = kwargs.pop('callback', None)
    if callback is not None and not callable(callback):
        raise TypeError('`callback` {} is not callable'.format(callback))

    # Initialize range variables (quantities in [] are zero initially)
    y = L.range.zero()
    # mu = [mu_old +] delta * (L(x) [- y])
    mu = delta * L(x)
    # mubar = 2 * mu [- mu_old]
    mubar = 2 * mu

    # Temporary for Lx
    tmp_ran = L.range.element()
    # Temporary for L^*(mubar)
    tmp_dom = L.domain.element()

    # Store constant g proximal since it may involve computation
    g_prox_sigma = g.proximal(sigma)

    for i in range(niter):
        # tau^k <- opnorm_factor / (delta * ||dL(x^k)||^2)
        A = L.derivative(x)
        xstart = A.domain.one() if i == 0 else x
        A_norm = power_method_opnorm(A, xstart, maxiter=opnorm_maxiter)
        tau = opnorm_factor / (delta * A_norm ** 2)

        # x^(k+1) <- prox[tau^k*f](x^k - tau^k * A^*(mubar^k))
        A.adjoint(mubar, out=tmp_dom)
        x.lincomb(1, x, -tau, tmp_dom)
        f.proximal(tau)(x, out=x)

        # y^(k+1) <- prox[sigma*g]((1 - sigma * delta) * y^k +
        #                          sigma * (mu^k + delta * L(x^(k+1))))
        L(x, out=tmp_ran)
        y.lincomb(1 - sigma * delta, y, sigma * delta, tmp_ran)
        y.lincomb(1, y, sigma, mu)
        g_prox_sigma(y, out=y)

        # mu^(k+1) <- mu^k + delta * (L(x^(k+1)) - y^(k+1))
        # tmp_ran still holds L(x^(k+1))
        # Using mubar as temporary
        mubar.assign(mu)
        mu.lincomb(1, mu, delta, tmp_ran)
        mu.lincomb(1, mu, -delta, y)

        # mubar^(k+1) = 2 * mu^(k+1) - mu^k
        mubar.lincomb(2, mu, -1, mubar)

        if callback is not None:
            callback(x)


def admm_precon_nonlinear_simple(x, f, g, L, delta, niter, sigma=None,
                                 **kwargs):
    """Non-optimized version of ``admm_precon_nonlinear``.

    This function is intended for debugging. It makes a lot of copies and
    performs no error checking.
    """
    if sigma is None:
        sigma = 0.5 / delta

    opnorm_maxiter = kwargs.pop('opnorm_maxiter', 2)
    opnorm_factor = kwargs.pop('opnorm_factor', 0.1)
    callback = kwargs.pop('callback', None)

    # Initialize range variables (quantities in [] are zero initially)
    y = L.range.zero()
    # mu = [mu_old +] delta * (L(x) [- y])
    mu = delta * L(x)
    # mubar = 2 * mu [- mu_old]
    mubar = 2 * mu

    for i in range(niter):
        A = L.derivative(x)
        xstart = A.domain.one() if i == 0 else x
        A_norm = power_method_opnorm(A, xstart, maxiter=opnorm_maxiter)
        tau = opnorm_factor / (delta * A_norm ** 2)
        x[:] = f.proximal(tau)(x - tau * A.adjoint(mubar))
        y = g.proximal(sigma)((1 - sigma * delta) * y +
                              sigma * (mu + delta * L(x)))
        mu_old = mu
        mu = mu + delta * (L(x) - y)
        mubar = 2 * mu - mu_old

        if callback is not None:
            callback(x)
