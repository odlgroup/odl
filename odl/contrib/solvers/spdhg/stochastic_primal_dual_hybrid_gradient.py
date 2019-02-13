# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Stochastic Primal-Dual Hybrid Gradient (SPDHG) algorithms"""

from __future__ import print_function, division
import numpy as np
import odl

__all__ = ('pdhg', 'spdhg', 'pa_spdhg', 'spdhg_generic', 'da_spdhg',
           'spdhg_pesquet')


def pdhg(x, f, g, A, tau, sigma, niter, **kwargs):
    """Computes a saddle point with PDHG.

    This algorithm is the same as "algorithm 1" in [CP2011a] but with
    extrapolation on the dual variable.


    Parameters
    ----------
    x : primal variable
        This variable is both input and output of the method.
    f : function
        Functional Y -> IR_infty that has a convex conjugate with a
        proximal operator, i.e. f.convex_conj.proximal(sigma) : Y -> Y.
    g : function
        Functional X -> IR_infty that has a proximal operator, i.e.
        g.proximal(tau) : X -> X.
    A : function
        Operator A : X -> Y that possesses an adjoint: A.adjoint
    tau : scalar / vector / matrix
        Step size for primal variable. Note that the proximal operator of g
        has to be well-defined for this input.
    sigma : scalar
        Scalar / vector / matrix used as step size for dual variable. Note that
        the proximal operator related to f (see above) has to be well-defined
        for this input.
    niter : int
        Number of iterations

    Other Parameters
    ----------------
    y: dual variable
        Dual variable is part of a product space
    z: variable
        Adjoint of dual variable, z = A^* y.
    theta : scalar
        Extrapolation factor.
    callback : callable
        Function called with the current iterate after each iteration.

    References
    ----------
    [CP2011a] Chambolle, A and Pock, T. *A First-Order
    Primal-Dual Algorithm for Convex Problems with Applications to
    Imaging*. Journal of Mathematical Imaging and Vision, 40 (2011),
    pp 120-145.
    """

    def fun_select(k):
        return [0]

    f = odl.solvers.SeparableSum(f)
    A = odl.BroadcastOperator(A, 1)

    # Dual variable
    y = kwargs.pop('y', None)
    if y is None:
        y_new = None
    else:
        y_new = A.range.element([y])

    spdhg_generic(x, f, g, A, tau, [sigma], niter, fun_select, y=y_new,
                  **kwargs)

    if y is not None:
        y.assign(y_new[0])


def spdhg(x, f, g, A, tau, sigma, niter, **kwargs):
    r"""Computes a saddle point with a stochastic PDHG.

    This means, a solution (x*, y*), y* = (y*_1, ..., y*_n) such that

    (x*, y*) in arg min_x max_y sum_i=1^n <y_i, A_i> - f*[i](y_i) + g(x)

    where g : X -> IR_infty and f[i] : Y[i] -> IR_infty are convex, l.s.c. and
    proper functionals. For this algorithm, they all may be non-smooth and no
    strong convexity is assumed.

    Parameters
    ----------
    x : primal variable
        This variable is both input and output of the method.
    f : functions
        Functionals Y[i] -> IR_infty that all have a convex conjugate with a
        proximal operator, i.e.
        f[i].convex_conj.proximal(sigma[i]) : Y[i] -> Y[i].
    g : function
        Functional X -> IR_infty that has a proximal operator, i.e.
        g.proximal(tau) : X -> X.
    A : functions
        Operators A[i] : X -> Y[i] that possess adjoints: A[i].adjoint
    tau : scalar / vector / matrix
        Step size for primal variable. Note that the proximal operator of g
        has to be well-defined for this input.
    sigma : scalar
        Scalar / vector / matrix used as step size for dual variable. Note that
        the proximal operator related to f (see above) has to be well-defined
        for this input.
    niter : int
        Number of iterations

    Other Parameters
    ----------------
    y : dual variable
        Dual variable is part of a product space. By default equals 0.
    z : variable
        Adjoint of dual variable, z = A^* y. By default equals 0 if y = 0.
    theta : scalar
        Global extrapolation factor.
    prob: list
        List of probabilities that an index i is selected each iteration. By
        default this is uniform serial sampling, p_i = 1/n.
    fun_select : function
        Function that selects blocks at every iteration IN -> {1,...,n}. By
        default this is serial sampling, fun_select(k) selects an index
        i \in {1,...,n} with probability p_i.
    callback : callable
        Function called with the current iterate after each iteration.

    References
    ----------
    [CERS2017] A. Chambolle, M. J. Ehrhardt, P. Richtarik and C.-B. Schoenlieb,
    *Stochastic Primal-Dual Hybrid Gradient Algorithm with Arbitrary Sampling
    and Imaging Applications*. ArXiv: http://arxiv.org/abs/1706.04957 (2017).

    [E+2017] M. J. Ehrhardt, P. J. Markiewicz, P. Richtarik, J. Schott,
    A. Chambolle and C.-B. Schoenlieb, *Faster PET reconstruction with a
    stochastic primal-dual hybrid gradient method*. Wavelets and Sparsity XVII,
    58 (2017) http://doi.org/10.1117/12.2272946.
    """

    # Probabilities
    prob = kwargs.pop('prob', None)
    if prob is None:
        prob = [1 / len(A)] * len(A)

    # Selection function
    fun_select = kwargs.pop('fun_select', None)
    if fun_select is None:
        def fun_select(x):
            return [int(np.random.choice(len(A), 1, p=prob))]

    # Dual variable
    y = kwargs.pop('y', None)

    extra = [1 / p for p in prob]

    spdhg_generic(x, f, g, A, tau, sigma, niter, fun_select=fun_select, y=y,
                  extra=extra, **kwargs)


def pa_spdhg(x, f, g, A, tau, sigma, niter, mu_g, **kwargs):
    r"""Computes a saddle point with a stochastic PDHG and primal acceleration.

    Next to other standard arguments, this algorithm requires the strong
    convexity constant mu_g of g.

    Parameters
    ----------
    x : primal variable
        This variable is both input and output of the method.
    f : functions
        Functionals Y[i] -> IR_infty that all have a convex conjugate with a
        proximal operator, i.e.
        f[i].convex_conj.proximal(sigma[i]) : Y[i] -> Y[i].
    g : function
        Functional X -> IR_infty that has a proximal operator, i.e.
        g.proximal(tau) : X -> X.
    A : functions
        Operators A[i] : X -> Y[i] that possess adjoints: A[i].adjoint
    tau : scalar
        Step size for primal variable.
    sigma : scalar
        Step size for dual variable.
    niter : int
        Number of iterations
    mu_g : scalar
        Strong convexity constant of g.

    Other Parameters
    ----------------
    y : dual variable
        Dual variable is part of a product space. By default equals 0.
    z : variable
        Adjoint of dual variable, z = A^* y. By default equals 0 if y = 0.
    prob: list
        List of probabilities that an index i is selected each iteration. By
        default this is uniform serial sampling, p_i = 1/n.
    fun_select : function
        Function that selects blocks at every iteration IN -> {1,...,n}. By
        default this is serial sampling, fun_select(k) selects an index
        i \in {1,...,n} with probability p_i.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    References
    ----------
    [CERS2017] A. Chambolle, M. J. Ehrhardt, P. Richtarik and C.-B. Schoenlieb,
    *Stochastic Primal-Dual Hybrid Gradient Algorithm with Arbitrary Sampling
    and Imaging Applications*. ArXiv: http://arxiv.org/abs/1706.04957 (2017).
    """
    # Probabilities
    prob = kwargs.pop('prob', None)
    if prob is None:
        prob = [1 / len(A)] * len(A)

    # Selection function
    fun_select = kwargs.pop('fun_select', None)
    if fun_select is None:
        def fun_select(x):
            return [int(np.random.choice(len(A), 1, p=prob))]

    # Dual variable
    y = kwargs.pop('y', None)

    extra = [1 / p for p in prob]

    spdhg_generic(x, f, g, A, tau, sigma, niter, fun_select=fun_select,
                  extra=extra, mu_g=mu_g, y=y, **kwargs)


def spdhg_generic(x, f, g, A, tau, sigma, niter, **kwargs):
    r"""Computes a saddle point with a stochastic PDHG.

    This means, a solution (x*, y*), y* = (y*_1, ..., y*_n) such that

    (x*, y*) in arg min_x max_y sum_i=1^n <y_i, A_i> - f*[i](y_i) + g(x)

    where g : X -> IR_infty and f[i] : Y[i] -> IR_infty are convex, l.s.c. and
    proper functionals. For this algorithm, they all may be non-smooth and no
    strong convexity is assumed.

    Parameters
    ----------
    x : primal variable
        This variable is both input and output of the method.
    f : functions
        Functionals Y[i] -> IR_infty that all have a convex conjugate with a
        proximal operator, i.e.
        f[i].convex_conj.proximal(sigma[i]) : Y[i] -> Y[i].
    g : function
        Functional X -> IR_infty that has a proximal operator, i.e.
        g.proximal(tau) : X -> X.
    A : functions
        Operators A[i] : X -> Y[i] that possess adjoints: A[i].adjoint
    tau : scalar / vector / matrix
        Step size for primal variable. Note that the proximal operator of g
        has to be well-defined for this input.
    sigma : scalar
        Scalar / vector / matrix used as step size for dual variable. Note that
        the proximal operator related to f (see above) has to be well-defined
        for this input.
    niter : int
        Number of iterations

    Other Parameters
    ----------------
    y : dual variable, optional
        Dual variable is part of a product space. By default equals 0.
    z : variable, optional
        Adjoint of dual variable, z = A^* y. By default equals 0 if y = 0.
    mu_g : scalar
        Strong convexity constant of g.
    theta : scalar
        Global extrapolation factor.
    extra: list
        List of local extrapolation paramters for every index i. By default
        extra_i = 1.
    fun_select : function
        Function that selects blocks at every iteration IN -> {1,...,n}. By
        default this is serial uniform sampling, fun_select(k) selects an index
        i \in {1,...,n} with probability 1/n.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    References
    ----------
    [CERS2017] A. Chambolle, M. J. Ehrhardt, P. Richtarik and C.-B. Schoenlieb,
    *Stochastic Primal-Dual Hybrid Gradient Algorithm with Arbitrary Sampling
    and Imaging Applications*. ArXiv: http://arxiv.org/abs/1706.04957 (2017).

    [E+2017] M. J. Ehrhardt, P. J. Markiewicz, P. Richtarik, J. Schott,
    A. Chambolle and C.-B. Schoenlieb, *Faster PET reconstruction with a
    stochastic primal-dual hybrid gradient method*. Wavelets and Sparsity XVII,
    58 (2017) http://doi.org/10.1117/12.2272946.
    """
    # Callback object
    callback = kwargs.pop('callback', None)
    if callback is not None and not callable(callback):
        raise TypeError('`callback` {} is not callable'
                        ''.format(callback))

    # Dual variable
    y = kwargs.pop('y', None)
    if y is None:
        y = A.range.zero()

    # Adjoint of dual variable
    z = kwargs.pop('z', None)
    if z is None:
        if y.norm() == 0:
            z = A.domain.zero()
        else:
            z = A.adjoint(y)

    # Strong convexity of g
    mu_g = kwargs.pop('mu_g', None)
    if mu_g is None:
        update_proximal_primal = False
    else:
        update_proximal_primal = True

    # Global extrapolation factor theta
    theta = kwargs.pop('theta', 1)

    # Second extrapolation factor
    extra = kwargs.pop('extra', None)
    if extra is None:
        extra = [1] * len(sigma)

    # Selection function
    fun_select = kwargs.pop('fun_select', None)
    if fun_select is None:
        def fun_select(x):
            return [int(np.random.choice(len(A), 1, p=1 / len(A)))]

    # Initialize variables
    z_relax = z.copy()
    dz = A.domain.element()
    y_old = A.range.element()

    # Save proximal operators
    proximal_dual_sigma = [fi.convex_conj.proximal(si)
                           for fi, si in zip(f, sigma)]
    proximal_primal_tau = g.proximal(tau)

    # run the iterations
    for k in range(niter):

        # select block
        selected = fun_select(k)

        # update primal variable
        # tmp = x - tau * z_relax; z_relax used as tmp variable
        z_relax.lincomb(1, x, -tau, z_relax)
        # x = prox(tmp)
        proximal_primal_tau(z_relax, out=x)

        # update extrapolation parameter theta
        if update_proximal_primal:
            theta = float(1 / np.sqrt(1 + 2 * mu_g * tau))

        # update dual variable and z, z_relax
        z_relax.assign(z)
        for i in selected:

            # save old yi
            y_old[i].assign(y[i])

            # tmp = Ai(x)
            A[i](x, out=y[i])

            # tmp = y_old + sigma_i * Ai(x)
            y[i].lincomb(1, y_old[i], sigma[i], y[i])

            # y[i]= prox(tmp)
            proximal_dual_sigma[i](y[i], out=y[i])

            # update adjoint of dual variable
            y_old[i].lincomb(-1, y_old[i], 1, y[i])
            A[i].adjoint(y_old[i], out=dz)
            z += dz

            # compute extrapolation
            z_relax.lincomb(1, z_relax, 1 + theta * extra[i], dz)

        # update the step sizes tau and sigma for acceleration
        if update_proximal_primal:
            for i in range(len(sigma)):
                sigma[i] /= theta
            tau *= theta

            proximal_dual_sigma = [fi.convex_conj.proximal(si)
                                   for fi, si in zip(f, sigma)]
            proximal_primal_tau = g.proximal(tau)

        if callback is not None:
            callback([x, y])


def da_spdhg(x, f, g, A, tau, sigma_tilde, niter, mu, **kwargs):
    r"""Computes a saddle point with a PDHG and dual acceleration.

    It therefore requires the functionals f*_i to be mu[i] strongly convex.

    Parameters
    ----------
    x : primal variable
        This variable is both input and output of the method.
    f : functions
        Functionals Y[i] -> IR_infty that all have a convex conjugate with a
        proximal operator, i.e.
        f[i].convex_conj.proximal(sigma[i]) : Y[i] -> Y[i].
    g : function
        Functional X -> IR_infty that has a proximal operator, i.e.
        g.proximal(tau) : X -> X.
    A : functions
        Operators A[i] : X -> Y[i] that possess adjoints: A[i].adjoint
    tau : scalar
        Initial step size for primal variable.
    sigma_tilde : scalar
        Related to initial step size for dual variable.
    niter : int
        Number of iterations
    mu: list
        List of strong convexity constants of f*, i.e. mu[i] is the strong
        convexity constant of f*[i].

    Other Parameters
    ----------------
    y: dual variable
        Dual variable is part of a product space
    z: variable
        Adjoint of dual variable, z = A^* y.
    prob: list
        List of probabilities that an index i is selected each iteration. By
        default this is uniform serial sampling, p_i = 1/n.
    fun_select : function
        Function that selects blocks at every iteration IN -> {1,...,n}. By
        default this is serial sampling, fun_select(k) selects an index
        i \in {1,...,n} with probability p_i.
    extra: list
        List of local extrapolation paramters for every index i. By default
        extra_i = 1 / p_i.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    References
    ----------
    [CERS2017] A. Chambolle, M. J. Ehrhardt, P. Richtarik and C.-B. Schoenlieb,
    *Stochastic Primal-Dual Hybrid Gradient Algorithm with Arbitrary Sampling
    and Imaging Applications*. ArXiv: http://arxiv.org/abs/1706.04957 (2017).
    """
    # Callback object
    callback = kwargs.pop('callback', None)
    if callback is not None and not callable(callback):
        raise TypeError('`callback` {} is not callable'
                        ''.format(callback))

    # Probabilities
    prob = kwargs.pop('prob', None)
    if prob is None:
        prob = [1 / len(A)] * len(A)

    # Selection function
    fun_select = kwargs.pop('fun_select', None)
    if fun_select is None:
        def fun_select(x):
            return [int(np.random.choice(len(A), 1, p=prob))]

    # Dual variable
    y = kwargs.pop('y', None)
    if y is None:
        y = A.range.zero()

    # Adjoint of dual variable
    z = kwargs.pop('z', None)
    if z is None and y.norm() == 0:
        z = A.domain.zero()

    # Extrapolation
    extra = kwargs.pop('extra', None)
    if extra is None:
        extra = [1 / p for p in prob]

    # Initialize variables
    z_relax = z.copy()
    dz = A.domain.element()
    y_old = A.range.element()

    # Save proximal operators
    prox_dual = [fi.convex_conj.proximal for fi in f]
    prox_primal = g.proximal

    # run the iterations
    for k in range(niter):

        # select block
        selected = fun_select(k)

        # update extrapolation parameter theta
        theta = float(1 / np.sqrt(1 + 2 * sigma_tilde))

        # update primal variable
        # tmp = x - tau * z_relax; z_relax used as tmp variable
        z_relax.lincomb(1, x, -tau, z_relax)
        # x = prox(tmp)
        prox_primal(tau)(z_relax, out=x)

        # update dual variable and z, z_relax
        z_relax.assign(z)
        for i in selected:

            # compute the step sizes sigma_i based on sigma_tilde
            sigma_i = sigma_tilde / (
                mu[i] * (prob[i] - 2 * (1 - prob[i]) * sigma_tilde))

            # save old yi
            y_old[i].assign(y[i])

            # tmp = Ai(x)
            A[i](x, out=y[i])

            # tmp = y_old + sigma_i * Ai(x)
            y[i].lincomb(1, y_old[i], sigma_i, y[i])

            # yi++ = fi*.prox_sigmai(yi)
            prox_dual[i](sigma_i)(y[i], out=y[i])

            # update adjoint of dual variable
            y_old[i].lincomb(-1, y_old[i], 1, y[i])
            A[i].adjoint(y_old[i], out=dz)
            z += dz

            # compute extrapolation
            z_relax.lincomb(1, z_relax, 1 + theta * extra[i], dz)

        # update the step sizes tau and sigma_tilde for acceleration
        sigma_tilde *= theta
        tau /= theta

        if callback is not None:
            callback([x, y])


def spdhg_pesquet(x, f, g, A, tau, sigma, niter, **kwargs):
    r"""Computes a saddle point with a stochstic variant of PDHG [PR2015].

    Parameters
    ----------
    x : primal variable
        This variable is both input and output of the method.
    f : functions
        Functionals Y[i] -> IR_infty that all have a convex conjugate with a
        proximal operator, i.e.
        f[i].convex_conj.proximal(sigma[i]) : Y[i] -> Y[i].
    g : function
        Functional X -> IR_infty that has a proximal operator, i.e.
        g.proximal(tau) : X -> X.
    A : functions
        Operators A[i] : X -> Y[i] that possess adjoints: A[i].adjoint
    tau : scalar / vector / matrix
        Step size for primal variable. Note that the proximal operator of g
        has to be well-defined for this input.
    sigma : list
        List of scalars / vectors / matrices used as step sizes for the dual
        variable. Note that the proximal operators related to f (see above)
        have to be well-defined for this input.
    niter : int
        Number of iterations

    Other Parameters
    ----------------
    y: dual variable
        Dual variable is part of a product space
    z: variable
        Adjoint of dual variable, z = A^* y.
    fun_select : function
        Function that selects blocks at every iteration IN -> {1,...,n}. By
        default this is uniform serial sampling, fun_select(k) selects
        uniformly an i \in {1,...,n}.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    References
    ----------
    [PR2015] J.-C. Pesquet and A. Repetti. *A Class of Randomized Primal-Dual
    Algorithms for Distributed Optimization*.
    ArXiv: http://arxiv.org/abs/1406.6404 (2015).
    """
    # Callback object
    callback = kwargs.pop('callback', None)
    if callback is not None and not callable(callback):
        raise TypeError('`callback` {} is not callable'
                        ''.format(callback))

    # Selection function
    fun_select = kwargs.pop('fun_select', None)
    if fun_select is None:
        def fun_select(x):
            return [int(np.random.choice(len(A), 1, p=1 / len(A)))]

    # Dual variable
    y = kwargs.pop('y', None)
    if y is None:
        y = A.range.zero()

    # Adjoint of dual variable
    z = kwargs.pop('z', None)
    if z is None and y.norm() == 0:
        z = A.domain.zero()

    # Save proximal operators
    prox_f = [f[i].convex_conj.proximal(sigma[i]) for i in range(len(sigma))]
    prox_g = g.proximal(tau)

    # Initialize variables
    x_relax = x.copy()
    y_old = A.range.element()
    dz = z.copy()

    # run the iterations
    for k in range(niter):
        x_relax.lincomb(-1, x)

        # update primal variable
        x.lincomb(1, x, -tau, z)
        prox_g(x, out=x)

        # compute extrapolation
        x_relax.lincomb(1, x_relax, 2, x)

        # select block
        selected = fun_select(k)

        # update dual variable and adj_y
        for i in selected:
            # update dual variable
            y_old[i].assign(y[i])

            A[i](x_relax, out=y[i])
            y[i] *= sigma[i]
            y[i] += y_old[i]
            prox_f[i](y[i], out=y[i])

            # update adjoint of dual variable
            A[i].adjoint(y[i] - y_old[i], out=dz)
            z += dz

        if callback is not None:
            callback([x, y])
