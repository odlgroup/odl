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

"""Maximum Likelihood Expectation Maximization algorithm."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np

__all__ = ('mlem', 'osmlem', 'loglikelihood')


AVAILABLE_MLEM_NOISE = ('poisson',)


def mlem(op, x, data, niter=1, noise='poisson', callback=None, **kwargs):

    """Maximum Likelihood Expectation Maximation algorithm.

    Attempts to solve::

        max_x L(data | x)

    where ``L(data, | x)`` is the likelihood of ``data`` given ``x``. The
    likelihood depends on the forward operator ``op`` such that
    (approximately)::

        op(x) = data

    With the precise form of *approximately* is determined by ``noise``.

    Parameters
    ----------
    op : `Operator`
        Forward operator in the inverse problem.
    x : ``op.domain`` element
        Vector to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
    data : ``op.range`` `element-like`
        Right-hand side of the equation defining the inverse problem.
    niter : int, optional
        Number of iterations.
    noise : {'poisson'}, optional
        Noise model determining the variant of MLEM.
        For ``'poisson'``, the initial value of ``x`` should be
        non-negative.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    Other Parameters
    ----------------
    sensitivities : float or ``op.domain`` `element-like`, optional
        Usable with ``noise='poisson'``. The algorithm contains a ``A^T 1``
        term, if this parameter is given, it is replaced by it.
        Default: ``op.adjoint(op.range.one())``

    Notes
    -----
    Given a forward model :math:`A`, data :math:`g` and noise model :math:`X`,
    the algorithm attempts find an :math:`x` that maximizes:

    .. math::

        P(g | g \\text{ is } X(A(x)) \\text{ distributed})

    where the expectation of :math:`X(A(x))` satisfies

    .. math::

        \\mathbb{E}(X(A(x))) = A(x)

    with 'poisson' noise the algorithm is given by:

    .. math::

       x_{n+1} = \\frac{x_n}{A^* 1} A^* (g / A(x_n))

    See Also
    --------
    loglikelihood : Function for calculating the logarithm of the likelihood
    """
    # TODO: Should this be call to osmlem?

    noise, noise_in = str(noise).lower(), noise
    if noise not in AVAILABLE_MLEM_NOISE:
        raise NotImplemented("noise '{}' not understood"
                             ''.format(noise_in))

    # Convert data to range element
    data = op.range.element(data)

    if noise == 'poisson':
        # Parameter used to enforce positivity.
        # TODO: let users give this.
        eps = 1e-8

        if np.any(np.less(x, 0)):
            raise ValueError('`x` must be non-negative')

        sensitivities = kwargs.pop('sensitivities', None)
        if sensitivities is None:
            sensitivities = np.maximum(op.adjoint(op.range.one()), eps)

        tmp_dom = op.domain.element()
        tmp_ran = op.range.element()

        for _ in range(niter):
            op(x, out=tmp_ran)
            tmp_ran.ufunc.maximum(eps, out=tmp_ran)
            data.divide(tmp_ran, out=tmp_ran)

            op.adjoint(tmp_ran, out=tmp_dom)
            tmp_dom /= sensitivities

            x *= tmp_dom

            if callback is not None:
                callback(x)
    else:
        raise RuntimeError('unknown noise model')


def osmlem(op, x, data, niter=1, noise='poisson', callback=None, **kwargs):
    """Ordered Subsets Maximum Likelihood Expectation Maximation algorithm.

    Attempts to solve::

        max_x L(data | x)

    where ``L(data, | x)`` is the likelihood of ``data`` given ``x``. The
    likelihood depends on the forward operators ``op[0], ..., op[n-1]`` such
    that (approximately)::

        op[i](x) = data[i]

    With the precise form of *approximately* is determined by ``noise``.

    Parameters
    ----------
    op : sequence of `Operator`
        Forward operators in the inverse problem.
    x : ``op.domain`` element
        Vector to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
    data : sequence of ``op.range`` `element-like`
        Right-hand sides of the equation defining the inverse problem.
    niter : int, optional
        Number of iterations.
    noise : {'poisson'}, optional
        Noise model determining the variant of MLEM.
        For ``'poisson'``, the initial value of ``x`` should be
        non-negative.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    Other Parameters
    ----------------
    sensitivities : float or ``op.domain`` `element-like`, optional
        Usable with ``noise='poisson'``. The algorithm contains a ``A^T 1``
        term, if this parameter is given, it is replaced by it.
        Default: ``op[i].adjoint(op[i].range.one())``

    Notes
    -----
    Given a forward models :math:`A_i`, data :math:`g_i`, :math:`i = 1, ..., M`
    and noise model :math:`X`, the algorithm attempts find an :math:`x` that
    maximizes:

    .. math::

        \prod_{i=1}^M P(g_i | g_i \\text{ is } X(A_i(x)) \\text{ distributed})

    where the expectation of :math:`X(A_i(x))` satisfies

    .. math::

        \\mathbb{E}(X(A_i(x))) = A_i(x)

    with 'poisson' noise the algorithm is given by partial updates:

    .. math::

       x_{n + m/M} =
       \\frac{x_{n + (m - 1)/M}}{A_i^* 1} A_i^* (g_i / A_i(x_{n + (m - 1)/M}))

    for :math:`m = 1, ..., M` and :math:`x_{n+1} = x_{n + M/M}`

    See Also
    --------
    mlem : Ordinary MLEM algorithm without subsets.
    loglikelihood : Function for calculating the logarithm of the likelihood
    """
    noise, noise_in = str(noise).lower(), noise
    if noise not in AVAILABLE_MLEM_NOISE:
        raise NotImplemented("noise '{}' not understood"
                             ''.format(noise_in))

    n_ops = len(op)
    if len(data) != n_ops:
        raise ValueError('number of data does not match number of operators')
    if not all(x in opi.domain for opi in op):
        raise ValueError('`x` not an element in the domains of all operators')

    # Convert data to range elements
    data = [op[i].range.element(data[i]) for i in range(len(op))]

    if noise == 'poisson':
        # Parameter used to enforce positivity.
        # TODO: let users give this.
        eps = 1e-8

        if np.any(np.less(x, 0)):
            raise ValueError('`x` must be non-negative')

        # Extract the sensitivites parameter
        sensitivities = kwargs.pop('sensitivities', None)
        if sensitivities is None:
            sensitivities = [np.maximum(opi.adjoint(opi.range.one()), eps)
                             for opi in op]
        else:
            # Make sure the sensitivities is a list of the correct size.
            try:
                list(sensitivities)
            except TypeError:
                sensitivities = [sensitivities] * n_ops

        tmp_dom = op[0].domain.element()
        tmp_ran = [opi.range.element() for opi in op]

        for _ in range(niter):
            for i in range(n_ops):
                op[i](x, out=tmp_ran[i])
                tmp_ran[i].ufunc.maximum(eps, out=tmp_ran[i])
                data[i].divide(tmp_ran[i], out=tmp_ran[i])

                op[i].adjoint(tmp_ran[i], out=tmp_dom)
                tmp_dom /= sensitivities[i]

                x *= tmp_dom

                if callback is not None:
                    callback(x)
    else:
        raise RuntimeError('unknown noise model')


def loglikelihood(x, data, noise='poisson'):
    """log-likelihood of ``data`` given noise parametrized by ``x``.

    Parameters
    ----------
    x : ``op.domain`` element
        Value to condition the log-likelihood on.
    data : ``op.range`` element
        Data whose log-likelihood given ``x`` shall be calculated.
    noise : {'poisson'}, optional
        The type of noise.
    """
    noise, noise_in = str(noise).lower(), noise
    if noise not in AVAILABLE_MLEM_NOISE:
        raise NotImplemented("noise '{}' not understood"
                             ''.format(noise_in))

    if noise == 'poisson':
        if np.any(np.less(x, 0)):
            raise ValueError('`x` must be non-negative')

        return np.sum(data * np.log(x + 1e-8) - x)
    else:
        raise RuntimeError('unknown noise model')
