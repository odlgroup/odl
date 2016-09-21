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

__all__ = ('mlem', 'loglikelihood')


AVAILABLE_MLEM_NOISE = ('poisson',)


def mlem(op, x, data, niter=1, noise='poisson', callback=None):

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
    data : ``op.range`` element
        Right-hand side of the equation defining the inverse problem.
    niter : int, optional
        Number of iterations.
    noise : {'poisson'}, optional
        Noise model determining the variant of MLEM.
        For ``'poisson'``, the initial value of ``x`` should be
        non-negative.
    callback : callable, optional
        Function called with the current iterate after each iteration.

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
    noise, noise_in = str(noise).lower(), noise
    if noise not in AVAILABLE_MLEM_NOISE:
        raise NotImplemented("noise '{}' not understood"
                             ''.format(noise_in))

    if noise == 'poisson':
        if np.any(np.less(x, 0)):
            raise ValueError('`x` must be non-negative')

        eps = 1e-8

        norm = np.maximum(op.adjoint(op.range.one()), eps)
        tmp_dom = op.domain.element()
        tmp_ran = op.range.element()

        for _ in range(niter):
            op(x, out=tmp_ran)
            tmp_ran.ufunc.maximum(eps, out=tmp_ran)
            data.divide(tmp_ran, out=tmp_ran)

            op.adjoint(tmp_ran, out=tmp_dom)
            tmp_dom /= norm

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
