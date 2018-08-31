# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Method(s) to find optimal reconstruction parameter(s) w.r.t. given FOM."""

import numpy as np
import scipy.optimize

__all__ = ('optimal_parameters', )


def optimal_parameters(reconstruction, fom, phantoms, data,
                       initial_param=0):
    r"""Find the optimal parameters for a reconstruction method.

    Notes
    -----
    For a forward operator :math:`A : X \to Y`, a reconstruction operator
    parametrized by :math:`\theta` is some operator
    :math:`R_\theta : Y \to X`
    such that

    .. math::
        R_\theta(A(x)) \approx x.

    The optimal choice of :math:`\theta` is given by

    .. math::
        \theta = \arg\min_\theta fom(R(A(x) + noise), x)

    where :math:`fom : X \times X \to \mathbb{R}` is a figure of merit.

    Parameters
    ----------
    reconstruction : callable
        Function that takes two parameters:

            * data : The data to be reconstructed
            * parameters : Parameters of the reconstruction method

        The function should return the reconstructed image.
    fom : callable
        Function that takes two parameters:

            * reconstructed_image
            * true_image

        and returns a scalar figure of merit.
    phantoms : sequence
        True images.
    data : sequence
        The data to reconstruct from.
    initial_param : array-like, optional
        Initial guess for the parameters, default is zero.

    Returns
    -------
    parameters : 'numpy.ndarray'
        The  optimal parameters for the reconstruction problem.
    """

    def func(lam):
        # Function to be minimized by scipy
        return sum(fom(reconstruction(datai, lam), phantomi)
                   for phantomi, datai in zip(phantoms, data))

    # Pick resolution to fit the one used by the space
    tol = np.finfo(phantoms[0].space.dtype).resolution * 10

    initial_param = np.asarray(initial_param)

    # We use a faster optimizer for the one parameter case
    if initial_param.size == 1:
        bracket = [initial_param - tol, initial_param + tol]
        result = scipy.optimize.minimize_scalar(
            func, bracket=bracket, tol=tol, bounds=None,
            options={'disp': False})
        return result.x
    else:
        # Use a gradient free method to find the best parameters
        parameters = scipy.optimize.fmin_powell(
            func, initial_param, xtol=tol, ftol=tol, disp=False)
        return parameters
