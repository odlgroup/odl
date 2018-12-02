# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Functions to create noise samples of different distributions."""

from __future__ import absolute_import, division, print_function

import numpy as np

from odl.util import npy_random_seed

__all__ = ('white_noise', 'poisson_noise', 'salt_pepper_noise',
           'uniform_noise')


def white_noise(space, mean=0, stddev=1, seed=None):
    """Standard gaussian noise in space, pointwise ``N(mean, stddev**2)``.

    Parameters
    ----------
    space : `TensorSpace` or `ProductSpace`
        The space in which the noise is created.
    mean : ``space.field`` element or ``space`` `element-like`, optional
        The mean of the white noise. If a scalar, it is interpreted as
        ``mean * space.one()``.
        If ``space`` is complex, the real and imaginary parts are interpreted
        as the mean of their respective part of the noise.
    stddev : `float` or ``space`` `element-like`, optional
        The standard deviation of the white noise. If a scalar, it is
        interpreted as ``stddev * space.one()``.
    seed : int, optional
        Random seed to use for generating the noise.
        For ``None``, use the current seed.

    Returns
    -------
    white_noise : ``space`` element

    See Also
    --------
    poisson_noise
    salt_pepper_noise
    numpy.random.normal
    """
    from odl.space import ProductSpace

    with npy_random_seed(seed):
        if isinstance(space, ProductSpace):
            values = [white_noise(subspace, mean, stddev)
                      for subspace in space]
        else:
            if space.is_complex:
                real = np.random.normal(
                    loc=mean.real, scale=stddev, size=space.shape)
                imag = np.random.normal(
                    loc=mean.imag, scale=stddev, size=space.shape)
                values = real + 1j * imag
            else:
                values = np.random.normal(
                    loc=mean, scale=stddev, size=space.shape)

    return space.element(values)


def uniform_noise(space, low=0, high=1, seed=None):
    """Uniformly distributed noise in ``space``, pointwise ``U(low, high)``.

    Parameters
    ----------
    space : `TensorSpace` or `ProductSpace`
        The space in which the noise is created.
    low : ``space.field`` element or ``space`` `element-like`, optional
        The lower bound of the uniform noise. If a scalar, it is interpreted as
        ``low * space.one()``.
        If ``space`` is complex, the real and imaginary parts are interpreted
        as their respective part of the noise.
    high : ``space.field`` element or ``space`` `element-like`, optional
        The upper bound of the uniform noise. If a scalar, it is interpreted as
        ``high * space.one()``.
        If ``space`` is complex, the real and imaginary parts are interpreted
        as their respective part of the noise.
    seed : int, optional
        Random seed to use for generating the noise.
        For ``None``, use the current seed.

    Returns
    -------
    white_noise : ``space`` element

    See Also
    --------
    poisson_noise
    salt_pepper_noise
    white_noise
    numpy.random.normal
    """
    from odl.space import ProductSpace

    with npy_random_seed(seed):
        if isinstance(space, ProductSpace):
            values = [uniform_noise(subspace, low, high)
                      for subspace in space]
        else:
            if space.is_complex:
                real = np.random.uniform(low=low.real, high=high.real,
                                         size=space.shape)
                imag = np.random.uniform(low=low.imag, high=high.imag,
                                         size=space.shape)
                values = real + 1j * imag
            else:
                values = np.random.uniform(low=low, high=high,
                                           size=space.shape)

    return space.element(values)


def poisson_noise(intensity, seed=None):
    r"""Poisson distributed noise with given intensity.

    Parameters
    ----------
    intensity : `TensorSpace` or `ProductSpace` element
        The intensity (usually called lambda) parameter of the noise.

    Returns
    -------
    poisson_noise : ``intensity.space`` element
        Poisson distributed random variable.
    seed : int, optional
        Random seed to use for generating the noise.
        For ``None``, use the current seed.

    Notes
    -----
    For a Poisson distributed random variable :math:`X` with intensity
    :math:`\lambda`, the probability of it taking the value
    :math:`k \in \mathbb{N}_0` is given by

    .. math::
        \frac{\lambda^k e^{-\lambda}}{k!}

    Note that the function only takes integer values.

    See Also
    --------
    white_noise
    salt_pepper_noise
    uniform_noise
    numpy.random.poisson
    """
    from odl.space import ProductSpace

    with npy_random_seed(seed):
        if isinstance(intensity.space, ProductSpace):
            values = [poisson_noise(subintensity)
                      for subintensity in intensity]
        else:
            values = np.random.poisson(intensity.asarray())

    return intensity.space.element(values)


def salt_pepper_noise(vector, fraction=0.05, salt_vs_pepper=0.5,
                      low_val=None, high_val=None, seed=None):
    """Add salt and pepper noise to vector.

    Salt and pepper noise replaces random elements in ``vector`` with
    ``low_val`` or ``high_val``.

    Parameters
    ----------
    vector : element of `TensorSpace` or `ProductSpace`
        The vector that noise should be added to.
    fraction : float, optional
        The propotion of the elements in ``vector`` that should be converted
        to noise.
    salt_vs_pepper : float, optional
        Relative abundance of salt (high) vs pepper (low) noise. A high value
        means more salt than pepper noise.
    low_val : float, optional
        The "pepper" color in the noise.
        Default: minimum value of ``vector``. For product spaces the minimum
        value per subspace is taken.
    high_val : float, optional
        The "salt" value in the noise.
        Default: maximuim value of ``vector``. For product spaces the maximum
        value per subspace is taken.
    seed : int, optional
        Random seed to use for generating the noise.
        For ``None``, use the current seed.

    Returns
    -------
    salt_pepper_noise : ``vector.space`` element
        ``vector`` with salt and pepper noise.

    See Also
    --------
    white_noise
    poisson_noise
    uniform_noise
    """
    from odl.space import ProductSpace

    # Validate input parameters
    fraction, fraction_in = float(fraction), fraction
    if not (0 <= fraction <= 1):
        raise ValueError('`fraction` ({}) should be a float in the interval '
                         '[0, 1]'.format(fraction_in))

    salt_vs_pepper, salt_vs_pepper_in = float(salt_vs_pepper), salt_vs_pepper
    if not (0 <= salt_vs_pepper <= 1):
        raise ValueError('`salt_vs_pepper` ({}) should be a float in the '
                         'interval [0, 1]'.format(salt_vs_pepper_in))

    with npy_random_seed(seed):
        if isinstance(vector.space, ProductSpace):
            values = [salt_pepper_noise(subintensity, fraction, salt_vs_pepper,
                                        low_val, high_val)
                      for subintensity in vector]
        else:
            # Extract vector of values
            values = vector.asarray().flatten()

            # Determine fill-in values if not given
            if low_val is None:
                low_val = np.min(values)
            if high_val is None:
                high_val = np.max(values)

            # Create randomly selected points as a subset of image.
            a = np.arange(vector.size)
            np.random.shuffle(a)
            salt_indices = a[:int(fraction * vector.size * salt_vs_pepper)]
            pepper_indices = a[int(fraction * vector.size * salt_vs_pepper):
                               int(fraction * vector.size)]

            values[salt_indices] = high_val
            values[pepper_indices] = -low_val
            values = values.reshape(vector.space.shape)

    return vector.space.element(values)


if __name__ == '__main__':
    # Show the phantoms
    import odl
    from odl.util.testutils import run_doctests

    r100 = odl.rn(100)
    white_noise(r100).show('white_noise')
    uniform_noise(r100).show('uniform_noise')
    white_noise(r100, mean=5).show('white_noise with mean')

    c100 = odl.cn(100)
    white_noise(c100).show('complex white_noise')
    uniform_noise(c100).show('complex uniform_noise')

    discr = odl.uniform_discr([-1, -1], [1, 1], [300, 300])
    white_noise(discr).show('white_noise 2d')
    uniform_noise(discr).show('uniform_noise 2d')

    vector = odl.phantom.shepp_logan(discr, modified=True)
    poisson_noise(vector * 100).show('poisson_noise 2d')
    salt_pepper_noise(vector).show('salt_pepper_noise 2d')

    # Run also the doctests
    run_doctests()
