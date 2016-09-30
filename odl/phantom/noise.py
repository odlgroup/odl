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

"""Functions to create noise samples of different distributions."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np


__all__ = ('white_noise', 'poisson_noise')


def white_noise(space):
    """Standard gaussian noise in space, pointwise N(0, 1).

    Parameters
    ----------
    space : `FnBase`
        The space that the noise should be created in

    Returns
    -------
    white_noise : ``space`` element

    See Also
    --------
    poisson_noise
    numpy.random.randn
    """
    values = np.random.randn(*space.shape)
    return space.element(values)


def poisson_noise(intensity):
    """Poisson distributed noise with given intensity.

    Parameters
    ----------
    intensity : `FnBase`
        The intensity (usually called lambda) parameter of the noise.

    Returns
    -------
    poisson_noise : ``intensity.space`` element
        Poisson distributed random variable.

    Notes
    -----
    The probability density function for value :math:`k` and intensity
    :math:`\\lambda` is given by:

    .. math::
        \\frac{\\lambda^k e^{-\\lambda}}{k!}

    Note that the function only takes integer values.

    See Also
    --------
    white_noise
    numpy.random.poisson
    """
    values = np.random.poisson(intensity.asarray())
    return intensity.space.element(values)


if __name__ == '__main__':
    # Show the phantoms
    import odl

    discr = odl.uniform_discr([-1, -1], [1, 1], [300, 300])
    white_noise(discr).show('white_noise')

    # Run also the doctests
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
