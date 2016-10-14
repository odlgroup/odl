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

"""Phantoms typically used in transmission tomography."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

from odl.phantom.geometric import ellipse_phantom


__all__ = ('shepp_logan_ellipses', 'shepp_logan')


def _shepp_logan_ellipse_2d():
    """Return ellipse parameters for a 2d Shepp-Logan phantom.

    This assumes that the ellipses are contained in the square
    [-1, -1]x[-1, -1].
    """
    #       value  axisx  axisy     x       y  rotation
    return [[2.00, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [-.02, .1100, .3100, 0.2200, 0.0000, -18],
            [-.02, .1600, .4100, -.2200, 0.0000, 18],
            [0.01, .2100, .2500, 0.0000, 0.3500, 0],
            [0.01, .0460, .0460, 0.0000, 0.1000, 0],
            [0.01, .0460, .0460, 0.0000, -.1000, 0],
            [0.01, .0460, .0230, -.0800, -.6050, 0],
            [0.01, .0230, .0230, 0.0000, -.6060, 0],
            [0.01, .0230, .0460, 0.0600, -.6050, 0]]


def _shepp_logan_ellipse_3d():
    """Return ellipse parameters for a 3d Shepp-Logan phantom.

    This assumes that the ellipses are contained in the cube
    [-1, -1, -1]x[1, 1, 1].
    """
    #       value  axisx  axisy  axisz,  x        y      z    rotation
    return [[2.00, .6900, .9200, .810, 0.0000, 0.0000, 0.00, 0.0, 0, 0],
            [-.98, .6624, .8740, .780, 0.0000, -.0184, 0.00, 0.0, 0, 0],
            [-.02, .1100, .3100, .220, 0.2200, 0.0000, 0.00, -18, 0, 0],
            [-.02, .1600, .4100, .280, -.2200, 0.0000, 0.00, 18., 0, 0],
            [0.01, .2100, .2500, .410, 0.0000, 0.3500, 0.00, 0.0, 0, 0],
            [0.01, .0460, .0460, .050, 0.0000, 0.1000, 0.00, 0.0, 0, 0],
            [0.01, .0460, .0460, .050, 0.0000, -.1000, 0.00, 0.0, 0, 0],
            [0.01, .0460, .0230, .050, -.0800, -.6050, 0.00, 0.0, 0, 0],
            [0.01, .0230, .0230, .020, 0.0000, -.6060, 0.00, 0.0, 0, 0],
            [0.01, .0230, .0460, .020, 0.0600, -.6050, 0.00, 0.0, 0, 0]]


def _modified_shepp_logan_ellipses(ellipses):
    """Modify ellipses to give the modified Shepp-Logan phantom.

    Works for both 2d and 3d.
    """
    intensities = [1.0, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    assert len(ellipses) == len(intensities)

    for ellipse, intensity in zip(ellipses, intensities):
        ellipse[0] = intensity


def shepp_logan_ellipses(ndim, modified=False):
    """Ellipses for the standard `Shepp-Logan phantom`_ in 2 or 3 dimensions.

    Parameters
    ----------
    ndim : {2, 3}
        Dimension of the space the ellipses should be in.
    modified : bool, optional
        True if the modified Shepp-Logan phantom should be given.
        The modified phantom has greatly amplified contrast to aid
        visualization.

    See Also
    --------
    odl.phantom.geometric.ellipse_phantom :
        Function for creating arbitrary ellipse phantoms
    shepp_logan : Create a phantom with these ellipses

    References
    ----------
    .. _Shepp-Logan phantom: en.wikipedia.org/wiki/Shepp–Logan_phantom
    """
    if ndim == 2:
        ellipses = _shepp_logan_ellipse_2d()
    elif ndim == 3:
        ellipses = _shepp_logan_ellipse_3d()
    else:
        raise ValueError('dimension not 2 or 3, no phantom available')

    if modified:
        _modified_shepp_logan_ellipses(ellipses)

    return ellipses


def shepp_logan(space, modified=False):
    """Standard `Shepp-Logan phantom`_ in 2 or 3 dimensions.

    Parameters
    ----------
    space : `DiscreteLp`
        Space in which the phantom is created, must be 2- or 3-dimensional.
    modified : `bool`, optional
        True if the modified Shepp-Logan phantom should be given.
        The modified phantom has greatly amplified contrast to aid
        visualization.

    See Also
    --------
    shepp_logan_ellipses : Get the parameters that define this phantom
    odl.phantom.geometric.ellipse_phantom :
        Function for creating arbitrary ellipse phantoms

    References
    ----------
    .. _Shepp-Logan phantom: en.wikipedia.org/wiki/Shepp–Logan_phantom
    """
    ellipses = shepp_logan_ellipses(space.ndim, modified)

    return ellipse_phantom(space, ellipses)


if __name__ == '__main__':
    # Show the phantoms
    import odl

    # 2D
    discr = odl.uniform_discr([-1, -1], [1, 1], [1000, 1000])
    shepp_logan(discr, modified=True).show('shepp_logan 2d modified=True')
    shepp_logan(discr, modified=False).show('shepp_logan 2d modified=False')

    # 3D
    discr = odl.uniform_discr([-1, -1, -1], [1, 1, 1], [300, 300, 300])
    shepp_logan(discr, modified=True).show('shepp_logan 3d modified=True')
    shepp_logan(discr, modified=False).show('shepp_logan 3d modified=False')

    # Run also the doctests
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
