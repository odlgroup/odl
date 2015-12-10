# Copyright 2014, 2015 The ODL development group
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

"""Utilities for internal use."""


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import numpy as np

__all__ = ('shepp_logan',)


def _shepp_logan_ellipse_2d():
    # Modified Shepp Logan
    return [[1.00, .6900, .9200, 0, 0, 0],
            [-.80, .6624, .8740, 0, -.0184, 0],
            [-.20, .1100, .3100, .22, 0, -18 * np.pi / 180],
            [-.20, .1600, .4100, -.22, 0, 18 * np.pi / 180],
            [.10, .2100, .2500, 0, .3500, 0],
            [.10, .0460, .0460, 0, .1000, 0],
            [.10, .0460, .0460, 0, -.1000, 0],
            [.10, .0460, .0230, -.08, -.6050, 0],
            [.10, .0230, .0230, 0, -.6060, 0],
            [.10, .0230, .0460, .06, -.6050, 0]]


def _phantom_2d(space, ellipses):
    """Create a shepp logan phantom in 2d space.

    Parameters
    ----------
    space : `DiscreteLp`
        The space the phantom should be generated in.
    ellipses : list of lists
        Each row should contain:
        'value', 'axis_1', 'axis_2', 'center_x', 'center_y', 'rotation'
        The ellipses should be contained the he rectangle [-1, -1] x [1, 1].

    Returns
    -------
    phantom : `DisceteLpVector`
        The phantom
    """
    ellipses = _shepp_logan_ellipse_2d()

    # Blank image
    p = np.zeros(space.size)

    # Create the pixel grid
    points = space.points()
    minp = space.grid.min()
    maxp = space.grid.max()

    # move points to [0, 1]
    points = (points - minp) / (maxp - minp)

    # move to [-1, 1]
    points = points * 2 - 1

    for ellip in ellipses:
        I = ellip[0]
        a2 = ellip[1] ** 2
        b2 = ellip[2] ** 2
        x0 = ellip[3]
        y0 = ellip[4]
        phi = ellip[5]

        # Create the offset x and y values for the grid
        offset_points = points - [x0, y0]

        cos_p = np.cos(phi)
        sin_p = np.sin(phi)

        # Find the pixels within the ellipse
        scales = [1 / a2, 1 / b2]
        mat = [[cos_p, sin_p],
               [-sin_p, cos_p]]
        radius = np.dot(scales, np.dot(mat, offset_points.T) ** 2)
        inside = radius <= 1

        # Add the ellipse intensity to those pixels
        p[inside] += I

    return space.element(p)


def _shepp_logan_ellipse_3d():
    # Modified Shepp Logan
    return [[1.00, .6900, .9200, .810, 0.0000, 0.0000, 0.00, 0.0, 0, 0],
            [-.80, .6624, .8740, .780, 0.0000, -.0184, 0.00, 0.0, 0, 0],
            [-.20, .1100, .3100, .220, 0.2200, 0.0000, 0.00, -18, 0, 10],
            [-.20, .1600, .4100, .280, -.2200, 0.0000, 0.00, 18., 0, 10],
            [.100, .2100, .2500, .410, 0.0000, 0.3500, -.15, 0.0, 0, 0],
            [.100, .0460, .0460, .050, 0.0000, 0.1000, 0.25, 0.0, 0, 0],
            [.100, .0460, .0460, .050, 0.0000, -.1000, 0.25, 0.0, 0, 0],
            [.100, .0460, .0230, .050, -.0800, -.6050, 0.00, 0.0, 0, 0],
            [.100, .0230, .0230, .020, 0.0000, -.6060, 0.00, 0.0, 0, 0],
            [.100, .0230, .0460, .020, 0.0600, -.6050, 0.00, 0.0, 0, 0]]


def _phantom_3d(space):
    """Create a shepp logan phantom in 3d space.

    Parameters
    ----------
    space : `DiscreteLp`
        The space the phantom should be generated in.
    ellipses : list of lists
        Each row should contain:
        'value', 'axis_1', 'axis_2', 'axis_2',
        'center_x', 'center_y', 'center_z',
        'rotation_phi', 'rotation_theta', 'rotation_psi'
        The ellipses should be contained the he rectangle
        [-1, -1, -1] x [1, 1, 1].

    Returns
    -------
    phantom : `DisceteLpVector`
        The phantom
    """
    ellipses = _shepp_logan_ellipse_3d()

    # Blank image
    p = np.zeros(space.size)

    # Create the pixel grid
    points = space.points()
    minp = space.grid.min()
    maxp = space.grid.max()

    # move points to [0, 1]
    points = (points - minp) / (maxp - minp)

    # move to [-1, 1]
    points = points * 2 - 1

    for ellip in ellipses:
        I = ellip[0]
        a2 = ellip[1] ** 2
        b2 = ellip[2] ** 2
        c2 = ellip[3] ** 2
        x0 = ellip[4]
        y0 = ellip[5]
        z0 = ellip[6]
        phi = ellip[7] * np.pi / 180
        theta = ellip[8] * np.pi / 180
        psi = ellip[9] * np.pi / 180

        # Create the offset x,y and z values for the grid
        offset_points = points - [x0, y0, z0]

        cphi = np.cos(phi)
        sphi = np.sin(phi)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)

        # Find the pixels within the ellipse
        scales = [1 / a2, 1 / b2, 1 / c2]
        mat = [[cpsi * cphi - ctheta * sphi * spsi,
                cpsi * sphi + ctheta * cphi * spsi,
                spsi * stheta],
               [-spsi * cphi - ctheta * sphi * cpsi,
                -spsi * sphi + ctheta * cphi * cpsi,
                cpsi * stheta],
               [stheta * sphi,
                -stheta * cphi,
                ctheta]]

        radius = np.dot(scales, np.dot(mat, offset_points.T) ** 2)
        inside = radius <= 1

        # Add the ellipse intensity to those pixels
        p[inside] += I

    return space.element(p)


def shepp_logan(space):
    if space.ndim == 2:
        return _phantom_2d(space, _shepp_logan_ellipse_2d())
    elif space.ndim == 3:
        return _phantom_3d(space, _shepp_logan_ellipse_3d())
    else:
        raise ValueError("Dimension not 2 or 3, no phantom available")

if __name__ == '__main__':
    # Show the phantoms
    import odl

    # Shepp-logan 2d
    disc = odl.uniform_discr([-1, -1], [1, 1], [100, 100])
    odl.util.shepp_logan(disc).show()

    # Shepp-logan 3d
    disc = odl.uniform_discr([-1, -1, -1], [1, 1, 1], [100, 100, 100])
    shepp_logan_3d = odl.util.shepp_logan(disc)
    for i in [30, 50]:
        shepp_logan_3d.show(indices=np.s_[:, :, i])
