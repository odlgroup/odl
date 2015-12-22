﻿# Copyright 2014, 2015 The ODL development group
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

__all__ = ('derenzo_sources', 'shepp_logan')


def _shepp_logan_ellipse_2d():
    # Modified Shepp Logan
    return [[2.00, .6900, .9200, 0, 0, 0],
            [-.98, .6624, .8740, 0, -.0184, 0],
            [-.02, .1100, .3100, .22, 0, -18],
            [-.02, .1600, .4100, -.22, 0, 18],
            [.01, .2100, .2500, 0, .3500, 0],
            [.01, .0460, .0460, 0, .1000, 0],
            [.01, .0460, .0460, 0, -.1000, 0],
            [.01, .0460, .0230, -.08, -.6050, 0],
            [.01, .0230, .0230, 0, -.6060, 0],
            [.01, .0230, .0460, .06, -.6050, 0]]


def _modified_shepp_logan_ellipse_2d():
    # Modified Shepp Logan
    return [[1.00, .6900, .9200, 0, 0, 0],
            [-.80, .6624, .8740, 0, -.0184, 0],
            [-.20, .1100, .3100, .22, 0, -18],
            [-.20, .1600, .4100, -.22, 0, 18],
            [.10, .2100, .2500, 0, .3500, 0],
            [.10, .0460, .0460, 0, .1000, 0],
            [.10, .0460, .0460, 0, -.1000, 0],
            [.10, .0460, .0230, -.08, -.6050, 0],
            [.10, .0230, .0230, 0, -.6060, 0],
            [.10, .0230, .0460, .06, -.6050, 0]]


def _derenzo_sources_2d():
    """ A popular phantom in spect/pet.

    This phantom defines the source terms.
    """
    return [[1.0, 0.047788, 0.047788, -0.77758, -0.11811, 0.0],
            [1.0, 0.063525, 0.063525, -0.71353, 0.12182, 0.0],
            [1.0, 0.047788, 0.047788, -0.68141, -0.28419, 0.0],
            [1.0, 0.063525, 0.063525, -0.58552, 0.3433, 0.0],
            [1.0, 0.047838, 0.047838, -0.58547, -0.45035, 0.0],
            [1.0, 0.047591, 0.047591, -0.58578, -0.11798, 0.0],
            [1.0, 0.047591, 0.047591, -0.48972, -0.61655, 0.0],
            [1.0, 0.047739, 0.047739, -0.48973, -0.28414, 0.0],
            [1.0, 0.063747, 0.063747, -0.45769, 0.12204, 0.0],
            [1.0, 0.063673, 0.063673, -0.4578, 0.5649, 0.0],
            [1.0, 0.04764, 0.04764, -0.39384, -0.45026, 0.0],
            [1.0, 0.047591, 0.047591, -0.39381, -0.11783, 0.0],
            [1.0, 0.063525, 0.063525, -0.32987, 0.3433, 0.0],
            [1.0, 0.03167, 0.03167, -0.31394, -0.7915, 0.0],
            [1.0, 0.047591, 0.047591, -0.29786, -0.28413, 0.0],
            [1.0, 0.032112, 0.032112, -0.25, -0.68105, 0.0],
            [1.0, 0.063488, 0.063488, -0.20192, 0.12185, 0.0],
            [1.0, 0.047442, 0.047442, -0.20192, -0.11804, 0.0],
            [1.0, 0.079552, 0.079552, -0.15405, 0.59875, 0.0],
            [1.0, 0.031744, 0.031744, -0.1862, -0.79155, 0.0],
            [1.0, 0.03167, 0.03167, -0.18629, -0.57055, 0.0],
            [1.0, 0.031892, 0.031892, -0.12224, -0.68109, 0.0],
            [1.0, 0.03167, 0.03167, -0.1217, -0.45961, 0.0],
            [1.0, 0.032039, 0.032039, -0.05808, -0.79192, 0.0],
            [1.0, 0.031744, 0.031744, -0.058285, -0.57011, 0.0],
            [1.0, 0.03167, 0.03167, -0.05827, -0.3487, 0.0],
            [1.0, 0.079434, 0.079434, 0.0057692, 0.32179, 0.0],
            [1.0, 0.031892, 0.031892, 0.0057692, -0.68077, 0.0],
            [1.0, 0.031446, 0.031446, 0.0057692, -0.45934, 0.0],
            [1.0, 0.031892, 0.031892, 0.0057692, -0.23746, 0.0],
            [1.0, 0.032039, 0.032039, 0.069619, -0.79192, 0.0],
            [1.0, 0.031744, 0.031744, 0.069824, -0.57011, 0.0],
            [1.0, 0.03167, 0.03167, 0.069809, -0.3487, 0.0],
            [1.0, 0.079552, 0.079552, 0.16558, 0.59875, 0.0],
            [1.0, 0.031892, 0.031892, 0.13378, -0.68109, 0.0],
            [1.0, 0.03167, 0.03167, 0.13324, -0.45961, 0.0],
            [1.0, 0.031744, 0.031744, 0.19774, -0.79155, 0.0],
            [1.0, 0.03167, 0.03167, 0.19783, -0.57055, 0.0],
            [1.0, 0.09533, 0.09533, 0.28269, 0.16171, 0.0],
            [1.0, 0.023572, 0.023572, 0.21346, -0.11767, 0.0],
            [1.0, 0.032112, 0.032112, 0.26154, -0.68105, 0.0],
            [1.0, 0.023968, 0.023968, 0.26122, -0.20117, 0.0],
            [1.0, 0.023968, 0.023968, 0.30933, -0.28398, 0.0],
            [1.0, 0.023771, 0.023771, 0.30939, -0.11763, 0.0],
            [1.0, 0.03167, 0.03167, 0.32548, -0.7915, 0.0],
            [1.0, 0.024066, 0.024066, 0.35722, -0.36714, 0.0],
            [1.0, 0.023968, 0.023968, 0.35703, -0.20132, 0.0],
            [1.0, 0.09538, 0.09538, 0.47446, 0.49414, 0.0],
            [1.0, 0.024066, 0.024066, 0.40532, -0.45053, 0.0],
            [1.0, 0.024066, 0.024066, 0.40532, -0.28408, 0.0],
            [1.0, 0.023671, 0.023671, 0.40537, -0.11771, 0.0],
            [1.0, 0.02387, 0.02387, 0.45299, -0.53331, 0.0],
            [1.0, 0.02387, 0.02387, 0.45305, -0.36713, 0.0],
            [1.0, 0.02387, 0.02387, 0.45299, -0.2013, 0.0],
            [1.0, 0.023671, 0.023671, 0.50152, -0.6169, 0.0],
            [1.0, 0.023968, 0.023968, 0.50132, -0.45066, 0.0],
            [1.0, 0.023968, 0.023968, 0.50132, -0.28395, 0.0],
            [1.0, 0.023671, 0.023671, 0.50152, -0.11771, 0.0],
            [1.0, 0.024066, 0.024066, 0.54887, -0.69934, 0.0],
            [1.0, 0.023771, 0.023771, 0.54894, -0.5333, 0.0],
            [1.0, 0.023771, 0.023771, 0.54872, -0.36731, 0.0],
            [1.0, 0.023771, 0.023771, 0.54894, -0.20131, 0.0],
            [1.0, 0.09533, 0.09533, 0.66643, 0.16163, 0.0],
            [1.0, 0.02387, 0.02387, 0.59739, -0.61662, 0.0],
            [1.0, 0.023968, 0.023968, 0.59748, -0.45066, 0.0],
            [1.0, 0.023968, 0.023968, 0.59748, -0.28395, 0.0],
            [1.0, 0.023572, 0.023572, 0.59749, -0.11763, 0.0],
            [1.0, 0.023572, 0.023572, 0.64482, -0.53302, 0.0],
            [1.0, 0.023671, 0.023671, 0.64473, -0.36716, 0.0],
            [1.0, 0.02387, 0.02387, 0.64491, -0.20124, 0.0],
            [1.0, 0.02387, 0.02387, 0.69317, -0.45038, 0.0],
            [1.0, 0.024066, 0.024066, 0.69343, -0.28396, 0.0],
            [1.0, 0.023771, 0.023771, 0.69337, -0.11792, 0.0],
            [1.0, 0.023572, 0.023572, 0.74074, -0.36731, 0.0],
            [1.0, 0.023671, 0.023671, 0.74079, -0.20152, 0.0],
            [1.0, 0.023671, 0.023671, 0.78911, -0.28397, 0.0],
            [1.0, 0.02387, 0.02387, 0.78932, -0.11793, 0.0],
            [1.0, 0.023572, 0.023572, 0.83686, -0.20134, 0.0],
            [1.0, 0.023968, 0.023968, 0.88528, -0.11791, 0.0]]


def _shepp_logan_ellipse_3d():
    return [[2.00, .6900, .9200, .810, 0.0000, 0.0000, 0.00, 0.0, 0, 0],
            [-.98, .6624, .8740, .780, 0.0000, -.0184, 0.00, 0.0, 0, 0],
            [-.02, .1100, .3100, .220, 0.2200, 0.0000, 0.00, -18, 0, 10],
            [-.02, .1600, .4100, .280, -.2200, 0.0000, 0.00, 18., 0, 10],
            [.010, .2100, .2500, .410, 0.0000, 0.3500, -.15, 0.0, 0, 0],
            [.010, .0460, .0460, .050, 0.0000, 0.1000, 0.25, 0.0, 0, 0],
            [.010, .0460, .0460, .050, 0.0000, -.1000, 0.25, 0.0, 0, 0],
            [.010, .0460, .0230, .050, -.0800, -.6050, 0.00, 0.0, 0, 0],
            [.010, .0230, .0230, .020, 0.0000, -.6060, 0.00, 0.0, 0, 0],
            [.010, .0230, .0460, .020, 0.0600, -.6050, 0.00, 0.0, 0, 0]]


def _modified_shepp_logan_ellipse_3d():
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


def _phantom_2d(space, ellipses):
    """Create a phantom in 2d space.

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
        phi = ellip[5] * np.pi / 180.0

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


def _phantom_3d(space, ellipses):
    """Create a phantom in 3d space.

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

    # Blank image
    p = np.zeros(space.size)

    # Create the pixel grid
    points = space.points()
    minp = space.grid.min()
    maxp = space.grid.max()

    # move points to [-1, 1]
    np.subtract(points, (minp + maxp) / 2.0, out=points)
    np.divide(points, (maxp - minp) / 2.0, out=points)

    # reusable temporary
    offset_points = np.empty_like(points)

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
        np.subtract(points, [x0, y0, z0], out=offset_points)
        scales = [1 / a2, 1 / b2, 1 / c2]

        if any([phi, theta, psi]):
            # Optimization, only rotate if needed.
            cphi = np.cos(phi)
            sphi = np.sin(phi)
            ctheta = np.cos(theta)
            stheta = np.sin(theta)
            cpsi = np.cos(psi)
            spsi = np.sin(psi)

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

            rotated = np.dot(mat, offset_points.T)
            np.square(rotated, out=rotated)
            radius = np.dot(scales, rotated)
        else:
            np.square(offset_points.T, out=offset_points.T)
            radius = np.dot(scales, offset_points.T)

        # Find the pixels within the ellipse
        inside = radius <= 1

        # Add the ellipse intensity to those pixels
        p[inside] += I

    return space.element(p)


def derenzo_sources(space):
    """ Creates the PET/SPECT derenzo phantom.

    The derenzo phantom contains a series of circles of decreasing size.
    """
    if space.ndim == 2:
        return _phantom_2d(space, _derenzo_sources_2d())
    else:
        raise ValueError("Dimension not 2, no phantom available")


def shepp_logan(space, modified=True):
    """ Create a Shepp-Logan phantom.

    The shepp-logan phantom

    References
    ----------
    Wikipedia : https://en.wikipedia.org/wiki/Shepp%E2%80%93Logan_phantom
    """
    if space.ndim == 2:
        if modified:
            return _phantom_2d(space, _modified_shepp_logan_ellipse_2d())
        else:
            return _phantom_2d(space, _shepp_logan_ellipse_2d())
    elif space.ndim == 3:
        if modified:
            return _phantom_3d(space, _modified_shepp_logan_ellipse_3d())
        else:
            return _phantom_3d(space, _shepp_logan_ellipse_3d())
    else:
        raise ValueError("Dimension not 2 or 3, no phantom available")


if __name__ == '__main__':
    # Show the phantoms
    import odl

    # 2D
    disc = odl.uniform_discr([-1, -1], [1, 1], [200, 200])

    shepp_logan(disc, modified=True).show()
    shepp_logan(disc, modified=False).show()
    derenzo_sources(disc).show()

    # Shepp-logan 3d
    disc = odl.uniform_discr([-1, -1, -1], [1, 1, 1], [200, 200, 200])
    shepp_logan_3d = shepp_logan(disc)
    for i in [60, 100]:
        shepp_logan_3d.show(indices=np.s_[:, :, i])
