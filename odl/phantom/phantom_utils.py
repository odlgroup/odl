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

"""Utilities for creating phantoms."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np


__all__ = ('ellipse_phantom', 'cylinders_from_ellipses')


def cylinders_from_ellipses(ellipses2d):
    """Create 3d cylinders from ellipses."""
    ellipses2d = np.asarray(ellipses2d)
    ellipses3d = np.zeros((ellipses2d.shape[0], 10))
    ellipses3d[:, [0, 1, 2, 4, 5, 7]] = ellipses2d
    ellipses3d[:, 3] = 100000.0

    return ellipses3d


def _getshapes_2d(center, max_radius, shape):
    """Calculate indices and slices for the bounding box of a disk."""
    index_mean = shape * center
    index_radius = max_radius / 2.0 * np.array(shape)

    min_idx = np.floor(index_mean - index_radius).astype(int)
    max_idx = np.ceil(index_mean + index_radius).astype(int)
    idx = [slice(minx, maxx) for minx, maxx in zip(min_idx, max_idx)]
    shapes = [(idx[0], slice(None)),
              (slice(None), idx[1])]
    return idx, shapes


def ellipse_phantom_2d(space, ellipses):
    """Create a phantom of ellipses in 2d space.

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
    phantom : `DiscreteLpVector`
        The phantom

    Notes
    -----
    This function is heavily optimized, achieving runtimes about 20 times
    faster than "trivial" implementations. It is therefore recommended to use
    in all phantoms.

    The main optimization is that it only considers a subset of all the
    points when updating for each ellipse. It does this by first finding
    a subset of points that could possibly be inside the ellipse. This
    optimization is very good for "spherical" ellipsoids, but not so
    much for elongated or rotated ones.

    It also does calculations wherever possible on the meshgrid instead of
    individual points.

    See Also
    --------
    shepp_logan : The typical use-case for this function.
    """

    # Blank image
    p = np.zeros(space.shape, dtype=space.dtype)

    # Create the pixel grid
    grid_in = space.grid.meshgrid
    minp = space.grid.min()
    maxp = space.grid.max()

    # move points to [-1, 1]
    grid = []
    for i in range(2):
        meani = (minp[i] + maxp[i]) / 2.0
        diffi = (maxp[i] - minp[i]) / 2.0
        grid += [(grid_in[i] - meani) / diffi]

    for ellip in ellipses:
        intensity = ellip[0]
        a_squared = ellip[1] ** 2
        b_squared = ellip[2] ** 2
        x0 = ellip[3]
        y0 = ellip[4]
        theta = ellip[5] * np.pi / 180

        scales = [1 / a_squared, 1 / b_squared]
        center = (np.array([x0, y0]) + 1.0) / 2.0

        # Create the offset x,y and z values for the grid
        if theta != 0:
            # Rotate the points to the expected coordinate system.
            ctheta = np.cos(theta)
            stheta = np.sin(theta)

            mat = np.array([[ctheta, stheta],
                            [-stheta, ctheta]])

            # Calculate the points that could possibly be inside the volume
            # Since the points are rotated, we cannot do anything directional
            # without more logic
            max_radius = np.sqrt(
                np.abs(mat).dot([a_squared, b_squared]))
            idx, shapes = _getshapes_2d(center, max_radius, space.shape)

            subgrid = [g[idi] for g, idi in zip(grid, shapes)]
            offset_points = [vec * (xi - x0i)[..., np.newaxis]
                             for xi, vec, x0i in zip(subgrid,
                                                     mat.T,
                                                     [x0, y0])]
            rotated = offset_points[0] + offset_points[1]
            np.square(rotated, out=rotated)
            radius = np.dot(rotated, scales)
        else:
            # Calculate the points that could possibly be inside the volume
            max_radius = np.sqrt([a_squared, b_squared])
            idx, shapes = _getshapes_2d(center, max_radius, space.shape)

            subgrid = [g[idi] for g, idi in zip(grid, shapes)]
            squared_dist = [ai * (xi - x0i) ** 2
                            for xi, ai, x0i in zip(subgrid,
                                                   scales,
                                                   [x0, y0])]

            # Parentheses to get best order for broadcasting
            radius = squared_dist[0] + squared_dist[1]

        # Find the pixels within the ellipse
        inside = radius <= 1

        # Add the ellipse intensity to those pixels
        p[idx][inside] += intensity

    return space.element(p)


def _getshapes_3d(center, max_radius, shape):
    """Calculate indices and slices for the bounding box of a ball."""
    index_mean = shape * center
    index_radius = max_radius / 2.0 * np.array(shape)

    min_idx = np.floor(index_mean - index_radius).astype(int)
    max_idx = np.ceil(index_mean + index_radius).astype(int)
    idx = [slice(minx, maxx) for minx, maxx in zip(min_idx, max_idx)]
    shapes = [(idx[0], slice(None), slice(None)),
              (slice(None), idx[1], slice(None)),
              (slice(None), slice(None), idx[2])]
    return idx, shapes


def ellipse_phantom_3d(space, ellipses):
    """Create an ellipse phantom in 3d space.

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
    phantom : `DiscreteLpVector`
        The phantom

    Notes
    -----
    This function is heavily optimized, achieving runtimes about 20 times
    faster than "trivial" implementations. It is therefore recommended to use
    in all phantoms.

    The main optimization is that it only considers a subset of all the
    points when updating for each ellipse. It does this by first finding
    a subset of points that could possibly be inside the ellipse. This
    optimization is very good for "spherical" ellipsoids, but not so
    much for elongated or rotated ones.

    It also does calculations wherever possible on the meshgrid instead of
    individual points.

    See Also
    --------
    shepp_logan : The typical use-case for this function.
    """

    # Blank image
    p = np.zeros(space.shape, dtype=space.dtype)

    # Create the pixel grid
    grid_in = space.grid.meshgrid
    minp = space.grid.min()
    maxp = space.grid.max()

    # move points to [-1, 1]
    grid = []
    for i in range(3):
        meani = (minp[i] + maxp[i]) / 2.0
        diffi = (maxp[i] - minp[i]) / 2.0
        grid += [(grid_in[i] - meani) / diffi]

    for ellip in ellipses:
        intensity = ellip[0]
        a_squared = ellip[1] ** 2
        b_squared = ellip[2] ** 2
        c_squared = ellip[3] ** 2
        x0 = ellip[4]
        y0 = ellip[5]
        z0 = ellip[6]
        phi = ellip[7] * np.pi / 180
        theta = ellip[8] * np.pi / 180
        psi = ellip[9] * np.pi / 180

        scales = [1 / a_squared, 1 / b_squared, 1 / c_squared]
        center = (np.array([x0, y0, z0]) + 1.0) / 2.0

        # Create the offset x,y and z values for the grid
        if any([phi, theta, psi]):
            # Rotate the points to the expected coordinate system.
            cphi = np.cos(phi)
            sphi = np.sin(phi)
            ctheta = np.cos(theta)
            stheta = np.sin(theta)
            cpsi = np.cos(psi)
            spsi = np.sin(psi)

            mat = np.array([[cpsi * cphi - ctheta * sphi * spsi,
                             cpsi * sphi + ctheta * cphi * spsi,
                             spsi * stheta],
                            [-spsi * cphi - ctheta * sphi * cpsi,
                             -spsi * sphi + ctheta * cphi * cpsi,
                             cpsi * stheta],
                            [stheta * sphi,
                             -stheta * cphi,
                             ctheta]])

            # Calculate the points that could possibly be inside the volume
            # Since the points are rotated, we cannot do anything directional
            # without more logic

            max_radius = np.sqrt(
                np.abs(mat).dot([a_squared, b_squared, c_squared]))
            idx, shapes = _getshapes_3d(center, max_radius, space.shape)

            subgrid = [g[idi] for g, idi in zip(grid, shapes)]
            offset_points = [vec * (xi - x0i)[..., np.newaxis]
                             for xi, vec, x0i in zip(subgrid,
                                                     mat.T,
                                                     [x0, y0, z0])]
            rotated = offset_points[0] + offset_points[1] + offset_points[2]
            np.square(rotated, out=rotated)
            radius = np.dot(rotated, scales)
        else:
            # Calculate the points that could possibly be inside the volume
            max_radius = np.sqrt([a_squared, b_squared, c_squared])
            idx, shapes = _getshapes_3d(center, max_radius, space.shape)

            subgrid = [g[idi] for g, idi in zip(grid, shapes)]
            squared_dist = [ai * (xi - x0i) ** 2
                            for xi, ai, x0i in zip(subgrid,
                                                   scales,
                                                   [x0, y0, z0])]

            # Parentheses to get best order for broadcasting
            radius = squared_dist[0] + (squared_dist[1] + squared_dist[2])

        # Find the pixels within the ellipse
        inside = radius <= 1

        # Add the ellipse intensity to those pixels
        p[idx][inside] += intensity

    return space.element(p)


def ellipse_phantom(space, ellipses):
    """Return a phantom given by ellipses.

    See Also
    --------
    ellipse_phantom_2d : The 2d implementation
    ellipse_phantom_3d : The 3d implementation
    """

    if space.ndim == 2:
        return ellipse_phantom_2d(space, ellipses)
    elif space.ndim == 3:
        return ellipse_phantom_3d(space, ellipses)
    else:
        raise ValueError('dimension not 2 or 3, no phantom available')


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
