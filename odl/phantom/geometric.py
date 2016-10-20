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

"""Phantoms given by simple geometric objects such as cubes or spheres."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np

__all__ = ('cuboid', 'ellipse_phantom', 'indicate_proj_axis')


def cuboid(space, min_pt=None, max_pt=None):
    """Rectangular cuboid.

    Parameters
    ----------
    space : `DiscretizedSpace`
        Discretized space in which the phantom is supposed to be created.
    min_pt : array-like of shape ``(space.ndim,)``, optional
        Lower left corner of the cuboid. If ``None`` is given, a quarter
        of the extent from ``space.min_pt`` towards the inside is chosen.
    min_pt : array-like of shape ``(space.ndim,)``, optional
        Upper right corner of the cuboid. If ``None`` is given, ``min_pt``
        plus half the extent is chosen.

    Returns
    -------
    phantom : `DiscretizedSpaceElement`
        The generated cuboid phantom in ``space``.

    Examples
    --------
    If both ``min_pt`` and ``max_pt`` are omitted, the cuboid lies in the
    middle of the space domain and extends halfway towards all sides:

    >>> space = odl.uniform_discr([0, 0], [1, 1], [4, 6])
    >>> print(odl.phantom.cuboid(space))
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
     [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    By specifying the corners, the cuboid can be arbitrarily shaped:

    >>> print(odl.phantom.cuboid(space, [0.25, 0], [0.75, 0.5]))
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
     [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    """
    dom_min_pt = np.asarray(space.domain.min())
    dom_max_pt = np.asarray(space.domain.max())

    if min_pt is None:
        min_pt = dom_min_pt * 0.75 + dom_max_pt * 0.25
    if max_pt is None:
        max_pt = dom_min_pt * 0.25 + dom_max_pt * 0.75

    min_pt = np.atleast_1d(min_pt)
    max_pt = np.atleast_1d(max_pt)

    if min_pt.shape != (space.ndim,):
        raise ValueError('shape of `min_pt` must be {}, got {}'
                         ''.format((space.ndim,), min_pt.shape))
    if max_pt.shape != (space.ndim,):
        raise ValueError('shape of `max_pt` must be {}, got {}'
                         ''.format((space.ndim,), max_pt.shape))

    def phantom(x):
        result = True

        for xi, xmin, xmax in zip(x, min_pt, max_pt):
            result = (result &
                      np.less_equal(xmin, xi) & np.less_equal(xi, xmax))
        return result

    return space.element(phantom)


def indicate_proj_axis(space, scale_structures=0.5):
    """Phantom indicating along which axis it is projected.

    The number (n) of rectangles in a parallel-beam projection along a main
    axis (0, 1, or 2) indicates the projection to be along the (n-1)the
    dimension.

    Parameters
    ----------
    space : `DiscretizedSpace`
        Discretized space in which the phantom is supposed to be created
    scale_structures : positive float in (0, 1]
        Scales objects (cube, cuboids)

    Returns
    -------
    phantom : ``space`` element
        Projection helper phantom in ``space``.

    Examples
    --------
    >>> space = odl.uniform_discr([0] * 3, [1] * 3, [8, 8, 8])
    >>> phan = indicate_proj_axis(space).asarray()
    >>> print(np.sum(phan, 0))
    [[ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  3.  3.  0.  0.  0.]
     [ 0.  0.  0.  3.  3.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]]
    >>> print(np.sum(phan, 1))
    [[ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  2.  2.  0.  0.  0.]
     [ 0.  0.  0.  2.  2.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  1.  1.  0.  0.  0.]
     [ 0.  0.  0.  1.  1.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]]
    >>> print(np.sum(phan, 2))
    [[ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  2.  2.  0.  0.  0.]
     [ 0.  0.  0.  2.  2.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  2.  0.  0.  0.]
     [ 0.  0.  0.  2.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]]
    """
    if not 0 < scale_structures <= 1:
        raise ValueError('`scale_structures` ({}) is not in (0, 1]'
                         ''.format(scale_structures))

    assert space.ndim == 3

    shape = space.shape
    phan = np.zeros(shape)
    shape = np.array(shape) - 1
    cen = np.round(0.5 * shape)
    dx = np.floor(scale_structures * 0.25 * shape)
    dx[dx == 0] = 1

    # cube of size 2 * dx
    x0 = (cen - 3 * dx)[0]
    x, y, z = cen - 1 * dx
    phan[x0:x, y:-y, z:-z] = 1

    # 1st cuboid of size (dx[0], dx[1], 2 * dx[2])
    x0 = (cen + 1 * dx)[1]
    x1 = (cen + 2 * dx)[1]
    y0 = cen[1]
    z = (cen - dx)[2]
    phan[x0:x1, y0:-y, z:-z] = 1

    # 2nd cuboid of (dx[0], dx[1], 2 * dx[2]) touching the first diagonally
    # at a long edge
    x0 = (cen + 2 * dx)[1]
    x1 = (cen + 3 * dx)[1]
    y1 = cen[1]
    z = (cen - dx)[2]
    phan[x0:x1, y:y1, z:-z] = 1

    return space.element(phan)


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


def _ellipse_phantom_2d(space, ellipses):
    """Create a phantom of ellipses in 2d space.

    Parameters
    ----------
    space : `DiscreteLp`
        Space the phantom should be generated in.
    ellipses : list of lists
        Each row should contain:
        'value', 'axis_1', 'axis_2', 'center_x', 'center_y', 'rotation'
        The ellipses should be contained the he rectangle [-1, -1] x [1, 1].

    Returns
    -------
    phantom : ``space`` element
        2D ellipse phantom in ``space``.

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
        assert len(ellip) == 6

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


def _ellipse_phantom_3d(space, ellipses):
    """Create an ellipse phantom in 3d space.

    Parameters
    ----------
    space : `DiscreteLp`
        Space the phantom should be generated in.
    ellipses : list of lists
        Each row should contain:
        'value', 'axis_1', 'axis_2', 'axis_3',
        'center_x', 'center_y', 'center_z',
        'rotation_phi', 'rotation_theta', 'rotation_psi'
        The ellipses should be contained the he rectangle
        [-1, -1, -1] x [1, 1, 1].

    Returns
    -------
    phantom : ``space`` element
        3D ellipse phantom in ``space``.

    See Also
    --------
    shepp_logan : The typical use-case for this function.
    """

    # Blank image
    p = np.zeros(space.shape)

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
        assert len(ellip) == 10

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

    Parameters
    ----------
    space : `DiscreteLp`
        Space in which the phantom is created, must be 2- or 3-dimensional.
    ellipses : sequence of sequences
        If ``space`` is 2-dimensional each row should contain:

        'value', 'axis_1', 'axis_2', 'center_x', 'center_y', 'rotation'

        If ``space`` is 3-dimensional each row should contain:

        'value', 'axis_1', 'axis_2', 'axis_3',
        'center_x', 'center_y', 'center_z',
        'rotation_phi', 'rotation_theta', 'rotation_psi'

        The ellipses need to be given such that the ellipses fall in the
        rectangle [-1, -1] x [1, 1] or equivalent in 3d.

    Notes
    -----
    The phantom is created by adding the values of each ellipse. The ellipses
    are defined by a center point (center_x, center_y, [center_z]) and the
    length of its principial axes (axis_1, axis_2, [axis_2]) and euler angles.

    This function is heavily optimized, achieving runtimes about 20 times
    faster than "trivial" implementations. It is therefore recommended to use
    it in all phantoms where applicable.

    The main optimization is that it only considers a subset of all the
    points when updating for each ellipse. It does this by first finding
    a subset of points that could possibly be inside the ellipse. This
    optimization is very good for "spherical" ellipsoids, but not so
    much for elongated or rotated ones.

    It also does calculations wherever possible on the meshgrid instead of
    individual points.

    Examples
    --------
    Create a circle with a smaller circle inside:

    >>> space = odl.uniform_discr([-1, -1], [1, 1], [5, 5])
    >>> ellipses = [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    ...             [1.0, 0.6, 0.6, 0.0, 0.0, 0.0]]
    >>> print(ellipse_phantom(space, ellipses))
    [[0.0, 0.0, 1.0, 0.0, 0.0],
     [0.0, 1.0, 2.0, 1.0, 0.0],
     [1.0, 2.0, 2.0, 2.0, 1.0],
     [0.0, 1.0, 2.0, 1.0, 0.0],
     [0.0, 0.0, 1.0, 0.0, 0.0]]

    See Also
    --------
    odl.phantom.transmission.shepp_logan : Classical Shepp-Logan phantom,
        typically used for transmission imaging
    odl.phantom.transmission.shepp_logan_ellipses : Ellipses for the
        Shepp-Logan phantom
    """

    if space.ndim == 2:
        return _ellipse_phantom_2d(space, ellipses)
    elif space.ndim == 3:
        return _ellipse_phantom_3d(space, ellipses)
    else:
        raise ValueError('dimension not 2 or 3, no phantom available')


if __name__ == '__main__':
    # Show the phantoms
    import odl

    # cuboid 1D
    discr = odl.uniform_discr(-1, 1, 300)
    cuboid(discr).show('cuboid 1d')

    # cuboid 2D
    discr = odl.uniform_discr([-1, -1], [1, 1], [300, 300])
    cuboid(discr).show('cuboid 2d')

    # cuboid 3D
    discr = odl.uniform_discr([-1, -1, -1], [1, 1, 1], [300, 300, 300])
    cuboid(discr).show('cuboid 3d')
    indicate_proj_axis(discr).show('indicate_proj_axis 3d')

    # ellipse phantom 2D
    discr = odl.uniform_discr([-1, -1], [1, 1], [300, 300])
    ellipses = [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.6, 0.6, 0.0, 0.0, 0.0]]
    ellipse_phantom(discr, ellipses).show('ellipse phantom 2d')

    # ellipse phantom 3D
    discr = odl.uniform_discr([-1, -1, -1], [1, 1, 1], [300, 300, 300])
    ellipses = [[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.6, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    ellipse_phantom(discr, ellipses).show('ellipse phantom 3d')

    # Run also the doctests
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
