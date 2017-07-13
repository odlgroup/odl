# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Phantoms given by simple geometric objects such as cubes or spheres."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np

__all__ = ('cuboid', 'defrise', 'ellipsoid_phantom', 'indicate_proj_axis',
           'smooth_cuboid', 'tgv_phantom')


def cuboid(space, min_pt=None, max_pt=None):
    """Rectangular cuboid.

    Parameters
    ----------
    space : `DiscretizedSpace`
        Discretized space in which the phantom is supposed to be created.
    min_pt : array-like of shape ``(space.ndim,)``, optional
        Lower left corner of the cuboid. If ``None`` is given, a quarter
        of the extent from ``space.min_pt`` towards the inside is chosen.
    max_pt : array-like of shape ``(space.ndim,)``, optional
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


def defrise(space, nellipses=8, alternating=False):
    """Phantom with regularily spaced ellipses.

    This phantom is often used to verify cone-beam algorithms.

    Parameters
    ----------
    space : `DiscretizedSpace`
        Discretized space in which the phantom is supposed to be created.
        Needs to be 2d or 3d.
    nellipses : int, optional
        Number of ellipses. If more ellipses are used, each ellipse becomes
        thinner.
    alternating : bool, optional
        True if the ellipses should have alternating densities (+1, -1),
        otherwise all ellipses have value +1.

    Returns
    -------
    phantom : ``space`` element
        The generated phantom in ``space``.

    See Also
    --------
    odl.phantom.transmission.shepp_logan
    """
    ellipses = defrise_ellipses(space.ndim, nellipses=nellipses,
                                alternating=alternating)

    return ellipsoid_phantom(space, ellipses)


def defrise_ellipses(ndim, nellipses=8, alternating=False):
    """Ellipses for the standard Defrise phantom in 2 or 3 dimensions.

    Parameters
    ----------
    ndim : {2, 3}
        Dimension of the space the ellipses should be in.
    nellipses : int, optional
        Number of ellipses. If more ellipses are used, each ellipse becomes
        thinner.
    alternating : bool, optional
        True if the ellipses should have alternating densities (+1, -1),
        otherwise all ellipses have value +1.

    See Also
    --------
    odl.phantom.geometric.ellipsoid_phantom :
        Function for creating arbitrary ellipsoids phantoms
    shepp_logan_ellipsoids
    """
    ellipses = []
    if ndim == 2:
        for i in range(nellipses):
            if alternating:
                value = (-1.0 + 2.0 * (i % 2))
            else:
                value = 1.0

            axis_1 = 0.5
            axis_2 = 0.5 / (nellipses + 1)
            center_x = 0.0
            center_y = -1 + 2.0 / (nellipses + 1.0) * (i + 1)
            rotation = 0
            ellipses.append(
                [value, axis_1, axis_2, center_x, center_y, rotation])
    elif ndim == 3:
        for i in range(nellipses):
            if alternating:
                value = (-1.0 + 2.0 * (i % 2))
            else:
                value = 1.0

            axis_1 = axis_2 = 0.5
            axis_3 = 0.5 / (nellipses + 1)
            center_x = center_y = 0.0
            center_z = -1 + 2.0 / (nellipses + 1.0) * (i + 1)
            rotation_phi = rotation_theta = rotation_psi = 0

            ellipses.append(
                [value, axis_1, axis_2, axis_3,
                 center_x, center_y, center_z,
                 rotation_phi, rotation_theta, rotation_psi])

    return ellipses


def indicate_proj_axis(space, scale_structures=0.5):
    """Phantom indicating along which axis it is projected.

    The number (n) of rectangles in a parallel-beam projection along a main
    axis (0, 1, or 2) indicates the projection to be along the (n-1)the
    dimension.

    Parameters
    ----------
    space : `DiscreteLp`
        Space in which the phantom is supposed to be created, must be
        2- or 3-dimensional.
    scale_structures : positive float in (0, 1], optional
        Scales objects (cube, cuboids)

    Returns
    -------
    phantom : ``space`` element
        Projection helper phantom in ``space``.

    Examples
    --------
    Phantom in 2D space:

    >>> np.set_printoptions(edgeitems=4)  # make numpy print whole arrays
    >>> space = odl.uniform_discr([0, 0], [1, 1], shape=(8, 8))
    >>> phantom = indicate_proj_axis(space).asarray()
    >>> print(phantom)
    [[ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  1.  1.  0.  0.  0.]
     [ 0.  0.  0.  1.  1.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  1.  0.  0.  0.]
     [ 0.  0.  0.  1.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]]

    >>> space = odl.uniform_discr([0, 0, 0], [1, 1, 1], shape=(8, 8, 8))
    >>> phantom = indicate_proj_axis(space).asarray()
    >>> np.set_printoptions(edgeitems=4)  # make numpy print whole arrays
    >>> print(np.sum(phantom, 0))
    [[ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  3.  3.  0.  0.  0.]
     [ 0.  0.  0.  3.  3.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]]
    >>> print(np.sum(phantom, 1))
    [[ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  2.  2.  0.  0.  0.]
     [ 0.  0.  0.  2.  2.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  1.  1.  0.  0.  0.]
     [ 0.  0.  0.  1.  1.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.]]
    >>> print(np.sum(phantom, 2))
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

    assert space.ndim in (2, 3)

    shape = space.shape
    phan = np.zeros(shape)
    shape = np.array(shape) - 1
    cen = np.round(0.5 * shape)
    dx = np.floor(scale_structures * 0.25 * shape)
    dx[dx == 0] = 1

    # cube of size 2 * dx, offset in x axis, symmetric in others
    ix0 = int((cen - 3 * dx)[0])
    if space.ndim == 2:
        ix, iy = (cen - 1 * dx).astype(int)
        phan[ix0:ix, iy:-iy] = 1
    elif space.ndim == 3:
        ix, iy, iz = (cen - 1 * dx).astype(int)
        phan[ix0:ix, iy:-iy, iz:-iz] = 1

    # 1st cuboid of size (dx[0], dx[1], 2 * dx[2]), offset in x and y axes,
    # symmetric in z axis
    ix0 = int((cen + 1 * dx)[1])
    ix1 = int((cen + 2 * dx)[1])
    iy0 = int(cen[1])
    if space.ndim == 2:
        phan[ix0:ix1, iy0:-iy] = 1
    elif space.ndim == 3:
        iz = int((cen - dx)[2])
        phan[ix0:ix1, iy0:-iy, iz:-iz] = 1

    # 2nd cuboid of (dx[0], dx[1], 2 * dx[2]) touching the first diagonally
    # at a long edge; offset in x and y axes, symmetric in z axis
    ix0 = int((cen + 2 * dx)[1])
    ix1 = int((cen + 3 * dx)[1])
    iy1 = int(cen[1])
    if space.ndim == 2:
        phan[ix0:ix1, iy:iy1] = 1
    elif space.ndim == 3:
        iz = int((cen - dx)[2])
        phan[ix0:ix1, iy:iy1, iz:-iz] = 1

    return space.element(phan)


def _getshapes_2d(center, max_radius, shape):
    """Calculate indices and slices for the bounding box of a disk."""
    index_mean = shape * center
    index_radius = max_radius / 2.0 * np.array(shape)

    min_idx = np.floor(index_mean - index_radius).astype(int)
    min_idx = np.maximum(min_idx, 0)  # avoid negative indices
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
        Space the phantom should be generated in. If ``space.shape`` is
        1 in an axis, a corresponding slice of the phantom is created.
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
        # Where space.shape = 1, we have minp = maxp, so we set diffi = 1
        # to avoid division by zero. Effectively, this allows constructing
        # a slice of a 2D phantom.
        diffi = (maxp[i] - minp[i]) / 2.0 or 1.0
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

        # Find the points within the ellipse
        inside = radius <= 1

        # Add the ellipse intensity to those points
        p[idx][inside] += intensity

    return space.element(p)


def _getshapes_3d(center, max_radius, shape):
    """Calculate indices and slices for the bounding box of a ball."""
    index_mean = shape * center
    index_radius = max_radius / 2.0 * np.array(shape)

    min_idx = np.floor(index_mean - index_radius).astype(int)
    min_idx = np.maximum(min_idx, 0)  # avoid negative indices
    max_idx = np.ceil(index_mean + index_radius).astype(int)
    idx = [slice(minx, maxx) for minx, maxx in zip(min_idx, max_idx)]
    shapes = [(idx[0], slice(None), slice(None)),
              (slice(None), idx[1], slice(None)),
              (slice(None), slice(None), idx[2])]
    return idx, shapes


def _ellipsoid_phantom_3d(space, ellipsoids):
    """Create an ellipsoid phantom in 3d space.

    Parameters
    ----------
    space : `DiscreteLp`
        Space the phantom should be generated in. If ``space.shape`` is
        1 in an axis, a corresponding slice of the phantom is created.
    ellipsoids : list of lists
        Each row should contain:
        'value', 'axis_1', 'axis_2', 'axis_3',
        'center_x', 'center_y', 'center_z',
        'rotation_phi', 'rotation_theta', 'rotation_psi'
        The ellipsoids should be contained in the rectangle
        [-1, -1, -1] x [1, 1, 1].

    Returns
    -------
    phantom : ``space`` element
        3D ellipsoid phantom in ``space``.

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
        # Where space.shape = 1, we have minp = maxp, so we set diffi = 1
        # to avoid division by zero. Effectively, this allows constructing
        # a slice of a 3D phantom.
        diffi = (maxp[i] - minp[i]) / 2.0 or 1.0
        grid += [(grid_in[i] - meani) / diffi]

    for ellip in ellipsoids:
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

        # Find the points within the ellipse
        inside = radius <= 1

        # Add the ellipse intensity to those points
        p[idx][inside] += intensity

    return space.element(p)


def ellipsoid_phantom(space, ellipsoids):
    """Return a phantom given by ellipsoids.

    Parameters
    ----------
    space : `DiscreteLp`
        Space in which the phantom is created, must be 2- or 3-dimensional.
        If ``space.shape`` is 1 in an axis, a corresponding slice of the
        phantom is created.
    ellipsoids : sequence of sequences
        If ``space`` is 2-dimensional each row should contain:

        'value', 'axis_1', 'axis_2', 'center_x', 'center_y', 'rotation'

        If ``space`` is 3-dimensional each row should contain:

        'value', 'axis_1', 'axis_2', 'axis_3',
        'center_x', 'center_y', 'center_z',
        'rotation_phi', 'rotation_theta', 'rotation_psi'

        The ellipsoids need to be given such that the ellipsoids fall in the
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
    >>> print(ellipsoid_phantom(space, ellipses))
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
    odl.phantom.geometric.defrise_ellipses : Ellipses for the
        Defrise phantom
    """

    if space.ndim == 2:
        return _ellipse_phantom_2d(space, ellipsoids)
    elif space.ndim == 3:
        return _ellipsoid_phantom_3d(space, ellipsoids)
    else:
        raise ValueError('dimension not 2 or 3, no phantom available')


def smooth_cuboid(space, min_pt=None, max_pt=None, axis=0):
    """Rectangular cuboid with smooth variations.

    Parameters
    ----------
    space : `DiscreteLp`
        Discretized space in which the phantom is supposed to be created.
    min_pt : array-like of shape ``(space.ndim,)``, optional
        Lower left corner of the cuboid. If ``None`` is given, a quarter
        of the extent from ``space.min_pt`` towards the inside is chosen.
    max_pt : array-like of shape ``(space.ndim,)``, optional
        Upper right corner of the cuboid. If ``None`` is given, ``min_pt``
        plus half the extent is chosen.
    axis : `int` or sequence of `int`
        Dimension(s) along which the smooth variation should happen.

    Returns
    -------
    phantom : ``space``-element
        The generated cuboid phantom in ``space``.
    """
    dom_min_pt = np.asarray(space.domain.min())
    dom_max_pt = np.asarray(space.domain.max())

    if min_pt is None:
        min_pt = dom_min_pt * 0.75 + dom_max_pt * 0.25
    if max_pt is None:
        max_pt = dom_min_pt * 0.25 + dom_max_pt * 0.75

    min_pt = np.atleast_1d(min_pt)
    max_pt = np.atleast_1d(max_pt)

    axis = np.array(axis, dtype=int, ndmin=1)

    if min_pt.shape != (space.ndim,):
        raise ValueError('shape of `min_pt` must be {}, got {}'
                         ''.format((space.ndim,), min_pt.shape))
    if max_pt.shape != (space.ndim,):
        raise ValueError('shape of `max_pt` must be {}, got {}'
                         ''.format((space.ndim,), max_pt.shape))

    sign = 0
    for i, coord in enumerate(space.meshgrid):
        sign = sign | (coord < min_pt[i]) | (coord > max_pt[i])

    values = 0
    for i in axis:
        coord = space.meshgrid[i]
        extent = (dom_max_pt[i] - dom_min_pt[i])
        values = values + 2 * (coord - dom_min_pt[i]) / extent - 1

    # Properly scale using sign
    sign = (3 * sign - 2) / axis.size

    return space.element(values * sign)


def tgv_phantom(space):
    """Piecewise affine phantom.

    This phantom is taken from [Bre+2010] and includes both linearly varying
    regions and sharp discontinuities. It is designed to work well with
    Total Generalized Variation (TGV) type regularization.

    Parameters
    ----------
    space : `DiscreteLp`, 2 dimensional
        Discretized space in which the phantom is supposed to be created.
        Needs to be two-dimensional.

    Returns
    -------
    phantom : ``space``-element
        The generated phantom in ``space``. Values have range [0, 1].

    Notes
    -----
    The original phantom is given by a specific image. In this implementation,
    we extracted the underlying parameters and the phantom thus works with
    spaces of any shape. Due to this, small variations may occur when compared
    to the original phantom.

    References
    ----------
    [Bre+2010] K. Bredies, K. Kunisch, and T. Pock.
    Total Generalized Variation. SIAM Journal on Imaging Sciences,
    3(3):492–526, Jan. 2010
    """
    # Note that all parameters used here were extracted by hand and may not fit
    # exactly with the reference image.

    if space.ndim != 2:
        raise ValueError('`space.ndim` must be 2, got {}'
                         ''.format(space.ndim))

    y, x = space.meshgrid

    # Use a smooth sigmoid to get some anti-aliasing across edges.
    scale = 0.1 * np.min(space.cell_sides)

    def sigmoid(val):
        val = val / scale
        return 1 / (1 + np.exp(-val))

    # Normalize to [0, 1]
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    y = (y - np.min(y)) / (np.max(y) - np.min(y))

    # Background
    values = -(x + y) / 2

    # Square-ish region
    indicator = np.ones(space.shape)
    indicator *= sigmoid(-(0.015199034981905914 * x - y + 0.13896260554885403))
    indicator *= sigmoid((0.3333333333333323 * y - x + 0.598958333333334))
    indicator *= sigmoid((-2.4193548387096726 * y - x + 2.684979838709672))

    values += indicator * 2 * (x + y - 1)

    # Ellipse part
    x_c = x - 0.71606842360499456
    y_c = y - 0.18357884949910641

    width = 0.55677657235995637
    height = 0.37279391542283741
    phi = 0.62911754900697558

    x_c_rot = (np.cos(phi) * x_c - np.sin(phi) * y_c) / width
    y_c_rot = (np.sin(phi) * x_c + np.cos(phi) * y_c) / height

    indicator = sigmoid(np.sqrt(x_c_rot ** 2 + y_c_rot ** 2) - 1)

    values = indicator * values + 1.5 * (1 - indicator) * (-x - 2 * y + 0.6)

    # Normalize values
    values = (values - np.min(values)) / (np.max(values) - np.min(values))

    return space.element(values)


if __name__ == '__main__':
    # Show the phantoms
    import odl

    # cuboid 1D
    space = odl.uniform_discr(-1, 1, 300)
    cuboid(space).show('cuboid 1d')

    # cuboid 2D
    space = odl.uniform_discr([-1, -1], [1, 1], [300, 300])
    cuboid(space).show('cuboid 2d')

    # smooth cuboid
    smooth_cuboid(space).show('smooth_cuboid x 2d')
    smooth_cuboid(space, axis=[0, 1]).show('smooth_cuboid x-y 2d')

    # TGV phantom
    tgv_phantom(space).show('tgv_phantom')

    # cuboid 3D
    space = odl.uniform_discr([-1, -1, -1], [1, 1, 1], [300, 300, 300])
    cuboid(space).show('cuboid 3d')

    # Indicate proj axis 3D
    indicate_proj_axis(space).show('indicate_proj_axis 3d')

    # ellipsoid phantom 2D
    space = odl.uniform_discr([-1, -1], [1, 1], [300, 300])
    ellipses = [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.6, 0.6, 0.0, 0.0, 0.0]]
    ellipsoid_phantom(space, ellipses).show('ellipse phantom 2d')

    # ellipsoid phantom 3D
    space = odl.uniform_discr([-1, -1, -1], [1, 1, 1], [300, 300, 300])
    ellipsoids = [[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [1.0, 0.6, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    ellipsoid_phantom(space, ellipsoids).show('ellipse phantom 3d')

    # Defrise phantom 2D
    space = odl.uniform_discr([-1, -1], [1, 1], [300, 300])
    defrise(space).show('defrise 2D')

    # Defrise phantom 2D
    space = odl.uniform_discr([-1, -1, -1], [1, 1, 1], [300, 300, 300])
    defrise(space).show('defrise 3D', coords=[0, None, None])

    # Run also the doctests
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
