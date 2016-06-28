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

__all__ = ('cuboid', 'indicate_proj_axis')


def cuboid(discr_space, begin=None, end=None):
    """Rectangular cuboid.

    Parameters
    ----------
    discr_space : `DiscretizedSpace`
        Discretized space in which the phantom is supposed to be created.
    begin : array-like of length ``discr_space.ndim``
        The lower left corner of the cuboid within the space.
        Default: A quarter of the volume from the minimum corner
    end : array-like of length ``discr_space.ndim``
        The upper right corner of the cuboid within the space.
        Default: A quarter of the volume from the maximum corner

    Returns
    -------
    phantom : `LinearSpaceVector`
        Returns an element in ``discr_space``

    Examples
    --------
    >>> import odl
    >>> space = odl.uniform_discr(0, 1, 6, dtype='float32')
    >>> print(cuboid(space, 0.5, 1))
    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    >>> space = odl.uniform_discr([0, 0], [1, 1], [4, 6], dtype='float32')
    >>> print(cuboid(space, [0.25, 0], [0.75, 0.5]))
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
     [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    """
    minp = np.asarray(discr_space.domain.min())
    maxp = np.asarray(discr_space.domain.max())

    if begin is None:
        begin = minp * 0.75 + maxp * 0.25
    if end is None:
        end = begin * 0.25 + maxp * 0.75

    begin = np.atleast_1d(begin)
    end = np.atleast_1d(end)

    assert begin.size == discr_space.ndim
    assert end.size == discr_space.ndim

    def phan(x):
        result = True

        for xp, minv, maxv in zip(x, begin, end):
            result = (result &
                      np.less_equal(minv, xp) & np.less_equal(xp, maxv))
        return result

    return discr_space.element(phan)


def indicate_proj_axis(discr_space, scale_structures=0.5):
    """Phantom indicating along which axis it is projected.

    The number (n) of rectangles in a parallel-beam projection along a main
    axis (0, 1, or 2) indicates the projection to be along the (n-1)the
    dimension.

    Parameters
    ----------
    discr_space : `DiscretizedSpace`
        Discretized space in which the phantom is supposed to be created
    scale_structures : positive `float` in (0, 1]
        Scales objects (cube, cuboids)

    Returns
    -------
    phantom : `LinearSpaceVector`
        Returns an element in ``discr_space``

    Examples
    --------
    >>> import odl
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

    assert discr_space.ndim == 3

    shape = discr_space.shape
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

    return discr_space.element(phan)

if __name__ == '__main__':
    # Show the phantoms
    import odl

    # 1D
    discr = odl.uniform_discr(-1, 1, 300)
    cuboid(discr).show('cuboid 1d')

    # 2D
    discr = odl.uniform_discr([-1, -1], [1, 1], [300, 300])
    cuboid(discr).show('cuboid 2d')

    # 3D
    discr = odl.uniform_discr([-1, -1, -1], [1, 1, 1], [300, 300, 300])
    cuboid(discr).show('cuboid 3d')
    indicate_proj_axis(discr).show('indicate_proj_axis 3d')

    # Run also the doctests
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
