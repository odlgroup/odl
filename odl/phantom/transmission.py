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

from odl.discr import DiscreteLp
from odl.phantom.geometric import ellipse_phantom
import numpy as np


__all__ = ('shepp_logan_ellipses', 'shepp_logan', 'forbild')


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

    # Add minimal numbers to ensure that the result is nowhere negative.
    # This is needed due to numerical issues.
    intensities[2] += 5e-17
    intensities[3] += 5e-17

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
        If ``space.shape`` is 1 in an axis, a corresponding slice of the
        phantom is created.
    modified : `bool`, optional
        True if the modified Shepp-Logan phantom should be given.
        The modified phantom has greatly amplified contrast to aid
        visualization.

    See Also
    --------
    forbild : Similar phantom but with more complexity. Only supports 2d.
    odl.phantom.geometric.defrise : Geometry test phantom
    shepp_logan_ellipses : Get the parameters that define this phantom
    odl.phantom.geometric.ellipse_phantom :
        Function for creating arbitrary ellipse phantoms

    References
    ----------
    .. _Shepp-Logan phantom: en.wikipedia.org/wiki/Shepp–Logan_phantom
    """
    ellipses = shepp_logan_ellipses(space.ndim, modified)

    return ellipse_phantom(space, ellipses)


def _analytical_forbild_phantom(resolution, ear):
    """Analytical description of FORBILD phantom.

    Parameters
    ----------
    resolution : bool
        If ``True``, insert a small resolution test pattern to the left.
    ear : bool
        If ``True``, insert an ear-like structure to the right.
    """
    sha = 0.2 * np.sqrt(3)
    y016b = -14.294530834372887
    a16b = 0.443194085308632
    b16b = 3.892760834372886

    E = [[-4.7, 4.3, 1.79989, 1.79989, 0, 0.010, 0],  # 1
         [4.7, 4.3, 1.79989, 1.79989, 0, 0.010, 0],  # 2
         [-1.08, -9, 0.4, 0.4, 0, 0.0025, 0],  # 3
         [1.08, -9, 0.4, 0.4, 0, -0.0025, 0],  # 4
         [0, 0, 9.6, 12, 0, 1.800, 0],  # 5
         [0, 8.4, 1.8, 3.0, 0, -1.050, 0],  # 7
         [1.9, 5.4, 0.41633, 1.17425, -31.07698, 0.750, 0],  # 8
         [-1.9, 5.4, 0.41633, 1.17425, 31.07698, 0.750, 0],  # 9
         [-4.3, 6.8, 1.8, 0.24, -30, 0.750, 0],  # 10
         [4.3, 6.8, 1.8, 0.24, 30, 0.750, 0],  # 11
         [0, -3.6, 1.8, 3.6, 0, -0.005, 0],  # 12
         [6.39395, -6.39395, 1.2, 0.42, 58.1, 0.005, 0],  # 13
         [0, 3.6, 2, 2, 0, 0.750, 4],  # 14
         [0, 9.6, 1.8, 3.0, 0, 1.800, 4],  # 15
         [0, 0, 9.0, 11.4, 0, 0.750, 3],  # 16a
         [0, y016b, a16b, b16b, 0, 0.750, 1],  # 16b
         [0, 0, 9.0, 11.4, 0, -0.750, ear],  # 6
         [9.1, 0, 4.2, 1.8, 0, 0.750, 1]]  # R_ear
    E = np.array(E)

    # generate the air cavities in the right ear
    cavity1 = np.arange(8.8, 5.6, -0.4)[:, None]
    cavity2 = np.zeros([9, 1])
    cavity3_7 = np.ones([53, 1]) * [0.15, 0.15, 0, -1.800, 0]

    for j in range(1, 4):
        kj = 8 - 2 * int(np.floor(j / 3))
        dj = 0.2 * int(np.mod(j, 2))

        cavity1 = np.vstack((cavity1,
                             cavity1[0:kj] - dj,
                             cavity1[0:kj] - dj))
        cavity2 = np.vstack((cavity2,
                             j * sha * np.ones([kj, 1]),
                             -j * sha * np.ones([kj, 1])))

    E_cavity = np.hstack((cavity1, cavity2, cavity3_7))

    # generate the left ear (resolution pattern)
    x0 = -7.0
    y0 = -1.0
    d0_xy = 0.04

    d_xy = [0.0357, 0.0312, 0.0278, 0.0250]
    ab = 0.5 * np.ones([5, 1]) * d_xy
    ab = ab.T.ravel()[:, None] * np.ones([1, 4])
    abr = ab.T.ravel()[:, None]

    leftear4_7 = np.hstack([abr, abr, np.ones([80, 1]) * [0, 0.75, 0]])

    x00 = np.zeros([0, 1])
    y00 = np.zeros([0, 1])
    for i in range(1, 5):
        y00 = np.vstack((y00,
                         (y0 + np.arange(0, 5) * 2 * d_xy[i - 1])[:, None]))
        x00 = np.vstack((x00,
                         (x0 + 2 * (i - 1) * d0_xy) * np.ones([5, 1])))

    x00 = x00 * np.ones([1, 4])
    x00 = x00.T.ravel()[:, None]
    y00 = np.vstack([y00, y00 + 12 * d0_xy,
                     y00 + 24 * d0_xy, y00 + 36 * d0_xy])

    leftear = np.hstack([x00, y00, leftear4_7])
    C = [[1.2, 1.2, 0.27884, 0.27884, 0.60687, 0.60687, 0.2,
          0.2, -2.605, -2.605, -10.71177, y016b + 10.71177, 8.88740, -0.21260],
         [0, 180, 90, 270, 90, 270, 0,
          180, 15, 165, 90, 270, 0, 0]]
    C = np.array(C)

    if not resolution and not ear:
        phantomE = E[:17, :]
        phantomC = C[:, :12]
    elif not resolution and ear:
        phantomE = np.vstack([E, E_cavity])
        phantomC = C
    elif resolution and not ear:
        phantomE = np.vstack([leftear, E[:17, :]])
        phantomC = C[:, :12]
    else:
        phantomE = np.vstack([leftear, E, E_cavity])
        phantomC = C

    return phantomE, phantomC


def forbild(space, resolution=False, ear=True):
    """Standard `FORBILD phantom` in 2 dimensions.

    The FORBILD phantom is intended for testing CT algorithms and is intended
    to be similar to a human head.

    Parameters
    ----------
    space : `DiscreteLp`
        The space in which the phantom should be corrected. Needs to be two-
        dimensional.
    resolution : bool, optional
        If ``True``, insert a small resolution test pattern to the left.
    ear : bool, optional
        If ``True``, insert an ear-like structure to the right.

    Returns
    -------
    forbild : ``space``-element
        FORBILD phantom discretized by ``space``.

    See Also
    --------
    shepp_logan : A simpler phantom for similar purposes, also working in 3d.

    References
    ----------
    .. _FORBILD phantom: www.imp.uni-erlangen.de/phantoms/head/head.html
    .. _algorithm: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3426508/
    """
    def transposeravel(arr):
        """Implement MATLAB's ``transpose(arr(:))``."""
        return arr.T.ravel()

    if not isinstance(space, DiscreteLp):
        raise TypeError('`space` must be a `DiscreteLp`')
    if not space.ndim == 2:
        raise TypeError('`space` must be two-dimensional')

    # Create analytic description of phantom
    phantomE, phantomC = _analytical_forbild_phantom(resolution, ear)

    # Rescale points to the default grid.
    # The forbild phantom is defined on [-12.8, 12.8] x [-12.8, 12.8]
    xcoord, ycoord = space.points().T
    xcoord = (xcoord - np.min(xcoord)) / (np.max(xcoord) - np.min(xcoord))
    xcoord = 25.8 * xcoord - 12.8
    ycoord = (ycoord - np.min(ycoord)) / (np.max(ycoord) - np.min(ycoord))
    ycoord = 25.8 * ycoord - 12.8

    # Compute the phantom values in each voxel
    image = np.zeros(space.size)
    nclipinfo = 0
    for k in range(phantomE.shape[0]):
        # Handle elliptic bounds
        Vx0 = np.array([transposeravel(xcoord) - phantomE[k, 0],
                        transposeravel(ycoord) - phantomE[k, 1]])
        D = np.array([[1 / phantomE[k, 2], 0],
                      [0, 1 / phantomE[k, 3]]])
        phi = np.deg2rad(phantomE[k, 4])
        Q = np.array([[np.cos(phi), np.sin(phi)],
                      [-np.sin(phi), np.cos(phi)]])
        f = phantomE[k, 5]
        nclip = int(phantomE[k, 6])
        equation1 = np.sum(D.dot(Q).dot(Vx0) ** 2, axis=0)
        i = (equation1 <= 1.0)

        # Handle clipping surfaces
        for _ in range(nclip):  # note: nclib can be 0
            d = phantomC[0, nclipinfo]
            psi = np.deg2rad(phantomC[1, nclipinfo])
            equation2 = np.array([np.cos(psi), np.sin(psi)]).dot(Vx0)
            i &= (equation2 < d)
            nclipinfo += 1

        image[i] += f

    return space.element(image)


if __name__ == '__main__':
    # Show the phantoms
    import odl

    # 2D
    discr = odl.uniform_discr([-1, -1], [1, 1], [1000, 1000])
    shepp_logan(discr, modified=True).show('shepp_logan 2d modified=True')
    shepp_logan(discr, modified=False).show('shepp_logan 2d modified=False')
    forbild(discr).show('FORBILD 2d', clim=[1.035, 1.065])

    # 3D
    discr = odl.uniform_discr([-1, -1, -1], [1, 1, 1], [300, 300, 300])
    shepp_logan(discr, modified=True).show('shepp_logan 3d modified=True')
    shepp_logan(discr, modified=False).show('shepp_logan 3d modified=False')

    # Run also the doctests
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
