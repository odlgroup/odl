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


def _shepp_logan_ellipse():
    # Modified Shepp Logan
    return [[   1, .6900, .9200,    0,      0,   0          ],
            [-.80, .6624, .8740,    0, -.0184,   0          ],
            [-.20, .1100, .3100,  .22,      0, -18*np.pi/180],
            [-.20, .1600, .4100, -.22,      0,  18*np.pi/180],
            [ .10, .2100, .2500,    0,  .3500,   0          ],
            [ .10, .0460, .0460,    0,  .1000,   0          ],
            [ .10, .0460, .0460,    0, -.1000,   0          ],
            [ .10, .0460, .0230, -.08, -.6050,   0          ],
            [ .10, .0230, .0230,    0, -.6060,   0          ],
            [ .10, .0230, .0460,  .06, -.6050,   0          ]]


def shepp_logan(space):
    """Create a shepp logan phantom in space."""
    ellipses = _shepp_logan_ellipse()

    # Blank image
    p = np.zeros(space.size)

    # Create the pixel grid
    points = space.points()

    for ellip in ellipses:
        I = ellip[0]
        a2 = ellip[1]**2
        b2 = ellip[2]**2
        x0 = ellip[3]
        y0 = ellip[4]
        phi = ellip[5]

        # Create the offset x and y values for the grid
        offset_points = points - [x0, y0]

        cos_p = np.cos(phi)
        sin_p = np.sin(phi)

        # Find the pixels within the ellipse
        scales = [1/a2, 1/b2]
        mat = [[cos_p, sin_p],
               [-sin_p, cos_p]]
        radius = np.dot(scales, np.dot(mat, offset_points.T)**2)
        inside = radius <= 1

        # Add the ellipse intensity to those pixels
        p[inside] += I

    return space.element(p)
