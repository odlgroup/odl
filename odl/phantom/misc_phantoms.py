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

"""Miscellaneous phantoms that do not fit in other categories."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np


__all__ = ('submarine',)


def submarine(space, smooth=True, taper=20.0):
    """Return a 'submarine' phantom consisting in an ellipsoid and a box.

    Parameters
    ----------
    space : `DiscreteLp`
        Discretized space in which the phantom is supposed to be created.
    smooth : bool, optional
        If ``True``, the boundaries are smoothed out. Otherwise, the
        function steps from 0 to 1 at the boundaries.
    taper : float, optional
        Tapering parameter for the boundary smoothing. Larger values
        mean faster taper, i.e. sharper boundaries.

    Returns
    -------
    phantom : ``space`` element
        The submarine phantom in ``space``.
    """
    if space.ndim == 2:
        if smooth:
            return _submarine_2d_smooth(space, taper)
        else:
            return _submarine_2d_nonsmooth(space)
    else:
        raise ValueError('phantom only defined in 2 dimensions, got {}'
                         ''.format(space.ndim))


def _submarine_2d_smooth(space, taper):
    """Return a 2d smooth 'submarine' phantom."""

    def logistic(x, c):
        """Smoothed step function from 0 to 1, centered at 0."""
        return 1. / (1 + np.exp(-c * x))

    def blurred_ellipse(x):
        """Blurred characteristic function of an ellipse.

        If ``space.domain`` is a rectangle ``[0, 1] x [0, 1]``,
        the ellipse is centered at ``(0.6, 0.3)`` and has half-axes
        ``(0.4, 0.14)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.4, 0.14]) * space.domain.extent()
        center = np.array([0.6, 0.3]) * space.domain.extent()
        center += space.domain.min()

        # Efficiently calculate |z|^2, z = (x - center) / radii
        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        out = np.sqrt(sq_ndist)
        out -= 1
        # Return logistic(taper * (1 - |z|))
        return logistic(out, -taper)

    def blurred_rect(x):
        """Blurred characteristic function of a rectangle.

        If ``space.domain`` is a rectangle ``[0, 1] x [0, 1]``,
        the rect has lower left ``(0.56, 0.4)`` and upper right
        ``(0.76, 0.6)``. For other domains, the values are scaled
        accordingly.
        """
        xlower = np.array([0.56, 0.4]) * space.domain.extent()
        xlower += space.domain.min()
        xupper = np.array([0.76, 0.6]) * space.domain.extent()
        xupper += space.domain.min()

        out = np.ones_like(x[0])
        for xi, low, upp in zip(x, xlower, xupper):
            length = upp - low
            out = out * (logistic((xi - low) / length, taper) *
                         logistic((upp - xi) / length, taper))
        return out

    out = space.element(blurred_ellipse)
    out += space.element(blurred_rect)
    return out.ufunc.minimum(1, out=out)


def _submarine_2d_nonsmooth(space):
    """Return a 2d nonsmooth 'submarine' phantom."""

    def ellipse(x):
        """Characteristic function of an ellipse.

        If ``space.domain`` is a rectangle ``[0, 1] x [0, 1]``,
        the ellipse is centered at ``(0.6, 0.3)`` and has half-axes
        ``(0.4, 0.14)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.4, 0.14]) * space.domain.extent()
        center = np.array([0.6, 0.3]) * space.domain.extent()
        center += space.domain.min()

        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        return np.where(sq_ndist <= 1, 1, 0)

    def rect(x):
        """Characteristic function of a rectangle.

        If ``space.domain`` is a rectangle ``[0, 1] x [0, 1]``,
        the rect has lower left ``(0.56, 0.4)`` and upper right
        ``(0.76, 0.6)``. For other domains, the values are scaled
        accordingly.
        """
        xlower = np.array([0.56, 0.4]) * space.domain.extent()
        xlower += space.domain.min()
        xupper = np.array([0.76, 0.6]) * space.domain.extent()
        xupper += space.domain.min()

        out = np.ones_like(x[0])
        for xi, low, upp in zip(x, xlower, xupper):
            out = out * ((xi >= low) & (xi <= upp))
        return out

    out = space.element(ellipse)
    out += space.element(rect)
    return out.ufunc.minimum(1, out=out)


if __name__ == '__main__':
    # Show the phantoms
    import odl

    space = odl.uniform_discr([-1, -1], [1, 1], [300, 300])
    submarine(space, smooth=False).show('submarine smooth=False')
    submarine(space, smooth=True).show('submarine smooth=True')
    submarine(space, smooth=True, taper=50).show('submarine taper=50')

    # Run also the doctests
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
