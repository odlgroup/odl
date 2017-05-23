# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Miscellaneous phantoms that do not fit in other categories."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np


__all__ = ('submarine', 'text')


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
        halfaxes = np.array([0.4, 0.14]) * space.domain.extent
        center = np.array([0.6, 0.3]) * space.domain.extent
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
        xlower = np.array([0.56, 0.4]) * space.domain.extent
        xlower += space.domain.min()
        xupper = np.array([0.76, 0.6]) * space.domain.extent
        xupper += space.domain.min()

        out = np.ones_like(x[0])
        for xi, low, upp in zip(x, xlower, xupper):
            length = upp - low
            out = out * (logistic((xi - low) / length, taper) *
                         logistic((upp - xi) / length, taper))
        return out

    out = space.element(blurred_ellipse)
    out += space.element(blurred_rect)
    return out.ufuncs.minimum(1, out=out)


def _submarine_2d_nonsmooth(space):
    """Return a 2d nonsmooth 'submarine' phantom."""

    def ellipse(x):
        """Characteristic function of an ellipse.

        If ``space.domain`` is a rectangle ``[0, 1] x [0, 1]``,
        the ellipse is centered at ``(0.6, 0.3)`` and has half-axes
        ``(0.4, 0.14)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.4, 0.14]) * space.domain.extent
        center = np.array([0.6, 0.3]) * space.domain.extent
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
        xlower = np.array([0.56, 0.4]) * space.domain.extent
        xlower += space.domain.min()
        xupper = np.array([0.76, 0.6]) * space.domain.extent
        xupper += space.domain.min()

        out = np.ones_like(x[0])
        for xi, low, upp in zip(x, xlower, xupper):
            out = out * ((xi >= low) & (xi <= upp))
        return out

    out = space.element(ellipse)
    out += space.element(rect)
    return out.ufuncs.minimum(1, out=out)


def text(space, text, font='arial', scale=0.8, inverted=True):
    """Create phantom of text.

    The text is represented by an scalar image taking values in [0, 1].
    Depending on the choice of font, the text may or may not be anti-aliased.
    anti-aliased text can take any value between 0 and 1, while
    non-anti-aliased text takes values exclusively in {0, 1}.

    This method requires the PIL package.

    Parameters
    ----------
    space : `DiscreteLp`
        Discretized space in which the phantom is supposed to be created.
        Must be two-dimensional.
    text : str
        The text that should be written in the phantom.
    font : str, optional
        The font that should be used to write the text. Available options are
        platform dependent.
    scale : float, optional
        Scaling of the text. 1.0 indicates that the phantom should occupy all
        of the space along its largest dimension.
    inverted : bool, optional
        If the phantoms should be given in inverted style, i.e. white on black.

    Returns
    -------
    phantom : ``space`` element
        The text phantom in ``space``.
    """
    from PIL import Image, ImageDraw, ImageFont

    if space.ndim != 2:
        raise ValueError('`space` must be two-dimensional')

    text = str(text)

    # Figure out what font size we should use by creating a very fine font
    # and calculating its size
    init_size = 1000
    init_pil_font = ImageFont.truetype(font + ".ttf", size=init_size,
                                       encoding="unic")
    init_text_width, init_text_height = init_pil_font.getsize(text)

    # True size is given by how much too large (or small) the example was
    size = scale * init_size * min([space.shape[0] / init_text_width,
                                    space.shape[1] / init_text_height])
    size = int(size)

    # Create font
    pil_font = ImageFont.truetype(font + ".ttf", size=size,
                                  encoding="unic")
    text_width, text_height = pil_font.getsize(text)

    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', space.shape, (255, 255, 255))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = ((space.shape[0] - text_width) // 2,
              (space.shape[1] - text_height) // 2)
    white = "#000000"
    draw.text(offset, text, font=pil_font, fill=white)

    # Convert the canvas into an array with values in [0, 1]
    arr = np.asarray(canvas)
    arr = np.sum(arr, -1)
    arr = arr / np.max(arr)
    arr = np.rot90(arr, -1)

    if inverted:
        arr = 1 - arr

    return space.element(arr)

if __name__ == '__main__':
    # Show the phantoms
    import odl

    space = odl.uniform_discr([-1, -1], [1, 1], [300, 300])
    submarine(space, smooth=False).show('submarine smooth=False')
    submarine(space, smooth=True).show('submarine smooth=True')
    submarine(space, smooth=True, taper=50).show('submarine taper=50')

    text(space, text='phantom').show('phantom')

    # Run also the doctests
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
