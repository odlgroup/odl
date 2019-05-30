# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Phantoms used in emission tomography."""

from __future__ import print_function, division, absolute_import

from odl.phantom.geometric import ellipsoid_phantom
from odl.phantom.phantom_utils import cylinders_from_ellipses


__all__ = ('derenzo_sources',)


def _derenzo_sources_2d():
    """Return ellipse parameters for a 2d Derenzo sources phantom.

    This is a popular phantom in SPECT and PET. It defines the source
    locations and intensities.
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


def derenzo_sources(space, min_pt=None, max_pt=None):
    """Create the PET/SPECT Derenzo sources phantom.

    The Derenzo phantom contains a series of circles of decreasing size.

    In 3d the phantom is simply the 2d phantom extended in the z direction as
    cylinders.

    Parameters
    ----------
    space : `DiscreteLp`
        Space in which the phantom should be created, must be 2- or
        3-dimensional. If ``space.shape`` is 1 in an axis, a corresponding
        slice of the phantom is created (instead of squashing the whole
        phantom into the slice).
    min_pt, max_pt : array-like, optional
        If provided, use these vectors to determine the bounding box of the
        phantom instead of ``space.min_pt`` and ``space.max_pt``.
        It is currently required that ``min_pt >= space.min_pt`` and
        ``max_pt <= space.max_pt``, i.e., shifting or scaling outside the
        original space is not allowed.

        Providing one of them results in a shift, e.g., for ``min_pt``::

            new_min_pt = min_pt
            new_max_pt = space.max_pt + (min_pt - space.min_pt)

        Providing both results in a scaled version of the phantom.

    Returns
    -------
    phantom : ``space`` element
        The Derenzo source phantom in the given space.
    """
    if space.ndim == 2:
        return ellipsoid_phantom(space, _derenzo_sources_2d(), min_pt, max_pt)
    if space.ndim == 3:
        return ellipsoid_phantom(
            space, cylinders_from_ellipses(_derenzo_sources_2d()),
            min_pt, max_pt)
    else:
        raise ValueError('dimension not 2, no phantom available')


if __name__ == '__main__':
    # Show the phantoms
    import odl
    from odl.util.testutils import run_doctests

    n = 300

    # 2D
    discr = odl.uniform_discr([-1, -1], [1, 1], [n, n])
    derenzo_sources(discr).show('derenzo_sources 2d')

    # 3D
    discr = odl.uniform_discr([-1, -1, -1], [1, 1, 1], [300, 300, 300])
    derenzo_sources(discr).show('derenzo_sources 3d')

    # Run also the doctests
    run_doctests()
