# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Source and detector shifts for divergent beam geometries."""

from __future__ import print_function, division, absolute_import
import numpy as np
from odl.discr.discr_utils import nearest_interpolator

__all__ = ('flying_focal_spot',)


def flying_focal_spot(angle, apart, shifts):
    """Flying focal spot shifts for divergent beam geometries.
    Shifts are defined only for grid points of angular partition.
    For all other angles nearest neighbor interpolation is used.

    Parameters
    ----------
    angle : float or `array-like`
        Angle(s) in radians describing the counter-clockwise
        rotation of source and detector.
    apart : 1-dim. `RectPartition`
        Partition of the angle interval.
    shifts : sequence of `array-like`
        Each vectors in a sequence represent a subsequent shift
        relative to the default source position. Vector elements
        represent shifts along the following directions:
        det_to_src, tangent to the rotation
        (projected on a plane perpendicular to rotation axis), rotation axis.
    """
    assert apart.ndim == 1

    angle = np.array(angle, dtype=float, copy=False, ndmin=1)
    assert angle.ndim == 1

    shifts = np.array(shifts, dtype=float, ndmin=2)
    if shifts.shape[1] not in [2, 3]:
        raise ValueError('Flying focal spot shifts must have '
                         'shape (2,) or (3,), got {}'.format(shifts))

    interpolator = nearest_interpolator(np.arange(apart.size),
                                        apart.coord_vectors)
    ind = interpolator(angle)

    k = len(shifts)
    result = [shifts[int(i) % k] for i in ind]
    return np.array(result, dtype=float, ndmin=2)
