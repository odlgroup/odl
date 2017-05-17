# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Single-photon emission computed tomography (SPECT) geometry."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

from odl.tomo.geometry.geometry import AxisOrientedGeometry
from odl.tomo.geometry.parallel import Parallel3dAxisGeometry
from odl.tomo.util.utility import perpendicular_vector

__all__ = ('ParallelHoleCollimatorGeometry', )


class ParallelHoleCollimatorGeometry(Parallel3dAxisGeometry):

    """Geometry for SPECT Parallel hole collimator."""

    def __init__(self, apart, dpart, det_rad, axis=(0, 0, 1), **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        apart : 1-dim. `RectPartition`
            Partition of the angle interval
        dpart : 2-dim. `RectPartition`
            Partition of the detector parameter rectangle
        det_rad : positive float
            Radius of the circular detector orbit.
        axis : `array-like`, shape ``(3,)``, optional
            Vector defining the fixed rotation axis of this geometry.
        orig_to_det_init : `array-like`, shape ``(3,)``, optional
            Vector pointing towards the initial position of the detector
            reference point. The default depends on ``axis``, see Notes.
            The zero vector is not allowed.
        det_axes_init : 2-tuple of `array-like`'s (shape ``(3,)``), optional
            Initial axes defining the detector orientation. The default
            depends on ``axis``, see Notes.
        extra_rot : `array_like`, shape ``(3, 3)``, optional
            Rotation matrix that should be applied at the end to the
            configuration of ``orig_to_det_init`` and ``det_axes_init``.
            The rotation is extrinsic, i.e., defined in the "world"
            coordinate system.

        Notes
        -----
        In the default configuration, the rotation axis is ``(0, 0, 1)``,
        the vector towards the initial detector reference point is
        ``(0, 1, 0)``, and the default detector axes are
        ``[(1, 0, 0), (0, 0, 1)]``.
        If a different ``axis`` is provided, the new default initial
        position and the new default axes are the computed by rotating
        the original ones by a matrix that transforms ``(0, 0, 1)`` to the
        new (normalized) ``axis``. This matrix is calculated with the
        `rotation_matrix_from_to` function. Expressed in code, we have ::

            init_rot = rotation_matrix_from_to((0, 0, 1), axis)
            orig_to_det_init = init_rot.dot((0, 1, 0))
            det_axes_init[0] = init_rot.dot((1, 0, 0))
            det_axes_init[1] = init_rot.dot((0, 0, 1))
        """
        self.__det_radius = float(det_rad)
        if self.det_radius <= 0:
            raise ValueError('expected a positive radius, got {}'
                             ''.format(det_rad))

        orig_to_det_init = kwargs.pop('orig_to_det_init', None)

        if orig_to_det_init is not None:
            init_pos_norm = np.linalg.norm(orig_to_det_init)
            if init_pos_norm == 0:
                raise ValueError('`orig_to_det_init` cannot be zero')
            else:
                orig_to_det_init *= self.det_radius / init_pos_norm
            kwargs['det_pos_init'] = orig_to_det_init
        super().__init__(apart, dpart, axis, **kwargs)

    @property
    def det_radius(self):
        """Radius of the detector orbit."""
        return self.__det_radius

    @property
    def orig_to_det_init(self):
        """Unit vector from origin towards initial detector reference point."""
        return self.det_pos_init / np.linalg.norm(self.det_pos_init)
