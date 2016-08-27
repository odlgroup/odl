# Copyright 2014 - 2016 The ODL development group
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

    def __init__(self, apart, dpart, det_rad, axis=[0, 0, 1],
                 **kwargs):
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
            Fixed rotation axis.
        orig_to_det_init : `array-like`, shape ``(3,)``, optional
            Vector pointing towards the detector reference point in
            the initial position.
            Default: a `perpendicular_vector` to ``axis``.
        det_init_axes : 2-tuple of `array-like`'s (shape ``(3,)``), optional
            Initial axes defining the detector orientation.
            Default: the normalized cross product of ``axis`` and
            ``orig_to_det_init`` is used as first axis and ``axis`` as second.
        """
        self._det_radius = float(det_rad)
        if self.det_radius <= 0:
            raise ValueError('expected a positive radius, got {}'
                             ''.format(det_rad))

        orig_to_det_init = kwargs.pop('orig_to_det_init',
                                      perpendicular_vector(axis))

        init_pos_norm = np.linalg.norm(orig_to_det_init)
        if init_pos_norm > 1e-10:
            orig_to_det_init *= self.det_radius / init_pos_norm
        else:
            raise ValueError('`orig_to_det_init` {} is too close to zero'
                             ''.format(orig_to_det_init))
        kwargs['det_init_pos'] = orig_to_det_init
        super().__init__(apart, dpart, axis, **kwargs)

    @property
    def det_radius(self):
        """Radius of the detector orbit."""
        return self._det_radius

    # Fix for bug in ABC thinking this is abstract
    rotation_matrix = AxisOrientedGeometry.rotation_matrix
