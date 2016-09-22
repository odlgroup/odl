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

"""Fan beam geometries in 2 dimensions."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

from odl.tomo.geometry.detector import Flat1dDetector
from odl.tomo.geometry.geometry import DivergentBeamGeometry
from odl.tomo.util.utility import euler_matrix, perpendicular_vector


__all__ = ('FanFlatGeometry',)


class FanFlatGeometry(DivergentBeamGeometry):

    """Abstract 2d fan beam geometry with flat 1d detector.

    The source moves on a circle with radius ``src_radius``, and the
    detector reference point is opposite to the source, i.e. at maximum
    distance, on a circle with radius ``det_radius``. One of the two
    radii can be chosen as 0, which corresponds to a stationary source
    or detector, respectively.

    The motion parameter is the 1d rotation angle parameterizing source
    and detector positions simultaneously.

    In the standard configuration, the source and detector start on the
    first coodinate axis with vector ``(1, 0)`` from source to detector,
    and the initial detector axis is ``(0, 1)``.
    """

    def __init__(self, apart, dpart, src_radius, det_radius, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        apart : 1-dim. `RectPartition`
            Partition of the angle interval
        dpart : 1-dim. `RectPartition`
            Partition of the detector parameter interval
        src_radius : nonnegative float
            Radius of the source circle
        det_radius : nonnegative float
            Radius of the detector circle
        src_to_det_init : `array-like` (shape ``(2,)``), optional
            Initial state of the vector pointing from source to detector
            reference point. The zero vector is not allowed.
            Default: ``(1, 0)``.
        det_init_axis : `array-like` (shape ``(2,)``), optional
            Initial axis defining the detector orientation.
            By default, a normalized `perpendicular_vector` to
            ``src_to_det_init`` is used.
        """
        src_to_det_init = kwargs.pop('src_to_det_init', (1.0, 0.0))
        det_init_axis = kwargs.pop('det_init_axis', None)

        if np.shape(src_to_det_init) != (2,):
            raise ValueError('`src_to_det_init` has shape {}, '
                             'expected (2,)'
                             ''.format(np.shape(src_to_det_init)))
        if np.linalg.norm(src_to_det_init) <= 1e-10:
            raise ValueError('`src_to_det_init` {} too close '
                             'to zero'.format(src_to_det_init))
        self._src_to_det_init = (np.asarray(src_to_det_init, dtype='float64') /
                                 np.linalg.norm(src_to_det_init))

        if det_init_axis is None:
            det_init_axis = perpendicular_vector(self._src_to_det_init)

        self._src_radius, src_radius_in = float(src_radius), src_radius
        if self.src_radius < 0:
            raise ValueError('`src_radius` {} is negative'
                             ''.format(src_radius_in))
        self._det_radius, det_radius_in = float(det_radius), det_radius
        if det_radius < 0:
            raise ValueError('`det_radius` {} is negative'
                             ''.format(det_radius_in))
        if self.src_radius == 0 and self.det_radius == 0:
            raise ValueError('source and detector radii cannot both be 0')

        detector = Flat1dDetector(dpart, det_init_axis)
        super().__init__(ndim=2, motion_part=apart, detector=detector)

    @property
    def src_radius(self):
        """Source circle radius of this geometry."""
        return self._src_radius

    @property
    def det_radius(self):
        """Detector circle radius of this geometry."""
        return self._det_radius

    @property
    def angles(self):
        """Discrete angles given in this geometry."""
        return self.motion_grid.coord_vectors[0]

    def src_position(self, angle):
        """Return the source position at ``angle``.

        For an angle ``phi``, the source position is given by::

            src(phi) = -src_rad * rot_matrix(phi) * src_to_det_init

        where ``src_to_det_init`` is the initial unit vector pointing
        from source to detector.

        Parameters
        ----------
        angle : float
            Rotation angle given in radians, must be contained in
            this geometry's `motion_params`

        Returns
        -------
        point : `numpy.ndarray`, shape ``(2,)``
            Source position corresponding to the given angle
        """
        if angle not in self.motion_params:
            raise ValueError('`angle` {} is not in the valid range {}'
                             ''.format(angle, self.motion_params))

        # Initial vector from 0 to the source. It can be computed this way
        # since source and detector are at maximum distance, i.e. the
        # connecting line passes the origin.
        origin_to_src_init = -self.src_radius * self._src_to_det_init
        return self.rotation_matrix(angle).dot(origin_to_src_init)

    def det_refpoint(self, angle):
        """Return the detector reference point position at ``angle``.

        For an angle ``phi``, the detector position is given by::

            ref(phi) = det_rad * rot_matrix(phi) * src_to_det_init

        where ``src_to_det_init`` is the initial unit vector pointing
        from source to detector.

        Parameters
        ----------
        angle : float
            Rotation angle given in radians, must be contained in
            this geometry's `motion_params`

        Returns
        -------
        point : `numpy.ndarray`, shape (2,)
            Detector reference point corresponding to the given angle

        See Also
        --------
        rotation_matrix
        """
        if angle not in self.motion_params:
            raise ValueError('`angle` {} is not in the valid range {}'
                             ''.format(angle, self.motion_params))

        # Initial vector from 0 to the detector. It can be computed this way
        # since source and detector are at maximum distance, i.e. the
        # connecting line passes the origin.
        origin_to_det_init = self.det_radius * self._src_to_det_init
        return self.rotation_matrix(angle).dot(origin_to_det_init)

    def rotation_matrix(self, angle):
        """Return the rotation matrix for ``angle``.

        For an angle ``phi``, the matrix is given by::

            rot(phi) = [[cos(phi), -sin(phi)],
                        [sin(phi), cos(phi)]]

        Parameters
        ----------
        angle : float
            Rotation angle given in radians, must be contained in
            this geometry's `motion_params`

        Returns
        -------
        rot : `numpy.ndarray`, shape (2, 2)
            The rotation matrix mapping the standard basis vectors in
            the fixed ("lab") coordinate system to the basis vectors of
            the local coordinate system of the detector reference point,
            expressed in the fixed system
        """
        angle = float(angle)
        if angle not in self.motion_params:
            raise ValueError('`angle` {} not in the valid range {}'
                             ''.format(angle, self.motion_params))
        return euler_matrix(angle)

    # TODO: back projection weighting function?

    def __repr__(self):
        """Return ``repr(self)``."""
        arg_fstr = '{!r}, {!r}, src_radius={}, det_radius={}'

        if not np.allclose(self._src_to_det_init, [1, 0]):
            arg_fstr += ',\n    src_to_det_init={src_to_det_init!r}'

        default_axis = perpendicular_vector(self._src_to_det_init)
        if not np.allclose(self.detector.axis, default_axis):
            arg_fstr += ',\n    det_init_axis={det_init_axis!r}'

        arg_str = arg_fstr.format(self.motion_partition, self.det_partition,
                                  self.src_radius, self.det_radius,
                                  src_to_det_init=self._src_to_det_init,
                                  det_init_axis=self.detector.axis)

        return '{}({})'.format(self.__class__.__name__, arg_str)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
