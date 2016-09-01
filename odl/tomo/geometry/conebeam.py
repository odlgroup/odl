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

"""Cone beam geometries in 3 dimensions."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

from odl.tomo.geometry.detector import Flat2dDetector
from odl.tomo.geometry.geometry import (
    DivergentBeamGeometry, AxisOrientedGeometry)
from odl.tomo.util.utility import perpendicular_vector


__all__ = ('CircularConeFlatGeometry', 'HelicalConeFlatGeometry',)


class HelicalConeFlatGeometry(DivergentBeamGeometry, AxisOrientedGeometry):

    """Cone beam geometry with helical source curve and flat detector.

    The source moves along a spiral oriented along a fixed ``axis``, with
    radius ``src_radius`` in the azimuthal plane and a given ``pitch``.
    The detector reference point is opposite to the source, i.e. in
    the point at distance ``src_rad + det_rad`` on the line in the
    azimuthal plane through the source point and ``axis``.

    The motion parameter is the 1d rotation angle parameterizing source
    and detector positions simultaneously.

    In the standard configuration, the rotation axis is ``(0, 0, 1)``,
    the initial source-to-detector vector is ``(1, 0, 0)``, and the
    initial detector axes are ``[(0, 1, 0), (0, 0, 1)]``.

    See Also
    --------
    CircularConeFlatGeometry : Case with zero pitch
    """

    def __init__(self, apart, dpart, src_radius, det_radius, pitch,
                 axis=[0, 0, 1], **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        apart : 1-dim. `RectPartition`
            Partition of the angle interval
        dpart : 2-dim. `RectPartition`
            Partition of the detector parameter rectangle
        src_radius : nonnegative float
            Radius of the source circle
        det_radius : nonnegative float
            Radius of the detector circle
        pitch : float
            Constant vertical distance that a point on the helix
            traverses when increasing the angle parameter by ``2 * pi``
        axis : `array-like`, shape ``(3,)``, optional
            Fixed rotation axis, the symmetry axis of the helix

        Other Parameters
        ----------------
        src_to_det_init : `array-like`, shape ``(2,)``, optional
            Initial state of the vector pointing from source to detector
            reference point. The zero vector is not allowed.
            By default, a `perpendicular_vector` to ``axis`` is used.
        det_init_axes : 2-tuple of `array-like`'s (shape ``(2,)``), optional
            Initial axes defining the detector orientation.
            By default, the normalized cross product of ``axis`` and
            ``src_to_det_init`` is used as first axis and ``axis`` as
            second.
        pitch_offset : float, optional
            Offset along the ``axis`` at ``angle=0``
        """
        AxisOrientedGeometry.__init__(self, axis)

        src_to_det_init = kwargs.pop('src_to_det_init',
                                     perpendicular_vector(self.axis))

        if np.linalg.norm(src_to_det_init) <= 1e-10:
            raise ValueError('initial source to detector vector {} is too '
                             'close to zero'.format(src_to_det_init))
        self._src_to_det_init = (np.array(src_to_det_init) /
                                 np.linalg.norm(src_to_det_init))

        det_init_axes = kwargs.pop('det_init_axes', None)
        if det_init_axes is None:
            det_init_axis_0 = np.cross(self.axis, self._src_to_det_init)
            det_init_axes = (det_init_axis_0, axis)

        detector = Flat2dDetector(dpart, det_init_axes)

        super().__init__(ndim=3, motion_part=apart, detector=detector)

        self._pitch = float(pitch)
        self._pitch_offset = float(kwargs.pop('pitch_offset', 0))
        self._src_radius = float(src_radius)
        if self.src_radius < 0:
            raise ValueError('source circle radius {} is negative'
                             ''.format(src_radius))
        self._det_radius = float(det_radius)
        if self.det_radius < 0:
            raise ValueError('detector circle radius {} is negative'
                             ''.format(det_radius))

        if self.src_radius == 0 and self.det_radius == 0:
            raise ValueError('source and detector circle radii cannot both be '
                             '0')

    @property
    def src_radius(self):
        """Source circle radius of this geometry."""
        return self._src_radius

    @property
    def det_radius(self):
        """Detector circle radius of this geometry."""
        return self._det_radius

    @property
    def pitch(self):
        """Constant vertical distance traversed in a full rotation."""
        return self._pitch

    @property
    def src_to_det_init(self):
        """Initial state of the vector pointing from source to detector
        reference point."""
        return self._src_to_det_init

    @property
    def det_init_axes(self):
        """Initial axes defining the detector orientation."""
        return self.detector.axes

    @property
    def pitch_offset(self):
        """Vertical offset at ``angle=0``."""
        return self._pitch_offset

    @property
    def angles(self):
        """Discrete angles given in this geometry."""
        return self.motion_grid.coord_vectors[0]

    def det_refpoint(self, angle):
        """Return the detector reference point position at ``angle``.

        For an angle ``phi``, the detector position is given by::

            ref(phi) = det_rad * rot_matrix(phi) * src_to_det_init +
                       (pitch_offset + pitch * phi) * axis

        where ``src_to_det_init`` is the initial unit vector pointing
        from source to detector.

        Parameters
        ----------
        angle : float
            Rotation angle given in radians, must be contained in
            this geometry's `motion_params`

        Returns
        -------
        point : `numpy.ndarray`, shape (3,)
            Detector reference point corresponding to the given angle

        See Also
        --------
        rotation_matrix
        """
        angle = float(angle)
        if angle not in self.motion_params:
            raise ValueError('`angle` {} is not in the valid range {}'
                             ''.format(angle, self.motion_params))

        # Initial vector from 0 to the detector. It can be computed this way
        # since source and detector are at maximum distance, i.e. the
        # connecting line passes the origin.
        origin_to_det_init = self.det_radius * self._src_to_det_init
        circle_component = self.rotation_matrix(angle).dot(origin_to_det_init)

        # Increment along the rotation axis according to pitch and pitch_offset
        pitch_component = self.axis * (self.pitch_offset +
                                       self.pitch * angle / (2 * np.pi))

        return circle_component + pitch_component

    def src_position(self, angle):
        """Return the source position at ``angle``.

        For an angle ``phi``, the source position is given by::

            src(phi) = -src_rad * rot_matrix(phi) * src_to_det_init +
                       (pitch_offset + pitch * phi) * axis

        where ``src_to_det_init`` is the initial unit vector pointing
        from source to detector.

        Parameters
        ----------
        angle : float
            Rotation angle given in radians, must be contained in
            this geometry's `motion_params`

        Returns
        -------
        point : `numpy.ndarray`, shape (3,)
            Detector reference point corresponding to the given angle

        See Also
        --------
        rotation_matrix
        """
        angle = float(angle)
        if angle not in self.motion_params:
            raise ValueError('`angle` {} is not in the valid range {}'
                             ''.format(angle, self.motion_params))

        # Initial vector from 0 to the source. It can be computed this way
        # since source and detector are at maximum distance, i.e. the
        # connecting line passes the origin.
        origin_to_src_init = -self.src_radius * self._src_to_det_init
        circle_component = self.rotation_matrix(angle).dot(origin_to_src_init)

        # Increment by pitch
        pitch_component = self.axis * (self.pitch_offset +
                                       self.pitch * angle / (np.pi * 2))

        return circle_component + pitch_component

    def __repr__(self):
        """Return ``repr(self)``."""

        arg_fstr = '\n    {!r},\n    {!r},\n    src_radius={}, det_radius={}'
        if self.pitch != 0:
            arg_fstr += ',\n    pitch={pitch}'
        if self.pitch_offset != 0:
            arg_fstr += ',\n    pitch_offset={pitch_offset}'
        if not np.allclose(self.axis, [0, 0, 1]):
            arg_fstr += ',\n    axis={axis}'
        default_src_to_det = perpendicular_vector(self.axis)
        if not np.allclose(self._src_to_det_init, default_src_to_det):
            arg_fstr += ',\n    src_to_det_init={src_to_det_init}'

        default_axes = [np.cross(self.axis, self._src_to_det_init), self.axis]
        if not np.allclose(self.detector.axes, default_axes):
            arg_fstr += ',\n    det_init_axes={det_init_axes!r}'

        arg_str = arg_fstr.format(
            self.motion_partition, self.det_partition,
            self.src_radius, self.det_radius,
            pitch=self.pitch,
            pitch_offset=self.pitch_offset,
            axis=list(self.axis),
            src_to_det_init=list(self.src_to_det_init),
            det_init_axes=[list(a) for a in self.det_init_axes])
        return '{}({})'.format(self.__class__.__name__, arg_str)

    # Fix for bug in ABC thinking this is abstract
    rotation_matrix = AxisOrientedGeometry.rotation_matrix


class CircularConeFlatGeometry(HelicalConeFlatGeometry):

    """Cone beam geometry with circular source curve and flat detector.

    The source moves along a circle with radius ``src_radius`` in the
    plane perpendicular to a fixed ``axis``. The detector reference
    point is opposite to the source, i.e. in the same plane on a circle
    with radius ``det_rad`` at maximum distance to the source. This
    implies that it lies on the line through the source point and
    the intersection of the ``axis`` with the azimuthal plane.

    The motion parameter is the 1d rotation angle parameterizing source
    and detector positions simultaneously.

    In the standard configuration, the rotation axis is ``(0, 0, 1)``,
    the initial source-to-detector vector is ``(1, 0, 0)``, and the
    initial detector axes are ``[(0, 1, 0), (0, 0, 1)]``.

    See Also
    --------
    HelicalConeFlatGeometry : General case with motion in z direction
    """

    def __init__(self, apart, dpart, src_radius, det_radius, axis=[0, 0, 1],
                 **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        apart : 1-dim. `RectPartition`
            Partition of the angle interval
        dpart : 2-dim. `RectPartition`
            Partition of the detector parameter rectangle
        src_radius : nonnegative float
            Radius of the source circle
        det_radius : nonnegative float
            Radius of the detector circle
        axis : array-like, shape ``(3,)``, optional
            Fixed rotation axis, the symmetry axis of the helix
        src_to_det_init : array-like, shape ``(2,)``, optional
            Initial state of the vector pointing from source to detector
            reference point. The zero vector is not allowed.
            By default, a `perpendicular_vector` to ``axis`` is used.
        det_init_axes : 2-tuple of `array-like`'s (shape ``(2,)``), optional
            Initial axes defining the detector orientation.
            By default, the normalized cross product of ``axis`` and
            ``src_to_det_init`` is used as first axis and ``axis`` as
            second.
        """
        kwargs.pop('pitch_offset', None)
        super().__init__(apart, dpart, src_radius, det_radius, pitch=0,
                         axis=axis, **kwargs)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
