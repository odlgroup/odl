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

"""Cone beam geometries in 3 dimensions."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

# External
import numpy as np

# Internal

from odl.tomo.geometry.detector import Flat2dDetector
from odl.tomo.geometry.geometry import (DivergentBeamGeometry,
                                        AxisOrientedGeometry)
from odl.tomo.util.trafos import perpendicular_vector


__all__ = ('CircularConeFlatGeometry',
           'HelicalConeFlatGeometry',)


class HelicalConeFlatGeometry(DivergentBeamGeometry, AxisOrientedGeometry):
    """Cone beam geometry with helical acquisition and flat detector.

    The source moves along a spiral with radius ``r`` in the azimuthal plane
    and a pitch``P``. The detector reference point is opposite to
    the source and moves on a spiral with radius ``R`` in the azimuthal
    plane and pitch ``P``.

    The motion parameter is the (1d) rotation angle parametrizing source and
    detector positions.
    """

    def __init__(self, angle_intvl, dparams, src_radius, det_radius, pitch,
                 agrid=None, dgrid=None, axis=[0, 0, 1], src_to_det=None,
                 detector_axes=None):
        """Initialize a new instance.

        Parameters
        ----------
        angle_intvl : `Interval` or 1-dim. `IntervalProd`
            The motion parameters given in radian
        dparams : `Rectangle` or 2-dim. `IntervalProd`
            The detector parameters
        src_radius : `float`
            Radius of the source circle, must be positive
        det_radius : `float`
            Radius of the detector circle, must be positive
        pitch : positive `float`
            Constant vertical distance between two source positions, one at
            angle ``phi``, the other at angle ``phi + 2 * pi``
        agrid : 1-dim. `TensorGrid`, optional
            A sampling grid for `angle_intvl`
        dgrid : 2-dim. `TensorGrid`, optional
            A sampling grid for `dparams`
        axis : 3-element array, optional
            Fixed rotation axis defined by a 3-element vector
        src_to_det : 3-element array, optional
            The direction from the source to the point (0, 0) of the detector
            angle=0. Default: Vector in x, y plane orthogonal to axis.
        detector_axes : sequence of two 3-element arrays, optional
            Unit directions along each detector parameter of the detector.
            Default: (normalized) [np.cross(axis, source_to_detector), axis]
        """

        AxisOrientedGeometry.__init__(self, axis)

        if src_to_det is None:
            src_to_det = perpendicular_vector(axis)

        self._src_to_det = (np.array(src_to_det) /
                            np.linalg.norm(src_to_det))

        if detector_axes is None:
            detector_axes = [np.cross(self.axis, self._src_to_det),
                             self.axis]

        detector = Flat2dDetector(dparams, detector_axes, dgrid)

        DivergentBeamGeometry.__init__(self, 3, angle_intvl, detector, agrid)

        self._pitch = pitch
        self._src_radius = float(src_radius)
        if self.src_radius <= 0:
            raise ValueError('source circle radius {} is not positive.'
                             ''.format(src_radius))
        self._det_radius = float(det_radius)
        if self.det_radius <= 0:
            raise ValueError('detector circle radius {} is not positive.'
                             ''.format(det_radius))

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
        """Constant vertical distance between a full rotation.

        Returns
        -------
        pitch : positive `float`
        """
        return self._pitch

    def det_refpoint(self, angle):
        """The detector reference point function.

        Parameters
        ----------
        angle : `float`
            The motion parameter given in radian. It must be
            contained in this geometry's motion parameter set

        Returns
        -------
        point : `numpy.ndarray`, shape (3,)
            The reference point on a circle in the azimuthal plane with
            radius ``R`` and at a longitudinal position ``z`` at a given
            rotation angle ``phi`` defined as ``(-R * sin(phi), R * cos(
            phi), z)`` where ``z`` is given by the pitch ``P``.
        """
        angle = float(angle)
        if angle not in self.motion_params:
            raise ValueError('angle {} is not in the valid range {}.'
                             ''.format(angle, self.motion_params))

        # Distance from 0 to detector
        origin_to_det = self.det_radius * self._src_to_det
        circle_component = self.rotation_matrix(angle).dot(origin_to_det)

        # Increment by pitch
        pitch_component = self.axis * self.pitch * angle / (np.pi * 2)

        return circle_component + pitch_component

    def src_position(self, angle):
        """The source position function.

        Parameters
        ----------
        angle : `float`
            The motion parameter given in radian. It must be contained
            in this geometry's motion parameter set

        Returns
        -------
        point : `numpy.ndarray`, shape (3,)
            The source position on a spiral with radius ``r`` and pitch
            ``P`` at a given rotation angle ``phi`` defined as
            ``(r * sin(phi), -r * cos(phi), P * phi / (2 * pi))``
        """
        angle = float(angle)
        if angle not in self.motion_params:
            raise ValueError('angle {} is not in the valid range {}.'
                             ''.format(angle, self.motion_params))

        # Distance from 0 to detector
        origin_to_src = -self.src_radius * self._src_to_det
        circle_component = self.rotation_matrix(angle).dot(origin_to_src)

        # Increment by pitch
        pitch_component = self.axis * self.pitch * angle / (np.pi * 2)

        return circle_component + pitch_component

    def __repr__(self):
        """Return ``repr(self)``"""

        arg_fstr = '{!r}, {!r},\n    src_radius={}, det_radius={}'
        if self.pitch != 0:
            arg_fstr += ',\n    pitch={pitch!r}'
        if self.has_motion_sampling:
            arg_fstr += ',\n    agrid={agrid!r}'
        if self.has_det_sampling:
            arg_fstr += ',\n    dgrid={dgrid!r}'
        if not np.allclose(self.axis, [0, 0, 1]):
            arg_fstr += ',\n    axis={axis!r}'
        if not np.allclose(self._src_to_det, [1, 0, 0]):
            arg_fstr += ',\n    src_to_det={src_to_det!r}'

        default_axes = [np.cross(self.axis, self._src_to_det), self.axis]
        if not np.allclose(self.detector.detector_axes, default_axes):
            arg_fstr += ',\n    detector_axes={detector_axes!r}'

        arg_str = arg_fstr.format(self.motion_params, self.det_params,
                                  self.src_radius, self.det_radius,
                                  pitch=self.pitch,
                                  agrid=self.motion_grid,
                                  dgrid=self.det_grid,
                                  axis=self.axis,
                                  src_to_det=self._src_to_det,
                                  detector_axes=self.detector.detector_axes)
        return '{}({})'.format(self.__class__.__name__, arg_str)

    # Fix for bug in ABC thinking this is abstract
    rotation_matrix = AxisOrientedGeometry.rotation_matrix


class CircularConeFlatGeometry(HelicalConeFlatGeometry):
    """Cone beam geometry with circular acquisition and flat detector.

    The source moves on a circle with radius ``r``, and the detector
    reference point is opposite to the source on a circle with radius ``R``
    and aligned tangential to the circle.

    The motion parameter is the (1d) rotation angle parametrizing source and
    detector positions.
    """

    def __init__(self, angle_intvl, dparams, src_radius, det_radius,
                 agrid=None, dgrid=None, axis=[0, 0, 1], src_to_det=None,
                 detector_axes=None):
        """Initialize a new instance.

        Parameters
        ----------
        angle_intvl : `Interval` or 1-dim. `IntervalProd`
            The motion parameters given in radian
        dparams : `Rectangle` or 2-dim. `IntervalProd`
            The detector parameters
        src_radius : `float`
            Radius of the source circle, must be positive
        det_radius : `float`
            Radius of the detector circle, must be positive
        agrid : 1-dim. `TensorGrid`, optional
            A sampling grid for ``angle_intvl``
        dgrid : 2-dim. `TensorGrid`, optional
            A sampling grid for ``dparams``
        axis : 3-element array, optional
            Fixed rotation axis defined by a 3-element vector
        src_to_det : 3-element array, optional
            The direction from the source to the point (0, 0) of the detector
            angle=0. Default: Vector in x, y plane orthogonal to axis.
        detector_axes : sequence of two 3-element arrays, optional
            Defines the unit directions along each detector parameter of the
            detector.
            Default: (normalized) [np.cross(axis, source_to_detector), axis]
        """
        pitch = 0
        super().__init__(angle_intvl, dparams, src_radius, det_radius,
                         pitch, agrid, dgrid, axis, src_to_det, detector_axes)
