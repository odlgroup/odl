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

"""Fanbeam geometries."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

# External
import numpy as np

# Internal
from odl.tomo.geometry.detector import Flat1dDetector
from odl.tomo.geometry.geometry import DivergentBeamGeometry
from odl.tomo.util.utility import euler_matrix


__all__ = ('FanFlatGeometry',)


class FanFlatGeometry(DivergentBeamGeometry):

    """Abstract 2d fan beam geometry.

    The source moves on a circle with radius ``r``, and the detector
    reference point is opposite to the source on a circle with radius
    ``R``.

    The motion parameter is the (1d) rotation angle parameterizing source and
    detector positions.
    """

    def __init__(self, angle_intvl, dparams, src_radius, det_radius,
                 agrid=None, dgrid=None, src_to_det=[1, 0],
                 detector_axis=None):
        """Initialize a new instance.

        Parameters
        ----------
        angle_intvl : `Interval` or 1-dim. `IntervalProd`
            The motion parameters given in radian
        src_radius : positive `float`
            Radius of the source circle, must be positive
        det_radius : positive `float`
            Radius of the detector circle, must be positive
        agrid : 1-dim. `TensorGrid`, optional
            A sampling grid for `angle_intvl`
        src_to_det : 2-element array, optional
            The direction from the source to the point (0) of the detector
            angle=0
        detector_axis : 2-element array, optional
            Unit direction along the detector parameter of the detector.
            Default: (normalized) [-self.src_to_det[1], self.src_to_det[0]]
        """

        self._src_to_det = (np.array(src_to_det) /
                            np.linalg.norm(src_to_det))

        if detector_axis is None:
            # Rotated by 90 degrees according to right hand rule
            detector_axis = np.array([-self._src_to_det[1],
                                      self._src_to_det[0]])

        self._src_radius = float(src_radius)
        if src_radius <= 0:
            raise ValueError('source circle radius {} is not positive.'
                             ''.format(src_radius))
        self._det_radius = float(det_radius)
        if det_radius <= 0:
            raise ValueError('detector circle radius {} is not positive.'
                             ''.format(det_radius))

        detector = Flat1dDetector(dparams, detector_axis, dgrid)
        super().__init__(2, angle_intvl, detector, agrid)

    @property
    def src_radius(self):
        """Source circle radius of this geometry."""
        return self._src_radius

    @property
    def det_radius(self):
        """Detector circle radius of this geometry."""
        return self._det_radius

    def src_position(self, angle):
        """The source position function.

        Parameters
        ----------
        angle : `float`
            The motion parameters given in radian. It must be contained in
            this geometry's motion parameter set

        Returns
        -------
        point : `numpy.ndarray`, shape (2,)
            The source position on the circle with radius `r` at the given
            rotation angle ``phi``, defined as ``-r * (cos(phi), sin(phi))``
        """
        angle = float(angle)
        if angle not in self.motion_params:
            raise ValueError('angle {} is not in the valid range {}.'
                             ''.format(angle, self.motion_params))

        # Distance from 0 to source
        origin_to_det = -self.src_radius * self._src_to_det
        return self.rotation_matrix(angle).dot(origin_to_det)

    def det_refpoint(self, angle):
        """The detector reference point function.

        Parameters
        ----------
        angle : `float`
            The motion parameter given in radian. It must be
            contained in this geometry's motion parameter set

        Returns
        -------
        point : `numpy.ndarray`, shape (2,)
            The reference point on the circle with radius ``R`` at a given
            rotation angle ``phi`` defined as ``R * (cos(phi), sin(phi))``
        """
        angle = float(angle)
        if angle not in self.motion_params:
            raise ValueError('angle {} is not in the valid range {}.'
                             ''.format(angle, self.motion_params))

        # Distance from 0 to detector
        origin_to_det = self.det_radius * self._src_to_det
        return self.rotation_matrix(angle).dot(origin_to_det)

    def rotation_matrix(self, angle):
        """The detector rotation function.

        Parameters
        ----------
        angle : `float`
            The motion parameter given in radian. It must be
            contained in this geometry's `motion_params`

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
            raise ValueError('angle {} not in the valid range {}.'
                             ''.format(angle, self.motion_params))
        return euler_matrix(angle)

    # TODO: back projection weighting function?

    def __repr__(self):
        """Returns ``repr(self)``."""
        arg_fstr = '{!r}, {!r}, src_radius={}, det_radius={}'
        if self.has_motion_sampling:
            arg_fstr += ',\n agrid={agrid!r}'
        if self.has_det_sampling:
            arg_fstr += ',\n dgrid={dgrid!r}'

        if not np.allclose(self._src_to_det, [1, 0]):
            arg_fstr += ',\n    src_to_det={src_to_det!r}'

        default_axis = [-self._src_to_det[1], self._src_to_det[0]]
        if not np.allclose(self.detector.detector_axis, default_axis):
            arg_fstr += ',\n    detector_axes={detector_axes!r}'

        arg_str = arg_fstr.format(self.motion_params, self.det_params,
                                  self.src_radius, self.det_radius,
                                  agrid=self.motion_grid,
                                  dgrid=self.det_grid,
                                  src_to_det=self._src_to_det,
                                  detector_axis=self.detector.detector_axis)

        return '{}({})'.format(self.__class__.__name__, arg_str)
