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

"""Positron emission tomography (PET) geometry."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import numpy as np

# Internal
import odl
from odl.tomo.geometry.detector import Flat2dDetector
from odl.tomo.geometry.geometry import (Geometry,
                                        AxisOrientedGeometry)
from odl.tomo.util.utility import perpendicular_vector

__all__ = ('CylindricalPetGeom')

class CylindricalPetGeom(AxisOrientedGeometry, Geometry):
    """Cylindrical PET scanner geometry."""

    def __init__(self, det_radius,
                 ring_center_to_det,
                 apart,
                 detector,
                 axialpart,
                 axis=[0, 0, 1]):

        AxisOrientedGeometry.__init__(self, axis)

        self._det_radius = float(det_radius)

        if self.det_radius <= 0:
            raise ValueError('ring circle radius {} is not positive.'
                             ''.format(det_radius))

        self._apart = apart
        self._ring_center_to_det = ring_center_to_det
        self._detector = detector
        self._axialpart = axialpart

        Geometry.__init__(self, 3, apart, detector)


    @property
    def ax_motion_partition(self):
        """Partition of the axial motion parameter set into subsets."""
        return self._ax_motion_part

    @property
    def ax_motion_params(self):
        """Continuous transaxial motion parameter range, an `IntervalProd`."""
        return self.ax_motion_partition.set

    @property
    def ax_motion_grid(self):
        """Sampling grid of `transax_motion_params."""
        return self.ax_motion_partition.grid

    @property
    def det_radius(self):
        """Gantry crystal radius."""
        return self._det_radius

    @property
    def ring_number(self):
        """Number of detector rings."""
        return self._num_rings

    @property
    def block_number(self):
        """Number of detector blocks in a ring."""
        return self._num_blocks

    def det_point_position(self, mpar, dpar):
        """The detector point position function.
        Parameters
        ----------
        mpar : element of motion parameters `motion_params`
            Motion parameter at which to evaluate
        dpar : element of detector parameters `det_params`
            Detector parameter at which to evaluate
        Returns
        -------
        pos : `numpy.ndarray`, shape (`ndim`,)
            The source position, a `ndim`-dimensional vector
        """
        angle = float(mpar[0])
        return np.asarray(
            (self.det_refpoint(mpar) +
             self.rotation_matrix(angle).dot(self.detector.surface(dpar))))

    def det_refpoint(self, mparams):
        """The detector reference point function.
        Parameters
        ----------
        mparams : `tuple`
            mparams = (angle, z_shift)
            The parameter angle `float` given in radian. It must be
            contained in this geometry's motion parameter set.
            z_shift `float` parameter related to the axial motion,
            i.e. movement to the isocenter of a ring
        Returns
        -------
        point : `numpy.ndarray`, shape (3,)
            The reference point on a circle in the transaxial plane with
            radius ``R`` and on at a axial position ``z``.
            Transaxial point given by the rotation angle ``phi``.
            Axial position given by z_shift.
        """
        angle = float(mparams[0])

        # Distance from 0 to detector
        origin_to_det = self.det_radius * self._ring_center_to_det
        circle_component = self.rotation_matrix(angle).dot(origin_to_det)

        # Longitudinal increment
        z_shift = int(mparams[1])
        z_axis_shift = self.axis * z_shift

        return circle_component + z_axis_shift


