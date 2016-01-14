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

"""Parallel beam geometries."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super


# External
import numpy as np

# Internal
from odl.set.domain import IntervalProd
from odl.discr.grid import TensorGrid
from odl.tomo.geometry.detector import Flat1dDetector, Flat2dDetector
from odl.tomo.geometry.geometry import Geometry, AxisOrientedGeometry
from odl.tomo.util.trafos import euler_matrix

__all__ = ('ParallelGeometry', 'Parallel2dGeometry', 'Parallel3dGeometry')


class ParallelGeometry(Geometry):

    """Abstract parallel beam geometry, arbitrary dimension.

    The motion parameter is a (1d) rotation angle.
    """

    def __init__(self, ndim, angle_intvl, detector, origin_to_det, agrid=None):
        """Initialize a new instance.

        Parameters
        ----------
        ndim : positive int
            The dimensionality of the problem
        angle_intvl : `Interval` or 1-dim. `IntervalProd`
            The motion parameters given in radians
        detector : `Detector`
            The detector to use
        origin_to_det : `ndim`-element array, optional
            The direction from the origin to the point (0) of the detector
            when angle=0
        agrid : 1-dim. `TensorGrid`, optional
            A sampling grid for the ``angle_intvl``. Default: `None`
        """
        if not isinstance(angle_intvl, IntervalProd) or angle_intvl.ndim != 1:
            raise TypeError('angle parameters {!r} are not an interval.'
                            ''.format(angle_intvl))

        if agrid is not None:
            if not isinstance(agrid, TensorGrid):
                raise TypeError('angle grid {!r} is not a `TensorGrid` '
                                'instance.'.format(agrid))
            if not angle_intvl.contains_set(agrid):
                raise ValueError('angular grid {} not contained in angle '
                                 'interval {}.'.format(agrid, angle_intvl))

        super().__init__(ndim)
        self._motion_params = angle_intvl
        self._motion_grid = agrid
        self._detector = detector
        self._origin_to_det = (np.array(origin_to_det) /
                               np.linalg.norm(origin_to_det))

    @property
    def motion_params(self):
        """Motion parameters of this geometry."""
        return self._motion_params

    @property
    def motion_grid(self):
        """Sampling grid for this geometry's motion parameters."""
        return self._motion_grid

    @property
    def detector(self):
        """Detector of this geometry."""
        return self._detector

    @property
    def origin_to_det(self):
        """The direction from the origin to the point (0) of the detector
        when angle=0
        """
        return self._origin_to_det

    def det_refpoint(self, angle):
        """The detector reference point function.

        This is always given by the origin.

        Parameters
        ----------
        angle : `float`
            The motion parameters given in radians. It must be
            contained in this geometry's motion parameter set

        Returns
        -------
        point : `numpy.ndarray`, shape (`ndim`,)
            The reference point, equal to the origin
        """
        angle = float(angle)
        if angle not in self.motion_params:
            raise ValueError('angle {} not in the valid range {}.'
                             ''.format(angle, self.motion_params))
        return np.zeros(self.ndim)

    def det_to_src(self, angle, dpar, normalized=True):
        """Direction from a detector location to the source.

        In parallel geometry, this function is independent of the
        detector parameter.

        Since the (virtual) source is infinitely far away, the
        non-normalized version will return a vector with signed
        ``inf`` according to the quadrant.

        Parameters
        ----------
        angle : `float`
            The motion parameters given in radians. Must be contained
            in this geometry's `motion_params`
        dpar : `float`
            The detector parameter. Must be contained in this
            geometry's `det_params`.
        normalized : `bool`
            If `True`, return the normalized version of that vector.
            False raises NotImplementedError in this case.

        Returns
        -------
        vec : `numpy.ndarray`, shape (`ndim`,)
            (Unit) vector pointing from the detector to the source
        """
        angle = float(angle)
        if angle not in self.motion_params:
            raise ValueError('angle {} not in the valid range {}.'
                             ''.format(angle, self.motion_params))

        if dpar not in self.det_params:
            raise ValueError('detector parameter {} not in the valid range {}.'
                             ''.format(dpar, self.det_params))

        if not normalized:
            raise NotImplementedError('non-normalized detector to source is '
                                      'not available in parallel case')

        return self.rotation_matrix(angle).dot(self.origin_to_det)


class Parallel2dGeometry(ParallelGeometry):

    """Parallel beam geometry in 2d.

    The motion parameter is the counter-clockwise rotation angle around the
    origin, and the detector is a line detector perpendicular to the ray
    direction.

    """

    def __init__(self, angle_intvl, dparams, agrid=None, dgrid=None,
                 origin_to_det=[1, 0]):
        """Initialize a new instance.

        Parameters
        ----------
        angle_intvl : `Interval` or 1-dim. `IntervalProd`
            The motion parameters given in radians
        dparams : `Interval` or 1-dim. `IntervalProd`
            The detector parameters
        agrid : 1-dim. `TensorGrid`, optional
            A sampling grid for the `angle_intvl`. Default: `None`
        dgrid : 1-dim. `TensorGrid`, optional
            A sampling grid for the detector parameters. Default: `None`
        origin_to_det : 2-element array, optional
            The direction from the origin to the point (0) of the detector
            when angle=0
        """

        direction = np.array(origin_to_det) / np.linalg.norm(origin_to_det)

        # Only one option since this is easily modified in data space otherwise
        detector_axis = np.array([-direction[1], direction[0]])

        detector = Flat1dDetector(dparams, detector_axis, dgrid)
        super().__init__(2, angle_intvl, detector, direction, agrid)

    def rotation_matrix(self, angle):
        """The detector rotation function.

        Parameters
        ----------
        angle : `float`
            The motion parameters given in radians. It must be contained in
            this geometry's `motion_params`

        Returns
        -------
        rot : `numpy.ndarray`, shape (2, 2)
            The rotation matrix mapping the standard basis vectors in
            the fixed ("lab") coordinate system to the basis vectors of
            the local coordinate system of the detector reference point,
            expressed in the fixed system.
        """
        angle = float(angle)
        if angle not in self.motion_params:
            raise ValueError('angle {} not in the valid range {}.'
                             ''.format(angle, self.motion_params))
        return euler_matrix(angle)

    def __repr__(self):
        """Returns ``repr(self)``."""
        inner_fstr = '{!r}, {!r}'
        if self.has_motion_sampling:
            inner_fstr += ',\n    agrid={agrid!r}'
        if self.has_det_sampling:
            inner_fstr += ',\n    dgrid={dgrid!r}'
        if not np.allclose(self.origin_to_det, [1, 0]):
            inner_fstr += ',\n    origin_to_det={origin_to_det!r}'

        inner_str = inner_fstr.format(self.motion_params,
                                      self.det_params,
                                      agrid=self.motion_grid,
                                      dgrid=self.det_grid,
                                      origin_to_det=self.origin_to_det)
        return '{}({})'.format(self.__class__.__name__, inner_str)


class Parallel3dGeometry(ParallelGeometry, AxisOrientedGeometry):

    """Parallel beam geometry in 3d.

    The motion parameter is the rotation angle around the 3rd unit axis,
    and the detector is a flat 2d detector perpendicular to the ray direction.

    """

    def __init__(self, angle_intvl, dparams, agrid=None, dgrid=None,
                 axis=[0, 0, 1], origin_to_det=[1, 0, 0], detector_axes=None):
        """Initialize a new instance.

        Parameters
        ----------
        angle_intvl : `Interval` or 1-dim. `IntervalProd`
            The motion parameters given in radians
        dparams : `Rectangle` or 2-dim. `IntervalProd`
            The detector parameters
        agrid : 1-dim. `TensorGrid`, optional
            A sampling grid for `angle_intvl`. Default: `None`
        dgrid : 2-dim. `TensorGrid`, optional. Default: `None`
            A sampling grid for `dparams`
        axis : 3-element array, optional
            Fixed rotation axis defined by a 3-element vector
        origin_to_det : 3-element array, optional
            The direction from the origin to the point (0, 0) of the detector
            angle=0
        detector_axes : sequence of two 3-element arrays, optional
            Unit directions along each detector parameter of the detector.
            Default: (normalized) [np.cross(axis, origin_to_detector), axis]
        """
        AxisOrientedGeometry.__init__(self, axis)

        direction = np.array(origin_to_det) / np.linalg.norm(origin_to_det)

        if detector_axes is None:
            detector_axes = [np.cross(self.axis, direction), self.axis]

        detector = Flat2dDetector(dparams, detector_axes, dgrid)
        ParallelGeometry.__init__(self, 3, angle_intvl, detector, direction,
                                  agrid)

    def __repr__(self):
        """Returns ``repr(self)``."""
        arg_fstr = '{!r}, {!r}'
        if self.has_motion_sampling:
            arg_fstr += ',\n    agrid={agrid!r}'
        if self.has_det_sampling:
            arg_fstr += ',\n    dgrid={dgrid!r}'
        if not np.allclose(self.axis, [0, 0, 1]):
            arg_fstr += ',\n    axis={axis!r}'
        if not np.allclose(self.origin_to_det, [1, 0, 0]):
            arg_fstr += ',\n    src_to_det={src_to_det!r}'

        default_axes = [np.cross(self.axis, self.origin_to_det), self.axis]
        if not np.allclose(self.detector.detector_axes, default_axes):
            arg_fstr += ',\n    detector_axes={detector_axes!r}'

        arg_str = arg_fstr.format(self.motion_params,
                                  self.det_params,
                                  agrid=self.motion_grid,
                                  dgrid=self.det_grid,
                                  axis=self.axis,
                                  src_to_det=self.src_to_det,
                                  detector_axes=self.detector.detector_axes)
        return '{}({})'.format(self.__class__.__name__, arg_str)

    # Fix for bug in ABC thinking this is abstract
    rotation_matrix = AxisOrientedGeometry.rotation_matrix
