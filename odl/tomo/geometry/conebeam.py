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

"""Cone beam geometries."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from abc import abstractmethod
from builtins import super

# External
import numpy as np

# Internal
from odl.set.domain import IntervalProd
from odl.discr.grid import TensorGrid
from odl.tomo.geometry.detector import Flat2dDetector
from odl.tomo.geometry.geometry import Geometry


__all__ = ('ConeBeamGeometry', 'CircularConeFlatGeometry',
           'HelicalConeFlatGeometry',)


class ConeBeamGeometry(Geometry):

    """Abstract nd cone beam geometry.

    A cone beam geometry is characterized by a source and a
    (n-1)d detector moving in space according to a 1d motion parameter.
    """

    def __init__(self, ndim, angle_intvl, detector, agrid=None):
        """Initialize a new instance.

        Parameters
        ----------
        ndim : {1, 2}
            number of dimensions of geometry
        angle_intvl : 1d `IntervalProd`
            Admissible angles.
        detector : `Detector`
            The detector to use.
        agrid : `TensorGrid`, optional
            Optional discretization of the angle_intvl
        """
        if not (isinstance(angle_intvl, IntervalProd) and
                angle_intvl.ndim == 1):
            raise TypeError('angle parameters {!r} are not an interval.'
                            ''.format(angle_intvl))
        self._motion_params = angle_intvl
        self._motion_grid = agrid
        self._detector = detector

        if agrid is not None:
            if not isinstance(agrid, TensorGrid):
                raise TypeError('angle grid {!r} is not a `TensorGrid` '
                                'instance.'.format(agrid))
            if not angle_intvl.contains_set(agrid):
                raise ValueError('angular grid {} not contained in angle '
                                 'interval {}.'.format(agrid, angle_intvl))

        super().__init__(ndim=ndim)

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

    @abstractmethod
    def src_position(self, mpar):
        """The source position function.

        Parameters
        ----------
        mpar : element of motion parameters `motion_params`
            Motion parameter for which to calculate the source position

        Returns
        -------
        pos : `numpy.ndarray`, shape (`ndim`,)
            The source position, a `ndim`-dimensional vector
        """
        raise NotImplementedError

    def det_to_src(self, mpar, dpar, normalized=True):
        """Vector pointing from a detector location to the source.

        A function of the motion and detector parameters.

        The default implementation uses the `det_point_position` and
        `src_position` functions. Implementations can override this, for
        example if no source position is given.

        Parameters
        ----------
        mpar : element of motion parameters `motion_params`
            Motion parameter at which to evaluate
        dpar : element of detector parameters `det_params`
            Detector parameter at which to evaluate
        normalized : `bool`, optional
            If `True`, return a normalized (unit) vector. Default: `True`

        Returns
        -------
        vec : `numpy.ndarray`, shape (`ndim`,)
            (Unit) vector pointing from the detector to the source
        """

        if mpar not in self.motion_params:
            raise ValueError('mpar {} is not in the valid range {}.'
                             ''.format(mpar, self.motion_params))
        if dpar not in self.det_params:
            raise ValueError('detector parameter {} is not in the valid '
                             'range {}.'.format(dpar, self.det_params))

        vec = self.det_point_position(mpar, dpar) - self.src_position(mpar)

        if normalized:
            vec /= np.linalg.norm(vec, axis=-1)

        return vec

    def __repr__(self):
        """Return ``repr(self)``"""

        # TODO

        inner_fstr = '{!r}, {!r}, src_radius={}, det_radius={}'
        if self.has_motion_sampling:
            inner_fstr += ',\n agrid={agrid!r}'
        if self.has_det_sampling:
            inner_fstr += ',\n dgrid={dgrid!r}'
        inner_str = inner_fstr.format(self.motion_params, self.det_params,
                                      self.src_radius, self.det_radius,
                                      agrid=self.motion_grid,
                                      dgrid=self.det_grid)
        return '{}({})'.format(self.__class__.__name__, inner_str)


class AxisOrientedGeometry(object):

    """ Mixin class for cone beam geometries oriented according to an axis.
    """

    def __init__(self, axis):
        """Initialize a new instance.

        Parameters
        ----------
        axis : 3-element array, optional
            Defines the rotation axis via a 3-element vector.
        """

        self._axis = np.array(axis) / np.linalg.norm(axis)

    @property
    def axis(self):
        """The normalized axis of rotation.

        Returns
        -------
        axis : `numpy.ndarray`, shape (3,)
            The normalized rotation axis
        """
        return self._axis

    def rotation_matrix(self, angle):
        """The detector rotation function.

        Returns the matrix for rotating a vector in 3d by an angle ``angle``
        about the rotation axis given by the property `axis` according to
        the right hand rule.

        The matrix is computed according to
        `Rodrigues' rotation formula
        <https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula>`_.

        Parameters
        ----------
        angle : `float`
            The motion parameter given in radian. It must be
            contained in this geometry's `motion_params`.

        Returns
        -------
        rot_mat : `numpy.ndarray`, shape ``(3, 3)``
            The rotation matrix mapping the standard basis vectors in
            the fixed ("lab") coordinate system to the basis vectors of
            the local coordinate system of the detector reference point,
            expressed in the fixed system.
        """

        angle = float(angle)
        if angle not in self.motion_params:
            raise ValueError('angle {} is not in the valid range {}.'
                             ''.format(angle, self.motion_params))

        axis = self.axis

        cross_mat = np.array([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]])
        dy_mat = np.outer(axis, axis)
        id_mat = np.eye(3)
        cos_ang = np.cos(angle)
        sin_ang = np.sin(angle)

        return cos_ang * id_mat + (1. - cos_ang) * dy_mat + sin_ang * cross_mat


class CircularConeFlatGeometry(ConeBeamGeometry, AxisOrientedGeometry):
    """Cone beam geometry with circular acquisition and flat detector.

    The source moves on a circle with radius ``r``, and the detector
    reference point is opposite to the source on a circle with radius ``R``
    and aligned tangential to the circle.

    The motion parameter is the (1d) rotation angle parametrizing source and
    detector positions.
    """

    def __init__(self, angle_intvl, dparams, src_radius, det_radius,
                 agrid=None, dgrid=None,
                 axis=[0, 0, 1], source_to_detector=[1, 0, 0],
                 detector_axises=None):
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
            A sampling grid for `angle_intvl`. Default: `None`
        dgrid : 2-dim. `TensorGrid`, optional
            A sampling grid for ``dparams``. Default: `None`
        axis : 3-element array, optional
            Defines the rotation axis via a 3-element vector.
        source_to_detector : 3-element array, optional
            Defines the direction from the source to the point (0,0) of the
            detector.
        detector_axises : tuple of two 3-element arrays, optional
            Defines the unit directions along each detector parameter of the
            detector.
            Default: (normalized) [np.cross(axis, source_to_detector), axis]
        """

        AxisOrientedGeometry.__init__(self, axis)

        self._source_to_detector = (np.array(source_to_detector) /
                                    np.linalg.norm(source_to_detector))

        if detector_axises is None:
            detector_axises = [np.cross(self.axis, self._source_to_detector),
                               self.axis]

        detector = Flat2dDetector(dparams, detector_axises, dgrid)

        ConeBeamGeometry.__init__(self, 3, angle_intvl, detector, agrid)

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

    def det_refpoint(self, angle):
        """The detector reference point function.

        Parameters
        ----------
        angle : `float`
            The motion parameter given in radian. It must be contained
            in this geometry's motion parameter set

        Returns
        -------
        point : `numpy.ndarray`, shape (3,)
            The reference point on the circle with radius ``R`` at a given
            rotation angle ``phi`` defined as ``R(-sin(phi), cos(phi), 0)``
        """
        angle = float(angle)
        if angle not in self.motion_params:
            raise ValueError('angle {} is not in the valid range {}.'
                             ''.format(angle, self.motion_params))

        offset = self.det_radius * self._source_to_detector
        return self.rotation_matrix(angle).dot(offset)

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
            The source position at ``z`` on the circle with radius ``r`` at
            a given rotation angle ``phi`` defined as `r * (sin(phi),
            -cos(phi), 0)``
        """
        angle = float(angle)
        if angle not in self.motion_params:
            raise ValueError('angle {} is not in the valid range {}.'
                             ''.format(angle, self.motion_params))

        offset = -self.src_radius * self._source_to_detector
        return self.rotation_matrix(angle).dot(offset)

    # Fix for bug in ABC thinking this is abstract
    rotation_matrix = AxisOrientedGeometry.rotation_matrix


class HelicalConeFlatGeometry(ConeBeamGeometry, AxisOrientedGeometry):
    """Cone beam geometry with helical acquisition and flat detector.

    The source moves along a spiral with radius ``r`` in the azimuthal plane
    and a pitch``P``. The detector reference point is opposite to
    the source and moves on a spiral with radius ``R`` in the azimuthal
    plane and pitch ``P``. The detector is aligned tangential to the
    circle.

    The motion parameter is the (1d) rotation angle parametrizing source and
    detector positions.
    """

    def __init__(self, angle_intvl, dparams, src_radius, det_radius,
                 pitch, agrid=None, dgrid=None,
                 axis=[0, 0, 1], source_to_detector=[1, 0, 0],
                 detector_axises=None):
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
            A sampling grid for `angle_intvl`. Default: `None`
        dgrid : 2-dim. `TensorGrid`, optional
            A sampling grid for `dparams`. Default: `None`
        axis : `int` or 3-element array, optional
            Defines the rotation axis via a 3-element vector or a single
            integer referring to a standard axis. Default: `None`
        source_to_detector : 3-element array, optional
            Defines the direction from the source to the point (0,0) of the
            detector.
        detector_axises : tuple of two 3-element arrays, optional
            Defines the unit directions along each detector parameter of the
            detector.
            Default: (normalized) [np.cross(axis, source_to_detector), axis]
        """

        AxisOrientedGeometry.__init__(self, axis)

        self._source_to_detector = (np.array(source_to_detector) /
                                    np.linalg.norm(source_to_detector))

        if detector_axises is None:
            detector_axises = [np.cross(self.axis, self._source_to_detector),
                               self.axis]

        detector = Flat2dDetector(dparams, detector_axises, dgrid)

        ConeBeamGeometry.__init__(self, 3, angle_intvl, detector, agrid)

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
        offset = self.det_radius * self._source_to_detector
        detector_offset = self.rotation_matrix(angle).dot(offset)

        # Increment by pitch
        ptich_offset = self.axis * angle / (np.pi * 2)

        return detector_offset + ptich_offset

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

        # ASTRA cone_vec geometries
        return np.array([
            self.src_radius * np.sin(angle),
            -self.src_radius * np.cos(angle),
            self.pitch * angle / (2 * np.pi)])

        # Distance from 0 to detector
        offset = -self.src_radius * self._source_to_detector
        source_offset = self.rotation_matrix(angle).dot(offset)

        # Increment by pitch
        pitch_offset = self.axis * angle / (np.pi * 2)

        return source_offset + pitch_offset

    # Fix for bug in ABC thinking this is abstract
    rotation_matrix = AxisOrientedGeometry.rotation_matrix
