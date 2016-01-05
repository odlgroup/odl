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

"""Cone beam related geometries."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from abc import ABCMeta
from future import standard_library
from future.utils import with_metaclass
standard_library.install_aliases()
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


class ConeBeamGeometry(with_metaclass(ABCMeta, Geometry)):

    """Abstract 3d cone beam geometry.

    The source moves on a circle with radius ``r``, and the detector
    reference point is opposite to the source on a circle with radius ``R``.

    The motion parameter is the (1d) rotation angle parametrizing source and
    detector positions.
    """

    def __init__(self, angle_intvl, src_radius, det_radius, agrid=None,
                 axis=None):
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
            A sampling grid for the `angle_intvl`
        axis : `int` or 3-element array
            Defines the rotation axis via a 3-element vector or a single
            integer referring to a standard axis
        """
        if not (isinstance(angle_intvl, IntervalProd) and
                angle_intvl.ndim == 1):
            raise TypeError('angle parameters {!r} are not an interval.'
                            ''.format(angle_intvl))

        src_radius = float(src_radius)
        if src_radius <= 0:
            raise ValueError('source circle radius {} is not positive.'
                             ''.format(src_radius))
        det_radius = float(det_radius)
        if det_radius <= 0:
            raise ValueError('detector circle radius {} is not positive.'
                             ''.format(det_radius))

        if agrid is not None:
            if not isinstance(agrid, TensorGrid):
                raise TypeError('angle grid {!r} is not a `TensorGrid` '
                                'instance.'.format(agrid))
            if not angle_intvl.contains_set(agrid):
                raise ValueError('angular grid {} not contained in angle '
                                 'interval {}.'.format(agrid, angle_intvl))

        super().__init__()
        self._motion_params = angle_intvl
        self._src_radius = src_radius
        self._det_radius = det_radius
        self._motion_grid = agrid
        self._axis = axis

    @property
    def motion_params(self):
        """Motion parameters of this geometry."""
        return self._motion_params

    @property
    def motion_grid(self):
        """Sampling grid for this geometry's motion parameters."""
        return self._motion_grid

    @property
    def angle_intvl(self):
        """Angles (= motion parameters) of this geometry."""
        return self._motion_params

    @property
    def angle_grid(self):
        """Angle (= motion parameter) sampling grid of this geometry."""
        return self._motion_grid

    @property
    def src_radius(self):
        """Source circle radius of this geometry."""
        return self._src_radius

    @property
    def det_radius(self):
        """Detector circle radius of this geometry."""
        return self._det_radius

    @property
    def ndim(self):
        """Number of dimensions of this geometry."""
        return 3

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


    def __repr__(self):
        """Return ``repr(self)``"""
        inner_fstr = '{!r}, {!r}, src_rad={}, det_rad={}'
        if self.has_motion_sampling:
            inner_fstr += ',\n agrid={agrid!r}'
        if self.has_det_sampling:
            inner_fstr += ',\n dgrid={dgrid!r}'
        inner_str = inner_fstr.format(self.motion_params, self.det_params,
                                      self.src_radius, self.det_radius,
                                      agrid=self.motion_grid,
                                      dgrid=self.det_grid)
        return '{}({})'.format(self.__class__.__name__, inner_str)


class ConeFlatGeometry(ConeBeamGeometry):
    """Cone beam geometry in 3d with flat detector.

    The source moves on a circle with radius ``r``, and the detector
    reference point is opposite to the source on a circle with radius ``R``
    and aligned tangential to the circle.

    The motion parameter is the (1d) rotation angle parametrizing source and
    detector positions.
    """

    def __init__(self, angle_intvl, dparams, src_radius, det_radius, agrid=None,
                 dgrid=None, axis=None):
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
            A sampling grid for `angle_intvl`
        dgrid : 2-dim. `TensorGrid`, optional
            A sampling grid for `dparams`
        axis : `int` or 3-element array
            Defines the rotation axis via a 3-element vector or a single
            integer referring to a standard axis
        """
        super().__init__(angle_intvl, src_radius, det_radius, agrid, axis)

        if not (isinstance(dparams, IntervalProd) and dparams.ndim == 2):
            raise TypeError('detector parameters {!r} are not an interval.'
                            ''.format(dparams))

        if dgrid is not None:
            if not isinstance(dgrid, TensorGrid):
                raise TypeError('detector grid {!r} is not a `TensorGrid` '
                                'instance.'.format(dgrid))
            if not dparams.contains_set(dgrid):
                raise ValueError('detector grid {} not contained in detector '
                                 'parameter interval {}.'
                                 ''.format(dgrid, dparams))

        self._detector = Flat2dDetector(dparams, dgrid)

    @property
    def detector(self):
        """Detector of this geometry."""
        return self._detector


class CircularConeFlatGeometry(ConeFlatGeometry):
    """Cone beam geometry with circular acquisition and flat detector.

    The source moves on a circle with radius ``r``, and the detector
    reference point is opposite to the source on a circle with radius ``R``
    and aligned tangential to the circle.

    The motion parameter is the (1d) rotation angle parametrizing source and
    detector positions.
    """

    def __init__(self, angle_intvl, dparams, src_radius, det_radius, agrid=None,
                 dgrid=None, axis=None):
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
        axis : `int` or 3-element array, optional
            Defines the rotation axis via a 3-element vector or a single
            integer referring to a standard axis. Default: `None`
        """

        super().__init__(angle_intvl, dparams, src_radius, det_radius, agrid,
                         dgrid, axis)

    @property
    def axis(self):
        """The normalized axis of rotation.

        Returns
        -------
        axis : `numpy.ndarray`, shape (3,)
            The normalized rotation axis
        """
        axis = self._axis
        if axis is None:
            axis = np.array([0, 0, 1])
        elif np.isscalar(axis):
            ind = axis
            axis = np.zeros(3)
            axis[ind] = 1
        elif len(axis) == 3:
            axis = np.array(axis) / np.linalg.norm(axis)
        else:
            raise ValueError('`axis` intializer {} has wrong format'.format(
                    axis))

        return axis

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
        if angle not in self.motion_params:
            raise ValueError('angle {} not in the valid range {}.'
                             ''.format(angle, self.motion_params))

        # ASTRA 'cone_vec' convention
        return self.det_radius * np.array([-np.sin(angle), np.cos(angle), 0])

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
        if angle not in self.motion_params:
            raise ValueError('angle {} not in the valid range {}.'
                             ''.format(angle, self.motion_params))

        # ASTRA cone_vec convention
        return self.src_radius * np.array([np.sin(angle), -np.cos(angle), 0])

        # TODO: backprojection weighting function?

    def det_to_src(self, angle, dpar, normalized=True):
        """Direction from a detector location to the source.

        Parameters
        ----------
        angle : `float`
            The motion parameter given in radian. It must be contained in this
            geometry's motion parameter set
        dpar : 2-tuple of `float`
            The detector parameter. It must be contained in this
            geometry's detector parameter set
        normalized : `bool`, optional
            If `False` return the vector from the detector point to the source
            parametrized by ``dpar`` and ``angle``. If `True`, return the
            normalized version of that vector. Default: `True`

        Returns
        -------
        vec : `numpy.ndarray`, shape (3,)
            (Unit) vector pointing from the detector to the source
        """
        if angle not in self.motion_params:
            raise ValueError('angle {} is not in the valid range {}.'
                             ''.format(angle, self.motion_params))
        if dpar not in self.det_params:
            raise ValueError('detector parameter {} is not in the valid '
                             'range {}.'.format(dpar, self.det_params))

        axis = self.axis

        # Angle of a detector point at `dpar` as seen from the source relative
        # to the line from the source to the detector reference point
        det_pt_angle = np.arctan2(dpar, self.src_radius + self.det_radius)

        # vector for spiral along z-direction
        vec = -np.array(
                [np.cos(det_pt_angle[1]) * np.cos(angle + det_pt_angle),
                 np.cos(det_pt_angle[1]) * np.sin(angle + det_pt_angle),
                 np.sin(det_pt_angle[1])])

        # rotate vector
        if not (axis[0] == 0 and axis[1] == 0):
            vec = -np.array(
                    self.rotation_matrix(angle)[0]).squeeze()

        if not normalized:
            vec *= self.src_radius + self.det_radius

        return vec


class HelicalConeFlatGeometry(ConeFlatGeometry):
    """Cone beam geometry with helical acquisition and flat detector.

    The source moves along a spiral with radius ``r`` in the azimuthal plane
    and a pitch factor ``P``. The detector reference point is opposite to
    the source and moves on a spiral with radius ``R`` in the azimuthal
    plane and pitch factor ``P``. The detector is aligned tangential to the
    circle.

    The motion parameter is the (1d) rotation angle parametrizing source and
    detector positions.
    """

    def __init__(self, angle_intvl, dparams, src_radius, det_radius,
                 spiral_pitch_factor, agrid=None, dgrid=None, axis=None):
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
        spiral_pitch_factor : `float`
            Dimensionless factor given by the table feed per rotation
            divided by the total collimation width. The total collimation
            width is considered at isocenter and  given by the slice thickness
            for a single-slice spiral acquisition and by the number of
            detector rows times the slice thickness for multiple slice
            spiral acquisition.
        agrid : 1-dim. `TensorGrid`, optional
            A sampling grid for `angle_intvl`. Default: `None`
        dgrid : 2-dim. `TensorGrid`, optional
            A sampling grid for `dparams`. Default: `None`
        axis : `int` or 3-element array, optional
            Defines the rotation axis via a 3-element vector or a single
            integer referring to a standard axis. Default: `None`
        """

        super().__init__(angle_intvl, dparams, src_radius, det_radius, agrid,
                         dgrid, axis)
        self._axis = axis
        det_height = (dparams.max() - dparams.min())[1]
        self._table_feed_per_rotation = spiral_pitch_factor * src_radius / (
            src_radius + det_radius) * det_height

    @property
    def axis(self):
        """The axis of rotation.

        Returns
        -------
        axis : `numpy.ndarray`, shape (3,)
            The rotation axis
        """
        axis = self._axis
        if axis is None:
            axis = np.array([0, 0, 1])
        elif np.isscalar(axis):
            ind = axis
            axis = np.zeros(3)
            axis[ind] = 1
        elif len(axis) == 3:
            axis = np.array(axis) / np.linalg.norm(axis)
        else:
            raise ValueError('`axis` intializer {} has wrong format'.format(
                    axis))

        return axis

    @property
    def table_feed_per_rotation(self):
        """Table feed per 360 degree rotation."""
        return self._table_feed_per_rotation

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
            phi), z)`` where ``z`` is given by the table feed
        """
        if angle not in self.motion_params:
            raise ValueError('angle {} is not in the valid range {}.'
                             ''.format(angle, self.motion_params))

        # ASTRA cone_vec geometries
        return np.array([
            -self.det_radius * np.sin(angle),
            self.det_radius * np.cos(angle),
            self.table_feed_per_rotation * angle / (2 * np.pi)])

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
            factor ``P`` at a given rotation angle ``phi`` defined as
            ``(r * sin(phi), -r * cos(phi), z)`` where ``z`` is given by the
            table feed
        """
        if angle not in self.motion_params:
            raise ValueError('angle {} is not in the valid range {}.'
                             ''.format(angle, self.motion_params))

        # ASTRA cone_vec geometries
        return np.array([
            self.src_radius * np.sin(angle),
            -self.src_radius * np.cos(angle),
            self.table_feed_per_rotation * angle / (2 * np.pi)])

        # TODO: backprojection weighting function?

    def det_to_src(self, angle, dpar, normalized=True):
        """Direction from a detector location to the source.

        Parameters
        ----------
        angle : `float`
            The motion parameter given in radian. It must be contained in this
            geometry's motion parameter set
        dpar : 2-tuple of `float`
            The detector parameter. It must be contained in this
            geometry's detector parameter set
        normalized : `bool`, optional
            If `False` return the vector from the detector point to the source
            parametrized by ``dpar`` and ``angle``. If `True`, return the
            normalized version of that vector. Default: `True`

        Returns
        -------
        vec : `numpy.ndarray`, shape (3,)
            (Unit) vector pointing from the detector to the source
        """
        if angle not in self.motion_params:
            raise ValueError('angle {} is not in the valid range {}.'
                             ''.format(angle, self.motion_params))
        if dpar not in self.det_params:
            raise ValueError('detector parameter {} is not in the valid '
                             'range {}.'.format(dpar, self.det_params))

        axis = self.axis

        # Angle of a detector point at `dpar` as seen from the source relative
        # to the line from the source to the detector reference point
        det_pt_angle = np.arctan2(dpar, self.src_radius + self.det_radius)

        # vector for spiral along z-direction
        vec = -np.array(
                [np.cos(det_pt_angle[1]) * np.cos(angle + det_pt_angle),
                 np.cos(det_pt_angle[1]) * np.sin(angle + det_pt_angle),
                 np.sin(det_pt_angle[1])])

        # rotate vector
        if not (axis[0] == 0 and axis[1] == 0):
            vec = -np.array(
                    self.rotation_matrix(angle)[0]).squeeze()

        if not normalized:
            vec *= self.src_radius + self.det_radius

        return vec
