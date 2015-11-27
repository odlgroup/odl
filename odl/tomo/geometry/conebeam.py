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

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super
from future.utils import with_metaclass

# External
from abc import ABCMeta
import numpy as np
from math import cos, sin

# Internal
from odl.set.domain import IntervalProd
from odl.discr.grid import TensorGrid
from odl.tomo.geometry.detector import Flat2dDetector
from odl.tomo.geometry.geometry import Geometry
# from odltomo.util.trafos import euler_matrix


__all__ = ('CircularConeFlatGeometry', 'HelicalConeFlatGeometry')


class ConeBeamGeometry(with_metaclass(ABCMeta, Geometry)):

    """Abstract 3d cone beam geometry.

    The source moves on a circle with radius `r`, and the detector
    reference point is opposite to the source on a circle with radius
    `R`.

    The motion parameter is the (1d) rotation angle parametrizing source and
    detector positions.

    """

    def __init__(self, angle_intvl, src_rad, det_rad, agrid=None,
                 angle_offset=0, axis=None):
        """Initialize a new instance.

        Parameters
        ----------
        angle_intvl : `Interval` or 1-dim. `IntervalProd`
            The motion parameters
        src_rad : positive float
            Radius of the source circle, must be positive
        det_rad : positive float
            Radius of the detector circle, must be positive
        agrid : 1-dim. `TensorGrid`, optional
            A sampling grid for the `angle_intvl`
        angle_offset : float
            Offset to the rotation angle in the azimuthal plane. Does not
            imply an offset in z-direction.
        axis : int or 3-element array
            Defines the rotation axis via a 3-element vector or a single
            integer referring to a standard axis
        """
        if not (isinstance(angle_intvl, IntervalProd) and
                angle_intvl.ndim == 1):
            raise TypeError('angle parameters {!r} are not an interval.'
                            ''.format(angle_intvl))

        src_rad = float(src_rad)
        if src_rad <= 0:
            raise ValueError('source circle radius {} is not positive.'
                             ''.format(src_rad))
        det_rad = float(det_rad)
        if det_rad <= 0:
            raise ValueError('detector circle radius {} is not positive.'
                             ''.format(det_rad))

        if agrid is not None:
            if not isinstance(agrid, TensorGrid):
                raise TypeError('angle grid {!r} is not a `TensorGrid` '
                                'instance.'.format(agrid))
            if not angle_intvl.contains_set(agrid):
                raise ValueError('angular grid {} not contained in angle '
                                 'interval {}.'.format(agrid, angle_intvl))

        angle_offset = float(angle_offset)

        super().__init__()
        self._motion_params = angle_intvl
        self._src_radius = src_rad
        self._det_radius = det_rad
        self._motion_grid = agrid
        self._motion_params_offset = angle_offset
        self._axis = axis

    @property
    def motion_params(self):
        """Motion parameters of this geometry."""
        return self._motion_params

    @property
    def motion_params_offset(self):
        """Offset to motion parameters. """
        return self._motion_params_offset

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
    def angle_offset(self):
        """Offset to the rotation angle in the azimuthal plane. Does not
        imply an offset in z-direction. The actual angles then reside
        within `angle_offset` + `angle_intvl`."""
        return self._motion_params_offset

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

    @property
    def axis(self):
        """The axis of rotation.

        Returns
        -------
        axis : `numpy.ndarray`, shape `(3,)`
            The rotation axis
        """
        axis = self._axis
        if axis is None:
            axis = np.array([0, 0, 1])
        elif isinstance(axis, (int, float)):
            ind = axis
            axis = np.zeros(3)
            axis[ind] = 1
        elif len(axis) == 3:
            axis = np.array(axis)
        else:
            raise ValueError('`axis` intializer {} has wrong format'.format(
                axis))

        return axis

    def det_rotation(self, angle):
        """The detector rotation function. Returns the matrix for rotating a
            vector in 3d by an angle `angle` about the rotation axis given by
            the property `axis` according to the right hand rule.

            The matrix is computed according to `Rodriguez' rotation formula`_.
            .. _Rodriguez' rotation formula:
                https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula

            Parameters
            ----------
            angle : float
                The motion parameter. It must be contained in this
                geometry's `motion_params`.

            Returns
            -------
            rot_mat : `numpy.matrix`, shape `(3, 3)`
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

        cross_mat = np.matrix([[0, -axis[2], axis[1]],
                               [axis[2], 0, -axis[0]],
                               [-axis[1], axis[0], 0]])
        dy_mat = np.asmatrix(np.outer(axis, axis))
        id_mat = np.asmatrix(np.eye(3))
        cos_ang = cos(angle)
        sin_ang = sin(angle)

        return cos_ang * id_mat + (1. - cos_ang) * dy_mat + sin_ang * cross_mat

        # return euler_matrix(angle)

    def __repr__(self):
        """`g.__repr__() <==> repr(g)`."""
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

    def __str__(self):
        """`g.__str__() <==> str(g)`."""
        return self.__repr__()  # TODO: prettify


class ConeFlatGeometry(ConeBeamGeometry):
    """Cone beam geometry in 3d with flat detector.

    The source moves on a circle with radius `r`, and the detector
    reference point is opposite to the source on a circle with radius
    `R` and aligned tangential to the circle.

    The motion parameter is the (1d) rotation angle parametrizing
    source and detector positions.
    """

    def __init__(self, angle_intvl, dparams, src_rad, det_rad, agrid=None,
                 dgrid=None, angle_offset=0, axis=None):
        """Initialize a new instance.

        Parameters
        ----------
        angle_intvl : `Interval` or 1-dim. `IntervalProd`
            The motion parameters
        dparams : `Rectangle` or 2-dim. `IntervalProd`
            The detector parameters
        src_rad : float
            Radius of the source circle, must be positive
        det_rad : float
            Radius of the detector circle, must be positive
        agrid : 1-dim. `TensorGrid`, optional
            A sampling grid for `angle_intvl`
        dgrid : 2-dim. `TensorGrid`, optional
            A sampling grid for `dparams`
        angle_offset : float
            Offset to the rotation angle in the azimuthal plane. Does not
            affect the offset in z-direction.
        axis : int or 3-element array
            Defines the rotation axis via a 3-element vector or a single
            integer referring to a standard axis
        """
        super().__init__(angle_intvl, src_rad, det_rad, agrid, angle_offset,
                         axis)

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

    def det_to_src(self, angle, dpar, normalized=True):
        """Direction from a detector location to the source.

        Parameters
        ----------
        angle : float
            The motion parameter. It must be contained in this
            geometry's motion parameter set.
        dpar : 2-tuple of floats
            The detector parameter. It must be contained in this
            geometry's detector parameter set.
        normalized : bool
            If `False` return the vector from the detector point
            parametrized by `dpar` and `angle` to the source at
            `angle`. If `True`, return the normalized version of
            that vector.

        Returns
        -------
        vec : ndarray, shape `(ndim,)`
            (Unit) vector pointing from the detector to the source
        """
        if angle not in self.motion_params:
            raise ValueError('angle {} is not in the valid range {}.'
                             ''.format(angle, self.motion_params))
        if dpar not in self.det_params:
            raise ValueError('detector parameter {} is not in the valid '
                             'range {}.'.format(dpar, self.det_params))


        axis = self.axis
        angle += self.angle_offset

        # Angle of a detector point at 'dpar' as seen from the source relative
        # to the line from the source to the detector reference point
        det_pt_angle = np.arctan2(dpar, self.src_radius + self.det_radius)


        # vector for spiral along z-direction
        vec = -np.array(
            [cos(det_pt_angle[1]) * cos(angle + det_pt_angle),
             cos(det_pt_angle[1]) * sin(angle + det_pt_angle),
             sin(det_pt_angle[1])])

        # rotate vector
        if not (axis[0] == 0 and axis[1] == 0):
            vec = -np.array(
                self.det_rotation(angle + self.angle_offset)[0]).squeeze()




        if not normalized:
            vec *= self.src_radius + self.det_radius

        return vec


class CircularConeFlatGeometry(ConeFlatGeometry):
    """Circular cone beam geometry in 3d with flat detector.

    The source moves on a circle with radius `r`, and the detector
    reference point is opposite to the source on a circle with radius
    `R` and aligned tangential to the circle.

    The motion parameter is the (1d) rotation angle parametrizing
    source and detector positions.
    """

    def __init__(self, angle_intvl, dparams, src_rad, det_rad, agrid=None,
                 dgrid=None, angle_offset=0, axis=None):
        """Initialize a new instance.

        Parameters
        ----------
        angle_intvl : `Interval` or 1-dim. `IntervalProd`
            The motion parameters
        dparams : `Rectangle` or 2-dim. `IntervalProd`
            The detector parameters
        src_rad : float
            Radius of the source circle, must be positive
        det_rad : float
            Radius of the detector circle, must be positive
        agrid : 1-dim. `TensorGrid`, optional
            A sampling grid for `angle_intvl`
        dgrid : 2-dim. `TensorGrid`, optional
            A sampling grid for `dparams`
        angle_offset : float
            Offset to the rotation angle in the azimuthal plane. Does not
            imply an offset in z-direction.
        axis : int or 3-element array
            Defines the rotation axis via a 3-element vector or a single
            integer referring to a standard axis
        """

        super().__init__(angle_intvl, dparams, src_rad, det_rad, agrid,
                         dgrid, angle_offset, axis)

    def det_refpoint(self, angle):
        """The detector reference point function.

        Parameters
        ----------
        angle : float
            The motion parameter. It must be contained in this
            geometry's motion parameter set.

        Returns
        -------
        point : ndarray, shape `(ndim,)`
            The reference point on the circle with radius `R` at
            the given rotation angle :math:`\phi`, defined as
            :math:`R(\cos\phi, \sin\phi)`
        """
        if angle not in self.motion_params:
            raise ValueError('angle {} not in the valid range {}.'
                             ''.format(angle, self.motion_params))

        angle += self.angle_offset
        return self.det_radius * np.array([cos(angle), sin(angle), 0])

    def src_position(self, angle):
        """The source position function.

        Parameters
        ----------
        angle : float
            The motion parameter. It must be contained in this
            geometry's motion parameter set.

        Returns
        -------
        point : ndarray, shape `(ndim,)`
            The source position at `z` on the circle with radius `r` at
            the given rotation angle :math:`\phi`, defined as
            :math:`-r(\cos\phi, \sin\phi, z)`
        """
        if angle not in self.motion_params:
            raise ValueError('angle {} not in the valid range {}.'
                             ''.format(angle, self.motion_params))

        angle += self.angle_offset
        return -self.src_radius * np.array([cos(angle), sin(angle), 0])

    # TODO: backprojection weighting function?


class HelicalConeFlatGeometry(ConeFlatGeometry):
    """Helical cone beam geometry in 3d with flat detector.

    The source moves on a circle with radius `r`, and the detector
    reference point is opposite to the source on a circle with radius
    `R` and aligned tangential to the circle.

    The motion parameter is the (1d) rotation angle parametrizing
    source and detector positions.
    """

    def __init__(self, angle_intvl, dparams, src_rad, det_rad,
                 spiral_pitch_factor, agrid=None, dgrid=None,
                 angle_offset=0, axis=None):
        """Initialize a new instance.

        Parameters
        ----------
        angle_intvl : `Interval` or 1-dim. `IntervalProd`
            The motion parameters
        dparams : `Rectangle` or 2-dim. `IntervalProd`
            The detector parameters
        src_rad : float
            Radius of the source circle, must be positive
        det_rad : float
            Radius of the detector circle, must be positive
        spiral_pitch_factor : float
            Dimensionless factor given by the ration of (Table feed per
            rotation (mm)) to (Total collimation width (mm)). The total
            collimation width is given by the slice thickness for a
            single-slice spiral acquisition and by the number of detector
            rows * slice thickness for multiple slice spiral acquisition.
            The collimation width is taken at isocenter.
        agrid : 1-dim. `TensorGrid`, optional
            A sampling grid for `angle_intvl`
        dgrid : 2-dim. `TensorGrid`, optional
            A sampling grid for `dparams`
        angle_offset : float
            Offset to the rotation angle in the azimuthal plane. Does not
            affect the offset in z-direction
        axis : int or 3-element array
            Defines the rotation axis via a 3-element vector or a single
            integer referring to a standard axis
        """

        super().__init__(angle_intvl, dparams, src_rad, det_rad, agrid,
                         dgrid, angle_offset, axis)
        det_height = (dparams.max() - dparams.min())[1]
        # TODO: correct for magnification ie det_height projected to isocente
        self._table_feed_per_rotation = spiral_pitch_factor * det_height

    @property
    def table_feed_per_rotation(self):
        """Table feed per 360 degree rotation. """
        return self._table_feed_per_rotation

    def det_refpoint(self, angle):
        """The detector reference point function.

        Parameters
        ----------
        angle : float
            The motion parameter. It must be contained in this
            geometry's motion parameter set.

        Returns
        -------
        point : ndarray, shape `(ndim,)`
            The reference point on the circle with radius `R` at
            the given rotation angle :math:`\phi`, defined as
            :math:`R(\cos\phi, \sin\phi, TableFeedPerRotation*\phi)`
        """
        if angle not in self.motion_params:
            raise ValueError('angle {} is not in the valid range {}.'
                             ''.format(angle, self.motion_params))

        return np.array([
            self.det_radius * cos(angle + self.angle_offset),
            self.det_radius * sin(angle + self.angle_offset),
            self.table_feed_per_rotation * angle / (2 * np.pi)])

    def src_position(self, angle):
        """The source position function.

        Parameters
        ----------
        angle : float
            The motion parameter. It must be contained in this
            geometry's motion parameter set.

        Returns
        -------
        point : ndarray, shape `(ndim,)`
            The source position on the spiral with radius `r` and pitch
            factor `P` at the given rotation angle :math:`\phi`, defined as
            :math:`-r(\cos\phi, \sin\phi, P*(total collimation width (mm))*\phi
            )`
        """
        if angle not in self.motion_params:
            raise ValueError('angle {} is not in the valid range {}.'
                             ''.format(angle, self.motion_params))

        return np.array([
            -self.src_radius * cos(angle + self.angle_offset),
            -self.src_radius * sin(angle + self.angle_offset),
            self.table_feed_per_rotation * angle / (2 * np.pi)])

    # TODO: backprojection weighting function?
