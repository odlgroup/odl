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
from abc import ABCMeta
from math import cos, sin  # , acos, atan2, sqrt
from future import standard_library
standard_library.install_aliases()
from builtins import super
from future.utils import with_metaclass

# External
import numpy as np

# Internal
from odl.set.domain import IntervalProd
from odl.discr.grid import TensorGrid
from odl.tomo.geometry.detector import LineDetector, Flat2dDetector
from odl.tomo.geometry.geometry import Geometry
from odl.tomo.util.trafos import euler_matrix

__all__ = ('Parallel2dGeometry', 'Parallel3dGeometry')


class ParallelGeometry(with_metaclass(ABCMeta, Geometry)):

    """Abstract parallel beam geometry, arbitrary dimension.

    The motion parameter is a (1d) rotation angle.
    """

    def __init__(self, angle_intvl, agrid=None, angle_offset=0):
        """Initialize a new instance.

        Parameters
        ----------
        angle_intvl : `Interval` or 1-dim. `IntervalProd`
            The motion parameters
        agrid : 1-dim. `TensorGrid`, optional
            A sampling grid for the `angle_intvl`
        angle_offset : float
            Offset to the rotation angle in the azimuthal plane. Does not
            imply an offset in z-direction.
        """
        if not isinstance(angle_intvl, IntervalProd) or angle_intvl.ndim != 1:
            raise TypeError('angle parameters {!r} are not an interval.'
                            ''.format(angle_intvl))
        angle_offset = float(angle_offset)

        if agrid is not None:
            if not isinstance(agrid, TensorGrid):
                raise TypeError('angle grid {!r} is not a `TensorGrid` '
                                'instance.'.format(agrid))
            if not angle_intvl.contains_set(agrid):
                raise ValueError('angular grid {} not contained in angle '
                                 'interval {}.'.format(agrid, angle_intvl))

        super().__init__()
        self._motion_params = angle_intvl
        self._motion_grid = agrid
        self._motion_params_offset = angle_offset

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
        imply an offset in z-direction. The actual angles then reside within
        `angle_offset` + `angle_intvl`. """
        return self._motion_params_offset

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
            The reference point, equal to the origin
        """
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
        `inf` according to the quadrant.

        Parameters
        ----------
        angle : float
            The motion parameter. Must be contained in this
            geometry's `motion_params`.
        dpar : float
            The detector parameter. Must be contained in this
            geometry's `det_params`.
        normalized : bool
            If `False` return the vector from the detector point
            parametrized by `par` to the source at `mpar`. If
            `True`, return the normalized version of that vector.

        Returns
        -------
        vec : ndarray, shape `(ndim,)`
            (Unit) vector pointing from the detector to the source
        """
        if angle not in self.motion_params:
            raise ValueError('angle {} not in the valid range {}.'
                             ''.format(angle, self.motion_params))
        # TODO: dpar not used, remove it?
        if dpar not in self.det_params:
            raise ValueError('detector parameter {} not in the valid range {}.'
                             ''.format(dpar, self.det_params))
        # TODO: is this right? assuming the detector to be at (1,0, 0) at zero
        # angle its postion at non-zero angle is the 1st column of the
        # rotation. matrix. rot_mat[0] however gives the 1t row.
        vec = -np.array(
            self.det_rotation(angle + self.angle_offset)[0]).squeeze()
        if not normalized:
            vec[vec != 0] *= np.inf
        return vec

    def src_position(self, angle):
        """The source position function.

        Parameters
        ----------
        angle : float
            The motion parameter. Must be contained in this
            geometry's `motion_params`.

        Returns
        -------
        pos : ndarray, shape `(2,)`
            The source position, an `ndim`-dimensional vector
        """
        return self.det_to_src(angle, 0, normalized=False)

    def __repr__(self):
        """`g.__repr__() <==> repr(g)`."""
        inner_fstr = '{!r}, {!r}'
        if self.has_motion_sampling:
            inner_fstr += ',\n agrid={agrid!r}'
        if self.has_det_sampling:
            inner_fstr += ',\n dgrid={dgrid!r}'
        inner_str = inner_fstr.format(self.motion_params, self.det_params,
                                      agrid=self.motion_grid,
                                      dgrid=self.det_grid)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`g.__str__() <==> str(g)`."""
        return self.__repr__()  # TODO: prettify


class Parallel2dGeometry(ParallelGeometry):

    """Parallel beam geometry in 2d.

    The motion parameter is the counter-clockwise rotation angle around
    the origin, and the detector is a line detector perpendicular to
    the ray direction.
    """

    def __init__(self, angle_intvl, dparams, agrid=None, dgrid=None,
                 angle_offset=0):
        """Initialize a new instance.

        Parameters
        ----------
        angle_intvl : `Interval` or 1-dim. `IntervalProd`
            The motion parameters
        dparams : `Interval` or 1-dim. `IntervalProd`
            The detector parameters
        agrid : 1-dim. `TensorGrid`, optional
            A sampling grid for the `angle_intvl`
        dgrid : 1-dim. `TensorGrid`, optional
            A sampling grid for the detector parameters
        angle_offset : float
            Offset to the rotation angle in the azimuthal plane. Does not
            imply an offset in z-direction.
        """
        super().__init__(angle_intvl, agrid, angle_offset)

        if not (isinstance(dparams, IntervalProd) and dparams.ndim == 1):
            raise TypeError('detector parameters {!r} are not an interval.'
                            ''.format(dparams))

        if dgrid is not None:
            if not isinstance(dgrid, TensorGrid):
                raise TypeError('detector grid {!r} is not a `TensorGrid` '
                                'instance.'.format(dgrid))
            if not dparams.contains_set(dgrid):
                raise ValueError('detector grid {} not contained in detector '
                                 'parameter interval {}.'
                                 ''.format(agrid, angle_intvl))

        self._detector = LineDetector(dparams, dgrid)

    @property
    def ndim(self):
        """Number of dimensions of this geometry."""
        return 2

    @property
    def detector(self):
        """Detector of this geometry."""
        return self._detector

    def det_rotation(self, angle):
        """The detector rotation function.

        Parameters
        ----------
        angle : float
            The motion parameter. It must be contained in this
            geometry's `motion_params`.

        Returns
        -------
        rot : matrix, shape `(2, 2)`
            The rotation matrix mapping the standard basis vectors in
            the fixed ("lab") coordinate system to the basis vectors of
            the local coordinate system of the detector reference point,
            expressed in the fixed system.
        """
        if angle not in self.motion_params:
            raise ValueError('angle {} not in the valid range {}.'
                             ''.format(angle, self.motion_params))
        return euler_matrix(angle + self.angle_offset)


class Parallel3dGeometry(ParallelGeometry):

    """Parallel beam geometry in 3d.

    The motion parameter is the rotation angle around the 3rd unit axis,
    and the detector is a flat 2d detector perpendicular to the ray
    direction.
    """

    def __init__(self, angle_intvl, dparams, agrid=None, dgrid=None,
                 angle_offset=0, axis=None):
        """Initialize a new instance.

        Parameters
        ----------
        angle_intvl : `Interval` or 1-dim. `IntervalProd`
            The motion parameters
        dparams : `Rectangle` or 2-dim. `IntervalProd`
            The detector parameters
        agrid : 1-dim. `TensorGrid`, optional
            A sampling grid for `angle_intvl`
        dgrid : 2-dim. `TensorGrid`, optional
            A sampling grid for `dparams`
        angle_offset : float
            Offset to the rotation angle in the azimuthal plane. Does not
            imply an offset in z-direction
        axis : int or 3-element array
            Defines the rotation axis via a 3-element vector or a single
            integer referring to a standard axis
        """
        super().__init__(angle_intvl, agrid)

        if not (isinstance(dparams, IntervalProd) and dparams.ndim == 2):
            raise TypeError('detector parameters {!r} are not a rectangle.'
                            ''.format(dparams))

        if dgrid is not None:
            if not isinstance(dgrid, TensorGrid):
                raise TypeError('detector grid {!r} is not a `TensorGrid` '
                                'instance.'.format(dgrid))
            if not dparams.contains_set(dgrid):
                raise ValueError('detector grid {} not contained in detector '
                                 'parameter rectangle {}.'
                                 ''.format(agrid, angle_intvl))

        self._axis = axis
        self._detector = Flat2dDetector(dparams, dgrid)

    @property
    def ndim(self):
        """Number of dimensions of this geometry."""
        return 3

    @property
    def detector(self):
        """Detector of this geometry."""
        return self._detector

    def det_rotation(self, angle):
        """The detector rotation function. Returns the matrix for rotating a
        vector in 3d counter-clockwise through an angle `angle` about the
        rotation axis given by the property `axis` according to the right
        hand rule.

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
            raise ValueError('`angle` {} not in the valid range {}.'
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
            tmp = axis
            axis = np.zeros(3)
            axis[tmp] = 1
        elif len(axis) == 3:
            axis = np.array(axis)
        else:
            raise ValueError('wrong format of `axis` intializer {}'.format(
                axis))

        return axis
