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
from abc import ABCMeta
from future import standard_library
standard_library.install_aliases()
from builtins import super


# External
import numpy as np

# Internal
from odl.util.utility import with_metaclass
from odl.set.domain import IntervalProd
from odl.discr.grid import TensorGrid
from odl.tomo.geometry.detector import Flat1dDetector, Flat2dDetector
from odl.tomo.geometry.geometry import Geometry
from odl.tomo.util.trafos import euler_matrix

__all__ = ('ParallelGeometry', 'Parallel2dGeometry', 'Parallel3dGeometry')


# TODO: rotation and position functions (probably not working properly)

class ParallelGeometry(with_metaclass(ABCMeta, Geometry)):

    """Abstract parallel beam geometry, arbitrary dimension.

    The motion parameter is a (1d) rotation angle.
    """

    def __init__(self, angle_intvl, agrid=None, ndim=None):
        """Initialize a new instance.

        Parameters
        ----------
        angle_intvl : `Interval` or 1-dim. `IntervalProd`
            The motion parameters given in radians
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
        """Angles (= motion parameters) of this geometry given in radians."""
        return self._motion_params

    @property
    def angle_grid(self):
        """Angle (= motion parameter) sampling grid of this geometry."""
        return self._motion_grid

    def det_refpoint(self, angle):
        """The detector reference point function.

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
            If `False` return the vector from the detector point
            parametrized by ``dpar`` to the source at ``angle``. If
            `True`, return the normalized version of that vector.

        Returns
        -------
        vec : `numpy.ndarray`, shape (`ndim`,)
            (Unit) vector pointing from the detector to the source
        """
        angle = float(angle)
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
            self.rotation_matrix(angle)[0]).squeeze()
        if not normalized:
            vec[vec != 0] *= np.inf
        return vec

    def src_position(self, angle):
        """The source position function.

        Parameters
        ----------
        angle : `float`
            The motion parameters given in radians. Must be contained
            in this geometry's `motion_params`

        Returns
        -------
        pos : `numpy.ndarray`, shape (2,)
            The source position, an `ndim`-dimensional vector
        """
        angle = float(angle)
        return self.det_to_src(angle, 0, normalized=False)

    def __repr__(self):
        """Returns ``repr(self)``."""
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

    The motion parameter is the counter-clockwise rotation angle around the
    origin, and the detector is a line detector perpendicular to the ray
    direction.

    """

    def __init__(self, angle_intvl, dparams, agrid=None, dgrid=None):
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
        """
        super().__init__(angle_intvl, agrid, ndim=2)

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

        self._detector = Flat1dDetector(dparams, dgrid)

    @property
    def detector(self):
        """Detector of this geometry."""
        return self._detector

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


class Parallel3dGeometry(ParallelGeometry):

    """Parallel beam geometry in 3d.

    The motion parameter is the rotation angle around the 3rd unit axis,
    and the detector is a flat 2d detector perpendicular to the ray direction.

    """

    def __init__(self, angle_intvl, dparams, agrid=None, dgrid=None,
                 axis=(1, 0, 0)):
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
        axis : array-like, shape (3,)
            3-element vector defining the rotation axis
        """
        super().__init__(angle_intvl, agrid, ndim=3)

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
        if axis is not None:
            if len(axis) != 3:
                raise ValueError('length ({}) of axis {} is not 3'.format(
                    len(axis), axis))
        self._axis = np.array(axis) / np.linalg.norm(axis)
        self._detector = Flat2dDetector(dparams, [[0, 1, 0], [0, 0, 1]], dgrid)

    @property
    def detector(self):
        """Detector of this geometry."""
        return self._detector

    def rotation_matrix(self, angle):
        """The detector rotation function.

        Returns the matrix for rotating a vector in 3d counter-clockwise
        through an angle ``angle`` about the rotation axis given by the
        property `axis` according to the right hand rule.

        The matrix is computed according to `Rodrigues' rotation formula
        <https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula>`_.

        Parameters
        ----------
        angle : `float`
            The motion parameter given in radians. It must be contained in this
            geometry's `motion_params`

        Returns
        -------
        rot_mat : `numpy.ndarray`, shape (3, 3)
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

        cross_mat = np.array([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]])
        dy_mat = np.outer(axis, axis)
        id_mat = np.eye(3)
        cos_ang = np.cos(angle)
        sin_ang = np.sin(angle)

        return cos_ang * id_mat + (1. - cos_ang) * dy_mat + sin_ang * cross_mat

    @property
    def axis(self):
        """The rotation axis.

        Returns
        -------
        axis : `numpy.ndarray`, shape (3,)
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
