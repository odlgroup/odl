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
from odl.tomo.geometry.detector import LineDetector
from odl.tomo.geometry.geometry import Geometry
from odl.tomo.util.trafos import euler_matrix


__all__ = ('FanFlatGeometry',)


class FanBeamGeometry(with_metaclass(ABCMeta, Geometry)):

    """Abstract 2d fan beam geometry.

    The source moves on a circle with radius ``r``, and the detector
    reference point is opposite to the source on a circle with radius
    ``R``.

    The motion parameter is the (1d) rotation angle parametrizing source and
    detector positions.

    """

    def __init__(self, angle_intvl, src_radius, det_radius, agrid=None,
                 angle_offset=None):
        """Initialize a new instance.

        Parameters
        ----------
        angle_intvl : `Interval` or 1-dim. `IntervalProd`
            The motion parameters given in radians
        src_radius : positive `float`
            Radius of the source circle, must be positive
        det_radius : positive `float`
            Radius of the detector circle, must be positive
        agrid : 1-dim. `TensorGrid`, optional
            A sampling grid for `angle_intvl`. Default: `None`
        angle_offset : `float`, optional
            Offset to the rotation angle given in radians. Default: `None`
        """
        if not isinstance(angle_intvl, IntervalProd) or angle_intvl.ndim != 1:
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
        """Angles (= motion parameters) of this geometry given in radians."""
        return self._motion_params

    @property
    def angle_grid(self):
        """Angle (= motion parameter) sampling grid of this geometry."""
        return self._motion_grid

    @property
    def angle_offset(self):
        """Offset to the rotation angle in the azimuthal plane given in rad.

        The actual angles then reside within `angle_offset` + `angle_intvl`.
        """
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
        return 2

    def det_refpoint(self, angle):
        """The detector reference point function.

        Parameters
        ----------
        angle : `float`
            The motion parameter given in radians. It must be
            contained in this geometry's motion parameter set

        Returns
        -------
        point : `numpy.ndarray`, shape (`ndim`,)

            The reference point on the circle with radius :math:`R` at a given
            rotation angle :math:`\\phi`, defined as :math:`R(\\cos\\phi,
            \\sin\\phi)`
        """
        if angle not in self.motion_params:
            raise ValueError('angle {} not in the valid range {}.'
                             ''.format(angle, self.motion_params))
        return self.det_radius * np.array([np.cos(angle), np.sin(angle)])

    def det_rotation(self, angle):
        """The detector rotation function.

        Parameters
        ----------
        angle : `float`
            The motion parameter given in radians. It must be
            contained in this geometry's `motion_params`

        Returns
        -------
        rot : `numpy.matrix`, shape (2, 2)
            The rotation matrix mapping the standard basis vectors in
            the fixed ("lab") coordinate system to the basis vectors of
            the local coordinate system of the detector reference point,
            expressed in the fixed system
        """
        if angle not in self.motion_params:
            raise ValueError('angle {} not in the valid range {}.'
                             ''.format(angle, self.motion_params))
        return euler_matrix(angle)

    def src_position(self, angle):
        """The source position function.

        Parameters
        ----------
        angle : `float`
            The motion parameters given in radians. It must be
            contained in this geometry's motion parameter set

        Returns
        -------
        point : `numpy.ndarray`, shape (`ndim`,)
            The source position on the circle with radius :math:`r` at the
            given rotation angle :math:`\\phi`, defined as :math:`-r(
            \\cos\\phi, \\sin\\phi)`
        """
        if angle not in self.motion_params:
            raise ValueError('angle {} not in the valid range {}.'
                             ''.format(angle, self.motion_params))
        return -self.src_radius * np.array([np.cos(angle), np.sin(angle)])

    # TODO: backprojection weighting function?

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


class FanFlatGeometry(FanBeamGeometry):

    """Fan beam geometry in 2d with flat detector.

    The source moves on a circle with radius ``r``, and the detector
    reference point is opposite to the source on a circle with radius
    ``R`` and aligned tangential to the circle.

    The motion parameter is the (1d) rotation angle parametrizing
    source and detector positions.
    """

    def __init__(self, angle_intvl, dparams, src_radius, det_radius, agrid=None,
                 dgrid=None, angle_offset=0):
        """Initialize a new instance.

        Parameters
        ----------
        Parameters
        ----------
        angle_intvl : `Interval` or 1-dim. `IntervalProd`
            The motion parameters given in radians
        dparams : `Interval` or 1-dim. `IntervalProd`
            The detector parameters
        src_radius : `float`
            Radius of the source circle, must be positive
        det_radius : `float`
            Radius of the detector circle, must be positive
        agrid : 1-dim. `TensorGrid`, optional
            A sampling grid for `angle_intvl`. Default: `None`
        dgrid : 1-dim. `TensorGrid`, optional
            A sampling grid for the detector parameters. Default: `None`
        angle_offset : `float`, optional
            Offset to the rotation angle given in radians. Default: 0
        """
        super().__init__(angle_intvl, src_radius, det_radius, agrid, angle_offset)

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
                                 ''.format(dgrid, dparams))

        self._detector = LineDetector(dparams, dgrid)

    @property
    def detector(self):
        """Detector of this geometry."""
        return self._detector

    def det_to_src(self, angle, dpar, normalized=True):
        """Direction from a detector location to the source.

        Parameters
        ----------
        angle : `float`
            The motion parameters given in radians. It must be
            contained in this geometry's motion parameter set
        dpar : `float`
            The detector parameter. It must be contained in this
            geometry's detector parameter set
        normalized : bool
            If `False` return the vector from the detector point
            parametrized by `dpar` and `angle` to the source at
            `angle`. If `True`, return the normalized version of
            that vector

        Returns
        -------
        vec : `numpy.ndarray`, shape (`ndim`,)
            (Unit) vector pointing from the detector to the source
        """
        if angle not in self.motion_params:
            raise ValueError('angle {} not in the valid range {}.'
                             ''.format(angle, self.motion_params))
        if dpar not in self.det_params:
            raise ValueError('detector parameter {} not in the valid range {}.'
                             ''.format(dpar, self.det_params))

        # Angle of a detector point at `dpar` as seen from the source relative
        # to the line from the source to the detector reference point
        det_pt_angle = np.arctan2(dpar, self.src_radius + self.det_radius)
        angle += self.angle_offset + det_pt_angle
        dvec = -np.array([np.cos(angle),
                          np.sin(angle)])
        if not normalized:
            dvec *= self.src_radius + self.det_radius
        return dvec
