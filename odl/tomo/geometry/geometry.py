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
from builtins import object
from future.utils import with_metaclass

# External
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np

# Internal
from odl.set.domain import IntervalProd
from odl.discr.grid import RegularGrid, TensorGrid


__all__ = ('Geometry',)


class Geometry(with_metaclass(ABCMeta, object)):

    """Abstract geometry class.

    A geometry is described by

    * a detector,
    * a set of detector motion parameters,
    * a function mapping motion parameters to the location of a
      reference point (e.g. the center of the detector surface),
    * a rotation applied to the detector surface, depending on the
      motion parameters,
    * a mapping from the motion and surface parameters to the detector
      pixel direction to the source,
    * optionally a mapping from the motion parameters to the source
      position
    """

    @abstractproperty
    def ndim(self):
        """The number of dimensions of the geometry."""

    @abstractproperty
    def motion_params(self):
        """The motion parameters given as an `IntervalProd`."""

    @abstractproperty
    def detector(self):
        """The detector representation."""

    @abstractmethod
    def det_refpoint(self, mpar):
        """The detector reference point function.

        Parameters
        ----------
        mpar : element of motion parameters
            Motion parameter for which to calculate the detector
            reference point

        Returns
        -------
        point : numpy.ndarray, shape `(ndim,)`
            The reference point, an `ndim`-dimensional vector
        """

    # TODO: rename to rotation_matrix?
    @abstractmethod
    def det_rotation(self, mpar):
        """The detector rotation function for calculating the detector
        reference position.

        Parameters
        ----------
        mpar : element of motion parameters
            Motion parameter for which to calculate the detector
            reference rotation

        Returns
        -------
        rot : `numpy.matrix`, shape `(ndim, ndim)`
            The rotation matrix mapping the standard basis vectors in
            the fixed ("lab") coordinate system to the basis vectors of
            the local coordinate system of the detector reference point,
            expressed in the fixed system.
        """

    @abstractmethod
    def det_to_src(self, mpar, dpar, normalized=True):
        """Vector pointing from a detector location to the source.

        A function of the motion and detector parameters.

        Parameters
        ----------
        mpar : element of motion parameters
            Motion parameter at which to evaluate
        dpar : element of detector parameters
            Detector parameter at which to evaluate
        normalized : `bool`
            If `True`, return a normalized (unit) vector

        Returns
        -------
        vec : `numpy.ndarray`, shape `(ndim,)`
            (Unit) vector pointing from the detector to the source
        """

    def src_position(self, mpar):
        """The source position function.

        Parameters
        ----------
        mpar : element of motion parameters
            Motion parameter for which to calculate the source position

        Returns
        -------
        pos : `numpy.ndarray`, shape `(ndim,)`
            The source position, a `ndim`-dimensional vector
        """
        raise NotImplementedError

    def det_point_position(self, mpar, dpar):
        """The detector point position function.

        Parameters
        ----------
        mpar : element of motion parameters
            Motion parameter at which to evaluate
        dpar : element of detector parameters
            Detector parameter at which to evaluate

        Returns
        -------
        pos : `numpy.ndarray`, shape `(ndim,)`
            The source position, a `ndim`-dimensional vector
        """

        # TODO: check and write test
        return np.asarray(
            (self.det_refpoint(mpar + self.motion_params_offset) +
             self.det_rotation(mpar).dot(self.detector.surface(dpar))))

    @property
    def motion_params_offset(self):
        """Offset to the motion parameters `motion_params`."""
        return None

    @property
    def motion_grid(self):
        """A sampling grid for `motion_params`."""
        return None

    @property
    def has_motion_sampling(self):
        """Whether there is a `motion_grid` or not."""
        return self.motion_grid is not None

    @property
    def det_params(self):
        """The detector parameters."""
        return self.detector.params

    @property
    def det_grid(self):
        """A sampling grid for `det_params`."""
        return self.detector.param_grid

    @property
    def has_det_sampling(self):
        """Whether there is a `det_grid` or not."""
        return self.det_grid is not None

    @property
    def params(self):
        """Joined motion and detector parameters."""
        params = IntervalProd([], [])
        params = params.insert(self.motion_params, params.ndim)
        params = params.insert(self.det_params, params.ndim)
        return params

    @property
    def grid(self):
        """Joined sampling grid for motion and detector parameters."""
        if (isinstance(self.motion_grid, RegularGrid) and
                isinstance(self.det_grid, RegularGrid)):
            grid = self.motion_grid
        else:
            grid = TensorGrid(*self.motion_grid.coord_vectors)
        return grid.insert(self.det_grid, grid.ndim)
