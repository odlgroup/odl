# Copyright 2014-2016 The ODL development group
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

"""Geometry base and mixin classes."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import object

from abc import ABCMeta, abstractmethod
import numpy as np

from odl.discr import RectPartition
from odl.tomo.geometry.detector import Detector
from odl.util.utility import with_metaclass


__all__ = ('Geometry', 'DivergentBeamGeometry', 'AxisOrientedGeometry')


class Geometry(with_metaclass(ABCMeta, object)):

    """Abstract geometry class.

    A geometry is described by

    * a detector,
    * a set of detector motion parameters,
    * a function mapping motion parameters to the location of a
      reference point (e.g. the center of the detector surface),
    * a rotation applied to the detector surface, depending on the motion
      parameters,
    * a mapping from the motion and surface parameters to the detector pixel
      direction to the source,
    * optionally a mapping from the motion parameters to the source position
    """

    def __init__(self, ndim, motion_part, detector):
        """Initialize a new instance.

        Parameters
        ----------
        ndim : positive int
            Number of dimensions of this geometry, i.e. dimensionality
            of the physical space in which this geometry is embedded
        motion_part : `RectPartition`
           Partition for the set of "motion" parameters
        detector : `Detector`
           The detector of this geometry
        """
        if int(ndim) <= 0:
            raise ValueError('number of dimensions {} is not positive'
                             ''.format(ndim))
        if not isinstance(motion_part, RectPartition):
            raise TypeError('`motion_part` {!r} not a RectPartition instance'
                            ''.format(motion_part))

        if not isinstance(detector, Detector):
            raise TypeError('`detector` {!r} not a Detector instance'
                            ''.format(detector))

        self._ndim = int(ndim)
        self._motion_part = motion_part
        self._detector = detector
        self._implementation_cache = {}

    @property
    def ndim(self):
        """Number of dimensions of the geometry."""
        return self._ndim

    @property
    def motion_partition(self):
        """Partition of the motion parameter set into subsets."""
        return self._motion_part

    @property
    def motion_params(self):
        """Continuous motion parameter range, an `IntervalProd`."""
        return self.motion_partition.set

    @property
    def motion_grid(self):
        """Sampling grid of `motion_params`."""
        return self.motion_partition.grid

    @property
    def detector(self):
        """Detector representation of this geometry."""
        return self._detector

    @property
    def det_partition(self):
        """Partition of the detector parameter set into subsets."""
        return self.detector.partition

    @property
    def det_params(self):
        """Continuous detector parameter range, an `IntervalProd`."""
        return self.detector.params

    @property
    def det_grid(self):
        """Sampling grid of `det_params`."""
        return self.detector.grid

    @property
    def partition(self):
        """Joined parameter set partition for motion and detector.

        Returns a `RectPartition` with the detector partition inserted
        after the motion partition.
        """
        # TODO: change when RectPartition.append is implemented
        return self.det_partition.insert(0, self.motion_partition)

    @property
    def params(self):
        """Joined parameter set for motion and detector.

        By convention, the motion parameters come before the detector
        parameters.
        """
        return self.partition.set

    @property
    def grid(self):
        """Joined sampling grid for motion and detector.

        By convention, the motion grid comes before the detector grid.
        """
        return self.partition.grid

    @abstractmethod
    def det_refpoint(self, mpar):
        """Detector reference point function.

        Parameters
        ----------
        mpar : `motion_params` element
            Motion parameter for which to calculate the detector
            reference point

        Returns
        -------
        point : `numpy.ndarray`, shape (`ndim`,)
            The reference point, an `ndim`-dimensional vector
        """

    @abstractmethod
    def rotation_matrix(self, mpar):
        """Detector rotation function for calculating the detector
        reference position.

        Parameters
        ----------
        mpar : `motion_params` element
            Motion parameter for which to calculate the detector
            reference rotation

        Returns
        -------
        rot : `numpy.ndarray`, shape (`ndim`, `ndim`)
            The rotation matrix mapping the standard basis vectors in
            the fixed ("lab") coordinate system to the basis vectors of
            the local coordinate system of the detector reference point,
            expressed in the fixed system.
        """

    def det_to_src(self, mpar, dpar, normalized=True):
        """Vector pointing from a detector location to the source.

        A function of the motion and detector parameters.

        Parameters
        ----------
        mpar : `motion_params` element
            Motion parameter at which to evaluate
        dpar : `det_params` element
            Detector parameter at which to evaluate
        normalized : bool, optional
            If ``True``, return a normalized (unit) vector.

        Returns
        -------
        vec : `numpy.ndarray`, shape (`ndim`,)
            (Unit) vector pointing from the detector to the source
        """
        raise NotImplementedError

    def det_point_position(self, mpar, dpar):
        """Detector point position function.

        Parameters
        ----------
        mpar : `motion_params` element
            Motion parameter at which to evaluate
        dpar : `det_params` element
            Detector parameter at which to evaluate

        Returns
        -------
        pos : `numpy.ndarray` (shape (`ndim`,))
            Source position, an `ndim`-dimensional vector
        """
        # TODO: check and write test
        return np.asarray(
            (self.det_refpoint(mpar) +
             self.rotation_matrix(mpar).dot(self.detector.surface(dpar))))

    @property
    def implementation_cache(self):
        """Dictionary acting as a cache for this geometry.

        Intended for reuse of computations. Implementations that use this
        storage should take care of unique naming.

        Returns
        -------
        implementations : dict
        """
        return self._implementation_cache


class DivergentBeamGeometry(Geometry):

    """Abstract divergent beam geometry class.

    A divergent beam geometry is characterized by the presence of a
    point source.

    Special cases include fan beam in 2d and cone beam in 3d.
    """

    @abstractmethod
    def src_position(self, mpar):
        """Source position function.

        Parameters
        ----------
        mpar : `motion_params` element
            Motion parameter for which to calculate the source position

        Returns
        -------
        pos : `numpy.ndarray` (shape (`ndim`,))
            Source position, an `ndim`-dimensional vector
        """

    def det_to_src(self, mpar, dpar, normalized=True):
        """Vector pointing from a detector location to the source.

        A function of the motion and detector parameters.

        The default implementation uses the `det_point_position` and
        `src_position` functions. Implementations can override this, for
        example if no source position is given.

        Parameters
        ----------
        mpar : `motion_params` element
            Motion parameter at which to evaluate
        dpar : `det_params` element
            Detector parameter at which to evaluate
        normalized : bool, optional
            If ``True``, return a normalized (unit) vector.

        Returns
        -------
        vec : `numpy.ndarray`, shape (`ndim`,)
            (Unit) vector pointing from the detector to the source
        """
        if mpar not in self.motion_params:
            raise ValueError('`mpar` {} is not in the valid range {}'
                             ''.format(mpar, self.motion_params))
        if dpar not in self.det_params:
            raise ValueError('`dpar` {} is not in the valid range {}'
                             ''.format(dpar, self.det_params))

        vec = self.src_position(mpar) - self.det_point_position(mpar, dpar)

        if normalized:
            # axis = -1 allows this to be vectorized
            vec /= np.linalg.norm(vec, axis=-1)

        return vec


class AxisOrientedGeometry(object):

    """Mixin class for 3d geometries oriented according to an axis."""

    def __init__(self, axis):
        """Initialize a new instance.

        Parameters
        ----------
        axis : `array-like` (shape ``(3,)``)
            Vector defining the fixed rotation axis after normalization
        """
        if np.linalg.norm(axis) <= 1e-10:
            raise ValueError('`axis` {} too close to zero'.format(axis))

        self._axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        if self.axis.shape != (3,):
            raise ValueError('`axis` has shape {}, expected (3,)'
                             ''.format(self.axis.shape))

    @property
    def axis(self):
        """Normalized axis of rotation, a 3-element vector."""
        return self._axis

    def rotation_matrix(self, angle):
        """Detector rotation function.

        Returns the matrix for rotating a vector in 3d by an angle ``angle``
        about the rotation axis given by the property `axis` according to
        the right hand rule.

        The matrix is computed according to
        `Rodrigues' rotation formula
        <https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula>`_.

        Parameters
        ----------
        angle : float
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
            raise ValueError('`angle` {} is not in the valid range {}'
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


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
