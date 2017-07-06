# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Geometry base and mixin classes."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from builtins import object

import numpy as np

from odl.discr import RectPartition
from odl.tomo.geometry.detector import Detector
from odl.tomo.util import axis_rotation_matrix


__all__ = ('Geometry', 'DivergentBeamGeometry', 'AxisOrientedGeometry')


class Geometry(object):

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
    * optionally a mapping from the motion parameters to the source position,
    * optionally a global translation of the geometry (shift of the origin)

    For details, check `the online docs
    <https://odlgroup.github.io/odl/guide/geometry_guide.html>`_.
    """

    def __init__(self, ndim, motion_part, detector, translation=None,
                 **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        ndim : positive int
            Number of dimensions of this geometry, i.e. dimensionality
            of the physical space in which this geometry is embedded.
        motion_part : `RectPartition`
            Partition for the set of "motion" parameters.
        detector : `Detector`
            The detector of this geometry.
        translation : `array-like`, optional
            Global translation of the geometry. This is added last in any
            method that computes an absolute vector, e.g., `det_refpoint`.
            Default: zero vector of length ``ndim``

        Other Parameters
        ----------------
        check_bounds : bool, optional
            If ``True``, methods perform sanity checks on provided input
            parameters.
            Default: ``True``
        """
        ndim, ndim_in = int(ndim), ndim
        if ndim != ndim_in or ndim <= 0:
            raise ValueError('`ndim` must be a positive integer, got {}'
                             ''.format(ndim_in))
        if not isinstance(motion_part, RectPartition):
            raise TypeError('`motion_part` must be a `RectPartition`, '
                            'instance, got {!r}'.format(motion_part))
        if not isinstance(detector, Detector):
            raise TypeError('`detector` must be a `Detector` instance, '
                            'got {!r}'.format(detector))

        self.__ndim = ndim
        self.__motion_partition = motion_part
        self.__detector = detector
        self.__check_bounds = bool(kwargs.pop('check_bounds', True))

        if translation is None:
            self.__translation = np.zeros(self.ndim)
        else:
            translation = np.asarray(translation, dtype=float)
            if translation.shape != (self.ndim,):
                raise ValueError('`translation` must have shape ({},), got {}'
                                 ''.format(self.ndim, translation.shape))
            self.__translation = translation

        # Cache geometry-related objects for backends that require computation
        self.__implementation_cache = {}

        # Make sure there are no leftover kwargs
        if kwargs:
            raise TypeError('got unexpected keyword arguments {}'
                            ''.format(kwargs))

    @property
    def ndim(self):
        """Number of dimensions of the geometry."""
        return self.__ndim

    @property
    def motion_partition(self):
        """Partition of the motion parameter set into subsets."""
        return self.__motion_partition

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
        return self.__detector

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

        A `RectPartition` with `det_partition` appended to `motion_partition`.
        """
        return self.motion_partition.append(self.det_partition)

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

    @property
    def translation(self):
        """Shift of the origin of this geometry."""
        return self.__translation

    @property
    def check_bounds(self):
        """Whether to check if method parameters are in the valid range."""
        return self.__check_bounds

    def det_refpoint(self, mparam):
        """Detector reference point function.

        Parameters
        ----------
        mparam : `array-like`
            Motion parameter(s) at which to evaluate. An array should
            stack parameters along axis 0.

        Returns
        -------
        point : `numpy.ndarray`, shape (ndim,) or (num_mparams, ndim)
            Vector(s) pointing from the origin to the detector reference
            point at ``mparam``.
            If ``mparam`` is a single parameter, a single vector is
            returned, otherwise a stack of vectors along axis 0.
        """
        raise NotImplementedError('abstract method')

    def rotation_matrix(self, mparam):
        """Return the rotation matrix to the system state at ``mparam``.

        Parameters
        ----------
        mparam : `array-like`
            Motion parameter(s) at which to evaluate. An array should
            stack parameters along axis 0.

        Returns
        -------
        rot : `numpy.ndarray`, shape (ndim, ndim) or (num_mparams, ndim, ndim)
            The rotation matrix (or matrices) mapping vectors at the
            initial state to the ones in the state defined by ``mparam``.
            The rotation is extrinsic, i.e., defined in the "world"
            coordinate system.
            If ``mparam`` is a single parameter, a single matrix is
            returned, otherwise a stack of matrices along axis 0.
        """
        raise NotImplementedError('abstract method')

    def det_to_src(self, mparam, dparam, normalized=True):
        """Vector pointing from a detector location to the source.

        Parameters
        ----------
        mparam : `motion_params` element or `array-like`
            Motion parameter(s) at which to evaluate. An array should
            stack parameters along axis 0.
        dparam : `det_params` element or `array-like`
            Detector parameter(s) at which to evaluate. An array should
            stack parameters along axis 0.
        normalized : bool, optional
            If ``True``, normalize the resulting vector(s) to unit length.

        Returns
        -------
        vec : `numpy.ndarray`
            (Unit) vector(s) pointing from the detector to the source.
            If both ``mparam`` and ``dparam`` are single parameters, a single
            vector is returned, otherwise a stack of vectors.
        """
        raise NotImplementedError('abstract method')

    def det_point_position(self, mparam, dparam):
        """Return the detector point at ``(mparam, dparam)``.

        The position is computed as follows::

            pos = refpoint(mparam) +
                  rotation_matrix(mparam).dot(detector.surface(dparam))

        In other words, the motion parameter ``mparam`` is used to move the
        detector reference point, and the detector parameter ``dparam``
        defines an intrinsic shift that is added to the reference point.

        Parameters
        ----------
        mparam : `motion_params` element or `array-like`
            Motion parameter(s) at which to evaluate. An array should
            stack parameters along axis 0.
        dparam : `det_params` element or `array-like`
            Detector parameter(s) at which to evaluate. An array should
            stack parameters along axis 0.

        Returns
        -------
        pos : `numpy.ndarray`
            Vector(s) pointing from the origin to the detector point.
            The shape of the returned array is as follows:

            - ``mparam`` and ``dparam`` single: ``(ndim,)``
            - ``mparam`` single, ``dparam`` stack: ``(num_dparams, ndim)``
            - ``mparam`` stack, ``dparam`` single: ``(num_mparams, ndim)``
            - ``mparam`` and ``dparam`` stacks:
              ``(num_mparams, num_dparams, ndim)``

        Examples
        --------
        The method works with single parameter values, in which case
        a single vector is returned:

        >>> apart = odl.uniform_partition(0, np.pi, 10)
        >>> dpart = odl.uniform_partition(-1, 1, 20)
        >>> geom = odl.tomo.Parallel2dGeometry(apart, dpart)
        >>> geom.det_point_position(0, 0)  # (0, 1) + 0 * (1, 0)
        array([ 0.,  1.])
        >>> geom.det_point_position(0, 1)  # (0, 1) + 1 * (1, 0)
        array([ 1.,  1.])
        >>> pt = geom.det_point_position(np.pi / 2, 0)  # (-1, 0) + 0 * (0, 1)
        >>> np.allclose(pt, [-1, 0])
        True
        >>> pt = geom.det_point_position(np.pi / 2, 1)  # (-1, 0) + 1 * (0, 1)
        >>> np.allclose(pt, [-1, 1])
        True

        Both variables support vectorized calls, i.e., stacks of
        parameters can be provided. The order of axes in the output (left
        of the ``ndim`` axis for the vector dimension) corresponds to the
        order of arguments:

        >>> geom.det_point_position(0, [-1, 0, 0.5, 1])
        array([[-1. ,  1. ],
               [ 0. ,  1. ],
               [ 0.5,  1. ],
               [ 1. ,  1. ]])
        >>> pts = geom.det_point_position([0, np.pi / 2, np.pi], 0)
        >>> np.allclose(pts, [[0, 1],
        ...                   [-1, 0],
        ...                   [0, -1]])
        True
        >>> pts = geom.det_point_position([0, np.pi / 2, np.pi],
        ...                               [-1, 0, 0.5, 1])
        >>> pts[0]  # Same as above with single angle, multiple dparams
        array([[-1. ,  1. ],
               [ 0. ,  1. ],
               [ 0.5,  1. ],
               [ 1. ,  1. ]])
        >>> pts.shape  # (num_mparams, num_dparams, ndim)
        (3, 4, 2)
        """
        if self.motion_params.ndim == 1:
            squeeze_mparam = np.isscalar(mparam)
            nd_mparam = 1
        else:
            squeeze_mparam = (np.shape(mparam) == (self.motion_params.ndim,))
            nd_mparam = 2

        if self.det_params.ndim == 1:
            squeeze_dparam = np.isscalar(dparam)
            nd_dparam = 1
        else:
            squeeze_dparam = (np.shape(dparam) == (self.det_params.ndim,))
            nd_dparam = 2

        # Always call the downstream methods with vectorized arguments
        # to be able to reliably manipulate the final axes of the result
        mparam = np.array(mparam, dtype=float, copy=False, ndmin=nd_mparam)
        dparam = np.array(dparam, dtype=float, copy=False, ndmin=nd_dparam)

        refpt = self.det_refpoint(mparam)  # shape (m, ndim)

        mat = self.rotation_matrix(mparam)  # shape (m, ndim, ndim)
        surf = self.detector.surface(dparam)  # shape (d, ndim)
        # Transpose to take dot along axis 1
        offset = mat.dot(surf.T)  # shape (m, ndim, d)
        offset = np.swapaxes(offset, 1, 2)  # shape (m, d, ndim)

        # Broadcast along axis 1 (detector)
        pos = refpt[:, None, :] + offset

        # Determine final shape by inserting depending on the `squeeze_*` flags
        final_shape = [self.ndim]
        if not squeeze_dparam:
            final_shape.insert(0, dparam.shape[0])
        if not squeeze_mparam:
            final_shape.insert(0, mparam.shape[0])

        return pos.reshape(final_shape)

    @property
    def implementation_cache(self):
        """Dictionary acting as a cache for this geometry.

        Intended for reuse of computations. Implementations that use this
        storage should take care of unique naming.

        Returns
        -------
        implementations : dict
        """
        return self.__implementation_cache


class DivergentBeamGeometry(Geometry):

    """Abstract divergent beam geometry class.

    A geometry characterized by the presence of a point-like ray source.

    In 2D such a geometry is usually called "fan beam geometry", while
    in 3D one speaks of "cone beam geometries".
    """

    def src_position(self, mparam):
        """Source position function.

        Parameters
        ----------
        mparam : `array-like`
            Motion parameter(s) at which to evaluate. An array should
            stack parameters along axis 0.

        Returns
        -------
        pos : `numpy.ndarray`, shape (ndim,) or (num_mparams, ndim)
            Vector(s) pointing from the origin to the source.
            If ``mparam`` is a single parameter, a single vector
            is returned, otherwise a stack of vectors along axis 0.
        """
        raise NotImplementedError('abstract method')

    def det_to_src(self, angle, dparam, normalized=True):
        """Vector or direction from a detector location to the source.

        The unnormalized version of this vector is computed as follows::

            vec = src_position(angle) - det_point_position(angle, dparam)

        Parameters
        ----------
        angle : `array-like`
            One or several (Euler) angles in radians at which to
            evaluate. An array should stack parameters along axis 0.
        dparam : `det_params` element or `array-like`
            Detector parameter(s) at which to evaluate. An array should
            stack parameters along axis 0.

        Returns
        -------
        det_to_src : `numpy.ndarray`
            Vector(s) pointing from a detector point to the source (at
            infinity).
            The shape of the returned array is as follows:

            - ``angle`` and ``dparam`` single: ``(ndim,)``
            - ``angle`` single, ``dparam`` stack: ``(num_dparams, ndim)``
            - ``angle`` stack, ``dparam`` single: ``(num_angles, ndim)``
            - ``angle`` and ``dparam`` stacks:
              ``(num_angle, num_dparams, ndim)``

        Examples
        --------
        The method works with single parameter values, in which case
        a single vector is returned:

        >>> apart = odl.uniform_partition(0, 2 * np.pi, 10)
        >>> dpart = odl.uniform_partition(-1, 1, 20)
        >>> geom = odl.tomo.FanFlatGeometry(apart, dpart, src_radius=2,
        ...                                 det_radius=3)
        >>> geom.det_to_src(0, 0)
        array([ 0., -1.])
        >>> geom.det_to_src(0, 0, normalized=False)
        array([ 0., -5.])
        >>> vec = geom.det_to_src(0, 1, normalized=False)
        >>> np.allclose(geom.det_point_position(0, 1) + vec,
        ...             geom.src_position(0))
        True
        >>> dir = geom.det_to_src(np.pi / 2, 0)
        >>> np.allclose(dir, [1, 0])
        True
        >>> dir = geom.det_to_src(np.pi / 2, 0, normalized=False)
        >>> np.allclose(dir, [5, 0])
        True

        Both variables support vectorized calls, i.e., stacks of
        parameters can be provided. The order of axes in the output (left
        of the ``ndim`` axis for the vector dimension) corresponds to the
        order of arguments:

        >>> dirs = geom.det_to_src(0, [-1, 0, 0.5, 1])
        >>> dirs[1]
        array([ 0., -1.])
        >>> dirs.shape  # (num_dparams, ndim)
        (4, 2)
        >>> dirs = geom.det_to_src([0, np.pi / 2, np.pi], 0)
        >>> np.allclose(dirs, [[0, -1],
        ...                    [1, 0],
        ...                    [0, 1]])
        True
        >>> dirs.shape  # (num_angles, ndim)
        (3, 2)
        >>> dirs = geom.det_to_src([0, np.pi / 2, np.pi], [-1, 0, 0.5, 1])
        >>> dirs.shape  # (num_angles, num_dparams, ndim)
        (3, 4, 2)
        """
        if self.motion_params.ndim == 1:
            squeeze_angle = np.isscalar(angle)
            nd_angle = 1
        else:
            squeeze_angle = (np.shape(angle) == (self.motion_params.ndim,))
            nd_angle = 2

        if self.det_params.ndim == 1:
            squeeze_dparam = np.isscalar(dparam)
            nd_dparam = 1
        else:
            squeeze_dparam = (np.shape(dparam) == (self.det_params.ndim,))
            nd_dparam = 2

        # Always call the downstream methods with vectorized arguments
        # to be able to reliably manipulate the final axes of the result
        angle = np.array(angle, dtype=float, copy=False, ndmin=nd_angle)
        dparam = np.array(dparam, dtype=float, copy=False, ndmin=nd_dparam)

        src_pos = self.src_position(angle)  # shape (a, ndim)
        det_pt_pos = self.det_point_position(angle, dparam)  # (a, d, ndim)
        # Broadcast along middle (detector) axis
        det_to_src = src_pos[:, None, :] - det_pt_pos

        if normalized:
            det_to_src /= np.linalg.norm(det_to_src, axis=-1, keepdims=True)

        # Determine final shape by inserting depending on the `squeeze_*` flags
        final_shape = [self.ndim]
        if not squeeze_dparam:
            final_shape.insert(0, dparam.shape[0])
        if not squeeze_angle:
            final_shape.insert(0, angle.shape[0])

        return det_to_src.reshape(final_shape)


class AxisOrientedGeometry(object):

    """Mixin class for 3d geometries oriented along an axis."""

    def __init__(self, axis):
        """Initialize a new instance.

        Parameters
        ----------
        axis : `array-like`, shape ``(3,)``
            Vector defining the fixed rotation axis of this geometry.
        """
        axis = np.asarray(axis, dtype=float)
        if axis.shape != (3,):
            raise ValueError('`axis.shape` must be (3,), got {}'
                             ''.format(axis.shape))

        if np.linalg.norm(axis) == 0:
            raise ValueError('`axis` cannot be zero')

        self.__axis = axis / np.linalg.norm(axis)

    @property
    def axis(self):
        """Normalized axis of rotation, a 3d vector."""
        return self.__axis

    def rotation_matrix(self, angle):
        """Return the rotation matrix to the system state at ``angle``.

        The matrix is computed according to
        `Rodrigues' rotation formula
        <https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula>`_.

        Parameters
        ----------
        angle : float or `array-like`
            Angle(s) in radians describing the counter-clockwise
            rotation of the system around `axis`.

        Returns
        -------
        rot : `numpy.ndarray`, shape (3, 3) or (num_angles, 3, 3)
            The rotation matrix (or matrices) mapping vectors at the
            initial state to the ones in the state defined by ``angle``.
            The rotation is extrinsic, i.e., defined in the "world"
            coordinate system.
            If ``angle`` is a single parameter, a single matrix is
            returned, otherwise a stack of matrices along axis 0.
        """
        squeeze_out = np.isscalar(angle)
        angle = np.array(angle, dtype=float, copy=False, ndmin=1)
        if self.check_bounds and not self.motion_params.contains_all(angle):
            raise ValueError('`angle` {} not in the valid range {}'
                             ''.format(angle, self.motion_params))

        matrix = axis_rotation_matrix(self.axis, angle)
        if squeeze_out:
            matrix = matrix.squeeze()

        return matrix


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
