# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Geometry base and mixin classes."""

from __future__ import print_function, division, absolute_import
from builtins import object
import numpy as np

from odl.discr import RectPartition
from odl.tomo.geometry.detector import Detector
from odl.tomo.util import axis_rotation_matrix, is_inside_bounds


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
            If ``True``, methods computing vectors check input arguments.
            Checks are vectorized and add only a small overhead.
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
        """If ``True``, methods computing vectors check input arguments.

        For very large input arrays, these checks can introduce significant
        overhead, but the overhead is kept low by vectorization.
        """
        return self.__check_bounds

    def det_refpoint(self, mparam):
        """Detector reference point function.

        Parameters
        ----------
        mparam : `array-like` or sequence
            Motion parameter(s) at which to evaluate. If
            ``motion_params.ndim >= 2``, a sequence of that length must be
            provided.

        Returns
        -------
        point : `numpy.ndarray`
            Vector(s) pointing from the origin to the detector reference
            point at ``mparam``.
        """
        raise NotImplementedError('abstract method')

    def rotation_matrix(self, mparam):
        """Return the rotation matrix to the system state at ``mparam``.

        Parameters
        ----------
        mparam : `array-like` or sequence
            Motion parameter(s) at which to evaluate. If
            ``motion_params.ndim >= 2``, a sequence of that length must be
            provided.

        Returns
        -------
        rot : `numpy.ndarray`
            The rotation matrix (or matrices) mapping vectors at the
            initial state to the ones in the state defined by ``mparam``.
            The rotation is extrinsic, i.e., defined in the "world"
            coordinate system.
        """
        raise NotImplementedError('abstract method')

    def det_to_src(self, mparam, dparam, normalized=True):
        """Vector pointing from a detector location to the source.

        Parameters
        ----------
        mparam : `array-like` or sequence
            Motion parameter(s) at which to evaluate. If
            ``motion_params.ndim >= 2``, a sequence of that length must be
            provided.
        dparam : `array-like` or sequence
            Detector parameter(s) at which to evaluate. If
            ``det_params.ndim >= 2``, a sequence of that length must be
            provided.
        normalized : bool, optional
            If ``True``, normalize the resulting vector(s) to unit length.

        Returns
        -------
        vec : `numpy.ndarray`
            (Unit) vector(s) pointing from the detector to the source.
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
        mparam : `array-like` or sequence
            Motion parameter(s) at which to evaluate. If
            ``motion_params.ndim >= 2``, a sequence of that length must be
            provided.
        dparam : `array-like` or sequence
            Detector parameter(s) at which to evaluate. If
            ``det_params.ndim >= 2``, a sequence of that length must be
            provided.

        Returns
        -------
        pos : `numpy.ndarray`
            Vector(s) pointing from the origin to the detector point.
            The shape of the returned array is obtained from the
            (broadcast) shapes of ``mparam`` and ``dparam``, and
            broadcasting is supported within both parameters and between
            them. The precise definition of the shape is
            ``broadcast(bcast_mparam, bcast_dparam).shape + (ndim,)``,
            where ``bcast_mparam`` is

            - ``mparam`` if `motion_params` is 1D,
            - ``broadcast(*mparam)`` otherwise,

            and ``bcast_dparam`` defined analogously.

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
        >>> # Providing 3 pairs of parameters, resulting in 3 vectors
        >>> pts = geom.det_point_position([0, np.pi / 2, np.pi],
        ...                               [-1, 0, 1])
        >>> pts[0]  # Corresponds to angle = 0, dparam = -1
        array([-1.,  1.])
        >>> pts.shape
        (3, 2)
        >>> # Pairs of parameters arranged in arrays of same size
        >>> geom.det_point_position(np.zeros((4, 5)), np.zeros((4, 5))).shape
        (4, 5, 2)
        >>> # "Outer product" type evaluation using broadcasting
        >>> geom.det_point_position(np.zeros((4, 1)), np.zeros((1, 5))).shape
        (4, 5, 2)

        More complicated 3D geometry with 2 angle variables and 2 detector
        variables:

        >>> apart = odl.uniform_partition([0, 0], [np.pi, 2 * np.pi],
        ...                               (10, 20))
        >>> dpart = odl.uniform_partition([-1, -1], [1, 1], (20, 20))
        >>> geom = odl.tomo.Parallel3dEulerGeometry(apart, dpart)
        >>> # 2 values for each variable, resulting in 2 vectors
        >>> angles = ([0, np.pi / 2], [0, np.pi])
        >>> dparams = ([-1, 0], [-1, 0])
        >>> pts = geom.det_point_position(angles, dparams)
        >>> pts[0]  # Corresponds to angle = (0, 0), dparam = (-1, -1)
        array([-1.,  1., -1.])
        >>> pts.shape
        (2, 3)
        >>> # 4 x 5 parameters for both
        >>> angles = dparams = (np.zeros((4, 5)), np.zeros((4, 5)))
        >>> geom.det_point_position(angles, dparams).shape
        (4, 5, 3)
        >>> # Broadcasting angles to shape (4, 5, 1, 1)
        >>> angles = (np.zeros((4, 1, 1, 1)), np.zeros((1, 5, 1, 1)))
        >>> # Broadcasting dparams to shape (1, 1, 6, 7)
        >>> dparams = (np.zeros((1, 1, 6, 1)), np.zeros((1, 1, 1, 7)))
        >>> # Total broadcast parameter shape is (4, 5, 6, 7)
        >>> geom.det_point_position(angles, dparams).shape
        (4, 5, 6, 7, 3)
        """
        # Always call the downstream methods with vectorized arguments
        # to be able to reliably manipulate the final axes of the result
        if self.motion_params.ndim == 1:
            squeeze_mparam = (np.shape(mparam) == ())
            mparam = np.array(mparam, dtype=float, copy=False, ndmin=1)
            matrix = self.rotation_matrix(mparam)  # shape (m, ndim, ndim)
        else:
            squeeze_mparam = (np.broadcast(*mparam).shape == ())
            mparam = tuple(np.array(a, dtype=float, copy=False, ndmin=1)
                           for a in mparam)
            matrix = self.rotation_matrix(mparam)  # shape (m, ndim, ndim)

        if self.det_params.ndim == 1:
            squeeze_dparam = (np.shape(dparam) == ())
            dparam = np.array(dparam, dtype=float, copy=False, ndmin=1)
        else:
            squeeze_dparam = (np.broadcast(*dparam).shape == ())
            dparam = tuple(np.array(p, dtype=float, copy=False, ndmin=1)
                           for p in dparam)

        surf = self.detector.surface(dparam)  # shape (d, ndim)

        # Perform matrix-vector multiplication along the last axis of both
        # `matrix` and `surf` while "zipping" all axes that do not
        # participate in the matrix-vector product. In other words, the axes
        # are labelled
        # [0, 1, ..., r-1, r, r+1] for `matrix` and
        # [0, 1, ..., r-1, r+1] for `surf`, and the output axes are set to
        # [0, 1, ..., r-1, r]. This automatically supports broadcasting
        # along the axes 0, ..., r-1.
        matrix_axes = list(range(matrix.ndim))
        surf_axes = list(range(matrix.ndim - 2)) + [matrix_axes[-1]]
        out_axes = list(range(matrix.ndim - 1))
        det_part = np.einsum(matrix, matrix_axes, surf, surf_axes, out_axes)

        refpt = self.det_refpoint(mparam)
        det_pt_pos = refpt + det_part
        if squeeze_mparam and squeeze_dparam:
            det_pt_pos = det_pt_pos.squeeze()

        return det_pt_pos

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

    def src_position(self, angle):
        """Source position function.

        Parameters
        ----------
        angle : `array-like` or sequence
            Motion parameter(s) at which to evaluate. If
            ``motion_params.ndim >= 2``, a sequence of that length must be
            provided.

        Returns
        -------
        pos : `numpy.ndarray`
            Vector(s) pointing from the origin to the source.
        """
        raise NotImplementedError('abstract method')

    def det_to_src(self, angle, dparam, normalized=True):
        """Vector or direction from a detector location to the source.

        The unnormalized version of this vector is computed as follows::

            vec = src_position(angle) - det_point_position(angle, dparam)

        Parameters
        ----------
        angle : `array-like` or sequence
            One or several (Euler) angles in radians at which to
            evaluate. If ``motion_params.ndim >= 2``, a sequence of that
            length must be provided.
        dparam : `array-like` or sequence
            Detector parameter(s) at which to evaluate. If
            ``det_params.ndim >= 2``, a sequence of that length must be
            provided.

        Returns
        -------
        det_to_src : `numpy.ndarray`
            Vector(s) pointing from a detector point to the source (at
            infinity).
            The shape of the returned array is obtained from the
            (broadcast) shapes of ``angle`` and ``dparam``, and
            broadcasting is supported within both parameters and between
            them. The precise definition of the shape is
            ``broadcast(bcast_angle, bcast_dparam).shape + (ndim,)``,
            where ``bcast_angle`` is

            - ``angle`` if `motion_params` is 1D,
            - ``broadcast(*angle)`` otherwise,

            and ``bcast_dparam`` defined analogously.

        Examples
        --------
        The method works with single parameter values, in which case
        a single vector is returned:

        >>> apart = odl.uniform_partition(0, 2 * np.pi, 10)
        >>> dpart = odl.uniform_partition(-1, 1, 20)
        >>> geom = odl.tomo.FanBeamGeometry(apart, dpart, src_radius=2,
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
        >>> vec = geom.det_to_src(np.pi / 2, 0, normalized=False)
        >>> np.allclose(vec, [5, 0])
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
        >>> # Providing 3 pairs of parameters, resulting in 3 vectors
        >>> dirs = geom.det_to_src([0, np.pi / 2, np.pi], [0, -1, 1])
        >>> dirs[0]  # Corresponds to angle = 0, dparam = 0
        array([ 0., -1.])
        >>> dirs.shape
        (3, 2)
        >>> # Pairs of parameters arranged in arrays of same size
        >>> geom.det_to_src(np.zeros((4, 5)), np.zeros((4, 5))).shape
        (4, 5, 2)
        >>> # "Outer product" type evaluation using broadcasting
        >>> geom.det_to_src(np.zeros((4, 1)), np.zeros((1, 5))).shape
        (4, 5, 2)
        """
        # Always call the downstream methods with vectorized arguments
        # to be able to reliably manipulate the final axes of the result
        if self.motion_params.ndim == 1:
            squeeze_angle = (np.shape(angle) == ())
            angle = np.array(angle, dtype=float, copy=False, ndmin=1)
        else:
            squeeze_angle = (np.broadcast(*angle).shape == ())
            angle = tuple(np.array(a, dtype=float, copy=False, ndmin=1)
                          for a in angle)

        if self.det_params.ndim == 1:
            squeeze_dparam = (np.shape(dparam) == ())
            dparam = np.array(dparam, dtype=float, copy=False, ndmin=1)
        else:
            squeeze_dparam = (np.broadcast(*dparam).shape == ())
            dparam = tuple(np.array(p, dtype=float, copy=False, ndmin=1)
                           for p in dparam)

        det_to_src = (self.src_position(angle) -
                      self.det_point_position(angle, dparam))

        if normalized:
            det_to_src /= np.linalg.norm(det_to_src, axis=-1, keepdims=True)

        if squeeze_angle and squeeze_dparam:
            det_to_src = det_to_src.squeeze()

        return det_to_src


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
        rot : `numpy.ndarray`
            The rotation matrix (or matrices) mapping vectors at the
            initial state to the ones in the state defined by ``angle``.
            The rotation is extrinsic, i.e., defined in the "world"
            coordinate system.
            If ``angle`` is a single parameter, the returned array has
            shape ``(3, 3)``, otherwise ``angle.shape + (3, 3)``.
        """
        squeeze_out = (np.shape(angle) == ())
        angle = np.array(angle, dtype=float, copy=False, ndmin=1)
        if (self.check_bounds and
                not is_inside_bounds(angle, self.motion_params)):
            raise ValueError('`angle` {} not in the valid range {}'
                             ''.format(angle, self.motion_params))

        matrix = axis_rotation_matrix(self.axis, angle)
        if squeeze_out:
            matrix = matrix.squeeze()

        return matrix


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
