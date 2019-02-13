# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Parallel beam geometries in 2 or 3 dimensions."""

from __future__ import print_function, division, absolute_import
import numpy as np

from odl.discr import uniform_partition
from odl.tomo.geometry.detector import Flat1dDetector, Flat2dDetector
from odl.tomo.geometry.geometry import Geometry, AxisOrientedGeometry
from odl.tomo.util import euler_matrix, transform_system, is_inside_bounds
from odl.util import signature_string, indent, array_str


__all__ = ('ParallelBeamGeometry',
           'Parallel2dGeometry',
           'Parallel3dEulerGeometry', 'Parallel3dAxisGeometry',
           'parallel_beam_geometry')


class ParallelBeamGeometry(Geometry):

    """Abstract parallel beam geometry in 2 or 3 dimensions.

    Parallel geometries are characterized by a virtual source at
    infinity, such that a unit vector from a detector point towards
    the source (`det_to_src`) is independent of the location on the
    detector.

    For details, check `the online docs
    <https://odlgroup.github.io/odl/guide/geometry_guide.html>`_.
    """

    def __init__(self, ndim, apart, detector, det_pos_init, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        ndim : {2, 3}
            Number of dimensions of this geometry, i.e. dimensionality
            of the physical space in which this geometry is embedded.
        apart : `RectPartition`
            Partition of the angle set.
        detector : `Detector`
            The detector to use in this geometry.
        det_pos_init : `array-like`
            Initial position of the detector reference point.
        kwargs :
            Further parameters passed on to `Geometry`.
        """
        super(ParallelBeamGeometry, self).__init__(
            ndim, apart, detector, **kwargs)

        if self.ndim not in (2, 3):
            raise ValueError('`ndim` must be 2 or 3, got {}'.format(ndim))

        self.__det_pos_init = np.asarray(det_pos_init, dtype='float64')
        if self.det_pos_init.shape != (self.ndim,):
            raise ValueError('`det_pos_init` must have shape ({},), got {}'
                             ''.format(self.ndim, self.det_pos_init.shape))

    @property
    def det_pos_init(self):
        """Initial position of the detector reference point."""
        return self.__det_pos_init

    @property
    def angles(self):
        """All angles of this geometry as an array.

        If ``motion_params.ndim == 1``, the array has shape ``(N,)``,
        where ``N`` is the number of angles.
        Otherwise, the array shape is ``(ndim, N)``, where ``N`` is the
        total number of angles, and ``ndim`` is ``motion_partitioin.ndim``.

        The order of axes is chosen such that ``geometry.angles`` can be
        used directly as input to any of the other methods of the
        geometry.
        """
        if self.motion_partition.ndim == 1:
            return self.motion_grid.coord_vectors[0]
        else:
            return self.motion_grid.points().T

    def det_refpoint(self, angle):
        """Return the position(s) of the detector ref. point at ``angle``.

        The reference point is given by a rotation of the initial
        position by ``angle``.

        For an angle (or a vector of angles) ``phi``, the detector position
        is given by ::

            det_ref(phi) = translation +
                           rotation_matrix(phi) * (det_pos_init - translation)

        where ``det_pos_init`` is the detector reference point at initial
        state.

        This default implementation assumes in the case of 2 or 3 motion
        parameters that they are to be interpreted as Euler angles.
        Subclasses with a deviating intended interpretation should override
        this method.

        Parameters
        ----------
        angle : `array-like` or sequence
            One or several (Euler) angles in radians at which to
            evaluate. If ``motion_params.ndim >= 2``, a sequence of that
            length must be provided.

        Returns
        -------
        refpt : `numpy.ndarray`
            Vector(s) pointing from the origin to the detector reference
            point at ``angle``.
            If ``angle`` is a single parameter, the returned array has
            shape ``(ndim,)``, otherwise

            - ``angle.shape + (ndim,)`` if `motion_params` is 1D,
            - ``broadcast(*angle).shape + (ndim,)`` if `motion_params` is 2D
              or 3D (Euler angles).

        See Also
        --------
        rotation_matrix

        Examples
        --------
        For 2d and default arguments, the detector starts at ``e_y`` and
        rotates to ``-e_x`` at 90 degrees:

        >>> apart = odl.uniform_partition(0, np.pi, 10)
        >>> dpart = odl.uniform_partition(-1, 1, 20)
        >>> geom = Parallel2dGeometry(apart, dpart)
        >>> geom.det_refpoint(0)
        array([ 0.,  1.])
        >>> np.allclose(geom.det_refpoint(np.pi / 2), [-1, 0])
        True

        The method is vectorized, i.e., it can be called with multiple
        angles at once (or n-dimensional arrays of parameters):

        >>> points = geom.det_refpoint([0, np.pi])
        >>> np.allclose(points[0], [0, 1])
        True
        >>> np.allclose(points[1], [0, -1])
        True
        >>> geom.det_refpoint(np.zeros((4, 5))).shape
        (4, 5, 2)

        In 3d with single rotation axis ``e_z``, we have the same situation,
        except that the vectors have a third component equal to 0:

        >>> apart = odl.uniform_partition(0, np.pi, 10)
        >>> dpart = odl.uniform_partition([-1, -1], [1, 1], (20, 20))
        >>> geom = Parallel3dAxisGeometry(apart, dpart)
        >>> geom.det_refpoint(0)
        array([ 0.,  1.,  0.])
        >>> np.allclose(geom.det_refpoint(np.pi / 2), [-1, 0, 0])
        True
        """
        if self.motion_params.ndim == 1:
            squeeze_out = (np.shape(angle) == ())
            angle = np.array(angle, dtype=float, copy=False, ndmin=1)
            rot_matrix = self.rotation_matrix(angle)
            extra_dims = angle.ndim
        elif self.motion_params.ndim in (2, 3):
            squeeze_out = (np.broadcast(*angle).shape == ())
            angle = tuple(np.array(a, dtype=float, copy=False, ndmin=1)
                          for a in angle)
            rot_matrix = self.rotation_matrix(angle)
            extra_dims = len(np.broadcast(*angle).shape)
        else:
            raise NotImplementedError(
                'no default implementation available for `det_refpoint` '
                'with `motion_params.ndim == {}`'
                ''.format(self.motion_params.ndim))

        rot_part = rot_matrix.dot(self.det_pos_init - self.translation)

        # Broadcast along extra dimensions
        pt_slc = (None,) * extra_dims + (slice(None),)
        refpoint = self.translation[pt_slc] + rot_part
        if squeeze_out:
            refpoint = refpoint.squeeze()

        return refpoint

    def det_to_src(self, angle, dparam):
        """Direction from a detector location to the source.

        The direction vector is computed as follows::

            dir = rotation_matrix(angle).dot(detector.surface_normal(dparam))

        Note that for flat detectors, ``surface_normal`` does not depend
        on the parameter ``dparam``, hence this function is constant in
        that variable.

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

        >>> apart = odl.uniform_partition(0, np.pi, 10)
        >>> dpart = odl.uniform_partition(-1, 1, 20)
        >>> geom = odl.tomo.Parallel2dGeometry(apart, dpart)
        >>> geom.det_to_src(0, 0)
        array([ 0., -1.])
        >>> geom.det_to_src(0, 1)
        array([ 0., -1.])
        >>> dir = geom.det_to_src(np.pi / 2, 0)
        >>> np.allclose(dir, [1, 0])
        True
        >>> dir = geom.det_to_src(np.pi / 2, 1)
        >>> np.allclose(dir, [1, 0])
        True

        Both variables support vectorized calls, i.e., stacks of
        parameters can be provided. The order of axes in the output (left
        of the ``ndim`` axis for the vector dimension) corresponds to the
        order of arguments:

        >>> dirs = geom.det_to_src(0, [-1, 0, 0.5, 1])
        >>> dirs
        array([[ 0., -1.],
               [ 0., -1.],
               [ 0., -1.],
               [ 0., -1.]])
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
        >>> dirs = geom.det_to_src([0, np.pi / 2, np.pi], [-1, 0, 1])
        >>> dirs[0]  # Corresponds to angle = 0, dparam = -1
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
            matrix = self.rotation_matrix(angle)  # shape (m, ndim, ndim)
        else:
            squeeze_angle = (np.broadcast(*angle).shape == ())
            angle = tuple(np.array(a, dtype=float, copy=False, ndmin=1)
                          for a in angle)
            matrix = self.rotation_matrix(angle)  # shape (m, ndim, ndim)

        if self.det_params.ndim == 1:
            squeeze_dparam = (np.shape(dparam) == ())
            dparam = np.array(dparam, dtype=float, copy=False, ndmin=1)
        else:
            squeeze_dparam = (np.broadcast(*dparam).shape == ())
            dparam = tuple(np.array(p, dtype=float, copy=False, ndmin=1)
                           for p in dparam)

        normal = self.detector.surface_normal(dparam)  # shape (d, ndim)

        # Perform matrix-vector multiplication along the last axis of both
        # `matrix` and `normal` while "zipping" all axes that do not
        # participate in the matrix-vector product. In other words, the axes
        # are labelled
        # [0, 1, ..., r-1, r, r+1] for `matrix` and
        # [0, 1, ..., r-1, r+1] for `normal`, and the output axes are set to
        # [0, 1, ..., r-1, r]. This automatically supports broadcasting
        # along the axes 0, ..., r-1.
        matrix_axes = list(range(matrix.ndim))
        normal_axes = list(range(matrix.ndim - 2)) + [matrix_axes[-1]]
        out_axes = list(range(matrix.ndim - 1))
        det_to_src = np.einsum(matrix, matrix_axes, normal, normal_axes,
                               out_axes)
        if squeeze_angle and squeeze_dparam:
            det_to_src = det_to_src.squeeze()

        return det_to_src


class Parallel2dGeometry(ParallelBeamGeometry):

    """Parallel beam geometry in 2d.

    The motion parameter is the counter-clockwise rotation angle around
    the origin, and the detector is a line detector.

    In the standard configuration, the detector is perpendicular to the
    ray direction, its reference point is initially at ``(0, 1)``, and
    the initial detector axis is ``(1, 0)``.

    For details, check `the online docs
    <https://odlgroup.github.io/odl/guide/geometry_guide.html>`_.
    """

    _default_config = dict(det_pos_init=(0, 1), det_axis_init=(1, 0))

    def __init__(self, apart, dpart, det_pos_init=(0, 1), **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        apart : 1-dim. `RectPartition`
            Partition of the angle interval.
        dpart : 1-dim. `RectPartition`
            Partition of the detector parameter interval.
        det_pos_init : `array-like`, shape ``(2,)``, optional
            Initial position of the detector reference point.

        Other Parameters
        ----------------
        det_axis_init : `array-like` (shape ``(2,)``), optional
            Initial axis defining the detector orientation. The default
            depends on ``det_pos_init``, see Notes.
        translation : `array-like`, shape ``(2,)``, optional
            Global translation of the geometry. This is added last in any
            method that computes an absolute vector, e.g., `det_refpoint`,
            and also shifts the center of rotation.
            Default: ``(0, 0)``
        check_bounds : bool, optional
            If ``True``, methods computing vectors check input arguments.
            Checks are vectorized and add only a small overhead.
            Default: ``True``

        Notes
        -----
        In the default configuration, the initial detector reference point
        is ``(0, 1)``, and the initial detector axis is ``(1, 0)``. If a
        different ``det_pos_init`` is chosen, the new default axis is
        given as a rotation of the original one by a matrix that transforms
        ``(0, 1)`` to the new (normalized) ``det_pos_init``. This matrix
        is calculated with the `rotation_matrix_from_to` function.
        Expressed in code, we have ::

            init_rot = rotation_matrix_from_to((0, 1), det_pos_init)
            det_axis_init = init_rot.dot((1, 0))

        If ``det_pos_init == (0, 0)``, no rotation is performed.

        Examples
        --------
        Initialization with default parameters:

        >>> apart = odl.uniform_partition(0, np.pi, 10)
        >>> dpart = odl.uniform_partition(-1, 1, 20)
        >>> geom = Parallel2dGeometry(apart, dpart)
        >>> np.allclose(geom.det_refpoint(0), (0, 1))
        True
        >>> np.allclose(geom.det_point_position(0, 1), (1, 1))
        True

        Checking the default orientation:

        >>> e_x, e_y = np.eye(2)  # standard unit vectors
        >>> np.allclose(geom.det_pos_init, e_y)
        True
        >>> np.allclose(geom.det_axis_init, e_x)
        True

        Specifying an initial detector position by default rotates the
        standard configuration to this position:

        >>> geom = Parallel2dGeometry(apart, dpart, det_pos_init=(-1, 0))
        >>> np.allclose(geom.det_pos_init, -e_x)
        True
        >>> np.allclose(geom.det_axis_init, e_y)
        True
        >>> geom = Parallel2dGeometry(apart, dpart, det_pos_init=(0, -1))
        >>> np.allclose(geom.det_pos_init, -e_y)
        True
        >>> np.allclose(geom.det_axis_init, -e_x)
        True

        The initial detector axis can also be set explicitly:

        >>> geom = Parallel2dGeometry(
        ...     apart, dpart, det_pos_init=(0, -1), det_axis_init=(1, 0))
        >>> np.allclose(geom.det_pos_init, -e_y)
        True
        >>> np.allclose(geom.det_axis_init, e_x)
        True
        """
        default_det_pos_init = self._default_config['det_pos_init']
        default_det_axis_init = self._default_config['det_axis_init']

        # Handle the initial coordinate system. We need to assign `None` to
        # the vectors first in order to signalize to the `transform_system`
        # utility that they should be transformed from default since they
        # were not explicitly given.
        det_axis_init = kwargs.pop('det_axis_init', None)

        # Store some stuff for repr
        if det_pos_init is not None:
            self._det_pos_init_arg = np.asarray(det_pos_init, dtype=float)
        else:
            self._det_pos_init_arg = None

        if det_axis_init is not None:
            self._det_axis_init_arg = np.asarray(det_axis_init, dtype=float)
        else:
            self._det_axis_init_arg = None

        # Compute the transformed system and the transition matrix. We
        # transform only those vectors that were not explicitly given.
        vecs_to_transform = []
        if det_axis_init is None:
            vecs_to_transform.append(default_det_axis_init)

        transformed_vecs = transform_system(
            det_pos_init, default_det_pos_init, vecs_to_transform)
        transformed_vecs = list(transformed_vecs)

        det_pos_init = transformed_vecs.pop(0)
        if det_axis_init is None:
            det_axis_init = transformed_vecs.pop(0)
        assert transformed_vecs == []

        # Translate the absolute vectors by the given translation
        translation = np.asarray(kwargs.pop('translation', (0, 0)),
                                 dtype=float)
        det_pos_init += translation

        # Initialize stuff. Normalization of the detector axis happens in
        # the detector class. `check_bounds` is needed for both detector
        # and geometry.
        check_bounds = kwargs.get('check_bounds', True)
        detector = Flat1dDetector(dpart, axis=det_axis_init,
                                  check_bounds=check_bounds)
        super(Parallel2dGeometry, self).__init__(
            ndim=2, apart=apart, detector=detector,
            det_pos_init=det_pos_init, translation=translation,
            **kwargs)

        if self.motion_partition.ndim != 1:
            raise ValueError('`apart` dimension {}, expected 1'
                             ''.format(self.motion_partition.ndim))

    @classmethod
    def frommatrix(cls, apart, dpart, init_matrix, **kwargs):
        """Create an instance of `Parallel2dGeometry` using a matrix.

        This alternative constructor uses a matrix to rotate and
        translate the default configuration. It is most useful when
        the transformation to be applied is already given as a matrix.

        Parameters
        ----------
        apart : 1-dim. `RectPartition`
            Partition of the angle interval.
        dpart : 1-dim. `RectPartition`
            Partition of the detector parameter interval.
        init_matrix : `array_like`, shape ``(2, 2)`` or ``(2, 3)``, optional
            Transformation matrix whose left ``(2, 2)`` block is multiplied
            with the default ``det_pos_init`` and ``det_axis_init`` to
            determine the new vectors. If present, the third column acts
            as a translation after the initial transformation.
            The resulting ``det_axis_init`` will be normalized.
        kwargs :
            Further keyword arguments passed to the class constructor.

        Returns
        -------
        geometry : `Parallel2dGeometry`

        Examples
        --------
        Mirror the second unit vector, creating a left-handed system:

        >>> apart = odl.uniform_partition(0, np.pi, 10)
        >>> dpart = odl.uniform_partition(-1, 1, 20)
        >>> matrix = np.array([[1, 0],
        ...                    [0, -1]])
        >>> geom = Parallel2dGeometry.frommatrix(apart, dpart, matrix)
        >>> e_x, e_y = np.eye(2)  # standard unit vectors
        >>> np.allclose(geom.det_pos_init, -e_y)
        True
        >>> np.allclose(geom.det_axis_init, e_x)
        True
        >>> np.allclose(geom.translation, (0, 0))
        True

        Adding a translation with a third matrix column:

        >>> matrix = np.array([[1, 0, 1],
        ...                    [0, -1, 1]])
        >>> geom = Parallel2dGeometry.frommatrix(apart, dpart, matrix)
        >>> np.allclose(geom.translation, (1, 1))
        True
        >>> np.allclose(geom.det_pos_init, -e_y + (1, 1))
        True
        """
        # Get transformation and translation parts from `init_matrix`
        init_matrix = np.asarray(init_matrix, dtype=float)
        if init_matrix.shape not in ((2, 2), (2, 3)):
            raise ValueError('`matrix` must have shape (2, 2) or (2, 3), '
                             'got array with shape {}'
                             ''.format(init_matrix.shape))
        trafo_matrix = init_matrix[:, :2]
        translation = init_matrix[:, 2:].squeeze()

        # Transform the default vectors
        default_det_pos_init = cls._default_config['det_pos_init']
        default_det_axis_init = cls._default_config['det_axis_init']
        vecs_to_transform = [default_det_axis_init]
        transformed_vecs = transform_system(
            default_det_pos_init, None, vecs_to_transform, matrix=trafo_matrix)

        # Use the standard constructor with these vectors
        det_pos, det_axis = transformed_vecs
        if translation.size != 0:
            kwargs['translation'] = translation

        return cls(apart, dpart, det_pos,
                   det_axis_init=det_axis, **kwargs)

    @property
    def det_axis_init(self):
        """Detector axis at angle 0."""
        return self.detector.axis

    def det_axis(self, angle):
        """Return the detector axis (axes) at ``angle``.

        Parameters
        ----------
        angle : float or `array-like`
            Angle(s) in radians describing the counter-clockwise
            rotation of the detector.

        Returns
        -------
        axis : `numpy.ndarray`
            Unit vector(s) along which the detector is aligned.
            If ``angle`` is a single parameter, the returned array has
            shape ``(2,)``, otherwise ``angle.shape + (2,)``.

        Examples
        --------
        Calling the method with a single angle produces a single vector:

        >>> apart = odl.uniform_partition(0, np.pi, 10)
        >>> dpart = odl.uniform_partition(-1, 1, 20)
        >>> geom = Parallel2dGeometry(apart, dpart)
        >>> geom.det_axis(0)
        array([ 1.,  0.])
        >>> np.allclose(geom.det_axis(np.pi / 2), [0, 1])
        True

        The method is vectorized, i.e., it can be called with multiple
        angles at once (or n-dimensional arrays of parameters):

        >>> np.allclose(geom.det_axis([0, np.pi / 2]), [[1, 0],
        ...                                             [0, 1]])
        True
        >>> geom.det_axis(np.zeros((4, 5))).shape
        (4, 5, 2)
        """
        return self.rotation_matrix(angle).dot(self.det_axis_init)

    def rotation_matrix(self, angle):
        """Return the rotation matrix to the system state at ``angle``.

        For an angle ``phi``, the matrix is given by ::

            rot(phi) = [[cos(phi), -sin(phi)],
                        [sin(phi), cos(phi)]]

        Parameters
        ----------
        angle : float or `array-like`
            Angle(s) in radians describing the counter-clockwise
            rotation of the detector.

        Returns
        -------
        rot : `numpy.ndarray`
            The rotation matrix (or matrices) mapping vectors at the
            initial state to the ones in the state defined by ``angle``.
            The rotation is extrinsic, i.e., defined in the "world"
            coordinate system.
            If ``angle`` is a single parameter, the returned array has
            shape ``(2, 2)``, otherwise ``angle.ndim + (2, 2)``.
        """
        squeeze_out = (np.shape(angle) == ())
        angle = np.array(angle, dtype=float, copy=False, ndmin=1)
        if (self.check_bounds and
                not is_inside_bounds(angle, self.motion_params)):
            raise ValueError('`angle` {} not in the valid range {}'
                             ''.format(angle, self.motion_params))

        if squeeze_out:
            matrix = euler_matrix(angle).squeeze()
        else:
            matrix = euler_matrix(angle)

        return matrix

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.motion_partition, self.det_partition]
        optargs = []

        if not np.allclose(self.det_pos_init - self.translation,
                           self._default_config['det_pos_init']):
            optargs.append(
                ['det_pos_init', array_str(self.det_pos_init), ''])

        if self._det_axis_init_arg is not None:
            optargs.append(
                ['det_axis_init', array_str(self._det_axis_init_arg), ''])

        if not np.array_equal(self.translation, (0, 0)):
            optargs.append(
                ['translation', array_str(self.translation), ''])

        sig_str = signature_string(posargs, optargs, sep=',\n')
        return '{}(\n{}\n)'.format(self.__class__.__name__, indent(sig_str))

    def __getitem__(self, indices):
        """Return self[slc]

        This is defined by::

            self[indices].partition == self.partition[indices]

        where all other parameters are the same.

        Examples
        --------
        >>> apart = odl.uniform_partition(0, 4, 4)
        >>> dpart = odl.uniform_partition(-1, 1, 20)
        >>> geom = odl.tomo.Parallel2dGeometry(apart, dpart)

        Extract sub-geometry with every second angle:

        >>> geom[::2]
        Parallel2dGeometry(
            nonuniform_partition(
                [ 0.5,  2.5],
                min_pt=0.0, max_pt=4.0
            ),
            uniform_partition(-1.0, 1.0, 20)
        )
        """
        part = self.partition[indices]
        apart = part.byaxis[0]
        dpart = part.byaxis[1]

        return Parallel2dGeometry(apart, dpart,
                                  det_pos_init=self.det_pos_init,
                                  det_axis_init=self._det_axis_init_arg,
                                  translation=self.translation)


class Parallel3dEulerGeometry(ParallelBeamGeometry):

    """Parallel beam geometry in 3d.

    The motion parameters are two or three Euler angles, and the detector
    is flat and two-dimensional.

    In the standard configuration, the detector reference point starts
    at ``(0, 1, 0)``, and the initial detector axes are
    ``[(1, 0, 0), (0, 0, 1)]``.

    For details, check `the online docs
    <https://odlgroup.github.io/odl/guide/geometry_guide.html>`_.
    """

    _default_config = dict(det_pos_init=(0, 1, 0),
                           det_axes_init=((1, 0, 0), (0, 0, 1)))

    def __init__(self, apart, dpart, det_pos_init=(0, 1, 0), **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        apart : 2- or 3-dim. `RectPartition`
            Partition of the angle parameter set.
        dpart : 2-dim. `RectPartition`
            Partition of the detector parameter set.
        det_pos_init : `array-like`, shape ``(3,)``, optional
            Initial position of the detector reference point.

        Other Parameters
        ----------------
        det_axes_init : 2-tuple of `array-like`'s (shape ``(3,)``), optional
            Initial axes defining the detector orientation. The default
            depends on ``det_pos_init``, see Notes.
        translation : `array-like`, shape ``(3,)``, optional
            Global translation of the geometry. This is added last in any
            method that computes an absolute vector, e.g., `det_refpoint`,
            and also shifts the center of rotation.
            Default: ``(0, 0, 0)``
        check_bounds : bool, optional
            If ``True``, methods computing vectors check input arguments.
            Checks are vectorized and add only a small overhead.
            Default: ``True``

        Notes
        -----
        In the default configuration, the initial detector reference point
        is ``(0, 1, 0)``, and the initial detector axes are
        ``[(1, 0, 0), (0, 0, 1)]``. If a different ``det_pos_init`` is
        chosen, the new default axes are given as a rotation of the original
        ones by a matrix that transforms ``(0, 1, 0)`` to the new
        (normalized) ``det_pos_init``. This matrix is calculated with the
        `rotation_matrix_from_to` function. Expressed in code, we have ::

            init_rot = rotation_matrix_from_to((0, 1, 0), det_pos_init)
            det_axes_init[0] = init_rot.dot((1, 0, 0))
            det_axes_init[1] = init_rot.dot((0, 0, 1))

        Examples
        --------
        Initialization with default parameters and 2 Euler angles:

        >>> apart = odl.uniform_partition([0, 0], [np.pi, 2 * np.pi],
        ...                               (10, 20))
        >>> dpart = odl.uniform_partition([-1, -1], [1, 1], (20, 20))
        >>> geom = Parallel3dEulerGeometry(apart, dpart)
        >>> geom.det_refpoint([0, 0])
        array([ 0.,  1.,  0.])
        >>> geom.det_point_position([0, 0], [-1, 1])
        array([-1.,  1.,  1.])

        Checking the default orientation:

        >>> e_x, e_y, e_z = np.eye(3)  # standard unit vectors
        >>> np.allclose(geom.det_pos_init, e_y)
        True
        >>> np.allclose(geom.det_axes_init, (e_x, e_z))
        True

        Specifying an initial detector position by default rotates the
        standard configuration to this position:

        >>> geom = Parallel3dEulerGeometry(apart, dpart,
        ...                                det_pos_init=(1, 0, 0))
        >>> np.allclose(geom.det_pos_init, e_x)
        True
        >>> np.allclose(geom.det_axes_init, (-e_y, e_z))
        True
        >>> geom = Parallel3dEulerGeometry(apart, dpart,
        ...                                det_pos_init=(0, 0, 1))
        >>> np.allclose(geom.det_pos_init, e_z)
        True
        >>> np.allclose(geom.det_axes_init, (e_x, -e_y))
        True

        The initial detector axes can also be set explicitly:

        >>> geom = Parallel3dEulerGeometry(
        ...     apart, dpart, det_pos_init=(-1, 0, 0),
        ...     det_axes_init=((0, 1, 0), (0, 0, 1)))
        >>> np.allclose(geom.det_pos_init, -e_x)
        True
        >>> np.allclose(geom.det_axes_init, (e_y, e_z))
        True
        """
        default_det_pos_init = self._default_config['det_pos_init']
        default_det_axes_init = self._default_config['det_axes_init']

        # Handle the initial coordinate system. We need to assign `None` to
        # the vectors first in order to signalize to the `transform_system`
        # utility that they should be transformed from default since they
        # were not explicitly given.
        det_axes_init = kwargs.pop('det_axes_init', None)

        # Store some stuff for repr
        if det_axes_init is not None:
            self._det_axes_init_arg = tuple(
                np.asarray(a, dtype=float) for a in det_axes_init)
        else:
            self._det_axes_init_arg = None

        # Compute the transformed system and the transition matrix. We
        # transform only those vectors that were not explicitly given.
        vecs_to_transform = []
        if det_axes_init is None:
            vecs_to_transform.extend(default_det_axes_init)

        transformed_vecs = transform_system(
            det_pos_init, default_det_pos_init, vecs_to_transform)
        transformed_vecs = list(transformed_vecs)

        det_pos_init = transformed_vecs.pop(0)
        if det_axes_init is None:
            det_axes_init = (transformed_vecs.pop(0), transformed_vecs.pop(0))
        assert transformed_vecs == []

        # Translate the absolute vectors by the given translation
        translation = np.asarray(kwargs.pop('translation', (0, 0, 0)),
                                 dtype=float)
        det_pos_init += translation

        # Initialize stuff. Normalization of the detector axis happens in
        # the detector class. `check_bounds` is needed for both detector
        # and geometry.
        check_bounds = kwargs.get('check_bounds', True)
        detector = Flat2dDetector(dpart, axes=det_axes_init,
                                  check_bounds=check_bounds)
        super(Parallel3dEulerGeometry, self).__init__(
            ndim=3, apart=apart, detector=detector,
            det_pos_init=det_pos_init, translation=translation,
            **kwargs)

        if self.motion_partition.ndim not in (2, 3):
            raise ValueError('`apart` has dimension {}, expected '
                             '2 or 3'.format(self.motion_partition.ndim))

    @classmethod
    def frommatrix(cls, apart, dpart, init_matrix, **kwargs):
        """Create an instance of `Parallel3dEulerGeometry` using a matrix.

        This alternative constructor uses a matrix to rotate and
        translate the default configuration. It is most useful when
        the transformation to be applied is already given as a matrix.

        Parameters
        ----------
        apart : 2- or 3-dim. `RectPartition`
            Partition of the parameter set consisting of 2 or 3
            Euler angles.
        dpart : 2-dim. `RectPartition`
            Partition of the detector parameter set.
        init_matrix : `array_like`, shape ``(3, 3)`` or ``(3, 4)``, optional
            Transformation matrix whose left ``(3, 3)`` block is multiplied
            with the default ``det_pos_init`` and ``det_axes_init`` to
            determine the new vectors. If present, the fourth column acts
            as a translation after the initial transformation.
            The resulting ``det_axes_init`` will be normalized.
        kwargs :
            Further keyword arguments passed to the class constructor.

        Returns
        -------
        geometry : `Parallel3dEulerGeometry`

        Examples
        --------
        Map unit vectors ``e_x -> e_z`` and ``e_z -> -e_x``, keeping the
        right-handedness:

        >>> apart = odl.uniform_partition([0, 0], [np.pi, 2 * np.pi],
        ...                               (10, 20))
        >>> dpart = odl.uniform_partition([-1, -1], [1, 1], (20, 20))
        >>> matrix = np.array([[0, 0, -1],
        ...                    [0, 1, 0],
        ...                    [1, 0, 0]])
        >>> geom = Parallel3dEulerGeometry.frommatrix(
        ...     apart, dpart, init_matrix=matrix)
        >>> geom.det_pos_init
        array([ 0.,  1.,  0.])
        >>> geom.det_axes_init
        array([[ 0.,  0.,  1.],
               [-1.,  0.,  0.]])

        Adding a translation with a fourth matrix column:

        >>> matrix = np.array([[0, 0, -1, 0],
        ...                    [0, 1, 0, 1],
        ...                    [1, 0, 0, 1]])
        >>> geom = Parallel3dEulerGeometry.frommatrix(apart, dpart, matrix)
        >>> geom.translation
        array([ 0.,  1.,  1.])
        >>> geom.det_pos_init  # (0, 1, 0) + (0, 1, 1)
        array([ 0.,  2.,  1.])
        """
        # Get transformation and translation parts from `init_matrix`
        init_matrix = np.asarray(init_matrix, dtype=float)
        if init_matrix.shape not in ((3, 3), (3, 4)):
            raise ValueError('`matrix` must have shape (3, 3) or (3, 4), '
                             'got array with shape {}'
                             ''.format(init_matrix.shape))
        trafo_matrix = init_matrix[:, :3]
        translation = init_matrix[:, 3:].squeeze()

        # Transform the default vectors
        default_det_pos_init = cls._default_config['det_pos_init']
        default_det_axes_init = cls._default_config['det_axes_init']
        vecs_to_transform = default_det_axes_init
        transformed_vecs = transform_system(
            default_det_pos_init, None, vecs_to_transform, matrix=trafo_matrix)

        # Use the standard constructor with these vectors
        det_pos, det_axis_0, det_axis_1 = transformed_vecs
        if translation.size != 0:
            kwargs['translation'] = translation

        return cls(apart, dpart, det_pos,
                   det_axes_init=[det_axis_0, det_axis_1],
                   **kwargs)

    @property
    def det_axes_init(self):
        """Initial axes of the detector."""
        return self.detector.axes

    def det_axes(self, angles):
        """Return the detector axes tuple at ``angles``.

        Parameters
        ----------
        angles : `array-like` or sequence
            Euler angles in radians describing the rotation of the detector.
            The length of the provided argument (along the first axis in
            case of an array) must be equal to the number of Euler angles
            in this geometry.

        Returns
        -------
        axes : `numpy.ndarray`
            Unit vector(s) along which the detector is aligned.
            If ``angles`` is a single pair (or triplet) of Euler angles,
            the returned array has shape ``(2, 3)``, otherwise
            ``broadcast(*angles).shape + (2, 3)``.

        Notes
        -----
        To get an array that enumerates the detector axes in the first
        dimension, move the second-to-last axis to the first position:

            axes = det_axes(angle)
            axes_enumeration = np.moveaxis(deriv, -2, 0)

        Examples
        --------
        Calling the method with a single set of angles produces a
        ``(2, 3)`` array of vertically stacked vectors:

        >>> apart = odl.uniform_partition([0, 0], [np.pi, 2 * np.pi],
        ...                               (10, 20))
        >>> dpart = odl.uniform_partition([-1, -1], [1, 1], (20, 20))
        >>> geom = Parallel3dEulerGeometry(apart, dpart)
        >>> geom.det_axes([0, 0])
        array([[ 1.,  0.,  0.],
               [ 0.,  0.,  1.]])
        >>> np.allclose(geom.det_axes([np.pi / 2, 0]), [[0, 1, 0],
        ...                                             [0, 0, 1]])
        True

        The method is vectorized, i.e., it can be called with multiple
        angle parameters at once. Each of the angle arrays can have
        different shapes and will be broadcast against each other to
        determine the final shape:

        >>> # The first axis enumerates the angles
        >>> np.allclose(geom.det_axes(([0, np.pi / 2], [0, 0])),
        ...             [[[1, 0, 0],
        ...               [0, 0, 1]],
        ...              [[0, 1, 0],
        ...               [0, 0, 1]]])
        True
        >>> # Pairs of Euler angles in a (4, 5) array each
        >>> geom.det_axes((np.zeros((4, 5)), np.zeros((4, 5)))).shape
        (4, 5, 2, 3)
        >>> # Using broadcasting for "outer product" type result
        >>> geom.det_axes((np.zeros((4, 1)), np.zeros((1, 5)))).shape
        (4, 5, 2, 3)
        """
        # Transpose to take dot along axis 1
        axes = self.rotation_matrix(angles).dot(self.det_axes_init.T)
        # `axes` has shape (a, 3, 2), need to roll the last dimensions
        # to the second to last place
        return np.rollaxis(axes, -1, -2)

    def rotation_matrix(self, angles):
        """Return the rotation matrix to the system state at ``angles``.

        Parameters
        ----------
        angles : `array-like` or sequence
            Euler angles in radians describing the rotation of the detector.
            The length of the provided argument (along the first axis in
            case of an array) must be equal to the number of Euler angles
            in this geometry.

        Returns
        -------
        rot : `numpy.ndarray`
            Rotation matrix (or matrices) mapping vectors at the
            initial state to the ones in the state defined by ``angles``.
            The rotation is extrinsic, i.e., defined in the "world"
            coordinate system.
            If ``angles`` is a single pair (or triplet) of Euler angles,
            an array of shape ``(3, 3)`` representing a single matrix is
            returned. Otherwise, the shape of the returned array is
            ``broadcast(*angles).shape + (3, 3)``.
        """
        squeeze_out = (np.broadcast(*angles).shape == ())
        angles_in = angles
        angles = tuple(np.array(angle, dtype=float, copy=False, ndmin=1)
                       for angle in angles)
        if (self.check_bounds and
                not is_inside_bounds(angles, self.motion_params)):
            raise ValueError('`angles` {} not in the valid range '
                             '{}'.format(angles_in, self.motion_params))

        matrix = euler_matrix(*angles)
        if squeeze_out:
            matrix = matrix.squeeze()

        return matrix

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.motion_partition, self.det_partition]
        optargs = []

        if not np.allclose(self.det_pos_init - self.translation,
                           self._default_config['det_pos_init']):
            optargs.append(
                ['det_pos_init', array_str(self.det_pos_init), ''])

        if self._det_axes_init_arg is not None:
            optargs.append(
                [('det_axes_init',
                  tuple(array_str(a) for a in self._det_axes_init_arg),
                  None)])

        if not np.array_equal(self.translation, (0, 0, 0)):
            optargs.append(['translation', array_str(self.translation), ''])

        sig_str = signature_string(posargs, optargs, sep=',\n')
        return '{}(\n{}\n)'.format(self.__class__.__name__, indent(sig_str))


class Parallel3dAxisGeometry(ParallelBeamGeometry, AxisOrientedGeometry):

    """Parallel beam geometry in 3d with single rotation axis.

    The motion parameter is the rotation angle around the specified
    axis, and the detector is a flat 2d detector perpendicular to the
    ray direction.

    In the standard configuration, the rotation axis is ``(0, 0, 1)``,
    the detector reference point starts at ``(0, 1, 0)``, and the
    initial detector axes are ``[(1, 0, 0), (0, 0, 1)]``.

    For details, check `the online docs
    <https://odlgroup.github.io/odl/guide/geometry_guide.html>`_.
    """

    _default_config = dict(axis=(0, 0, 1),
                           det_pos_init=(0, 1, 0),
                           det_axes_init=((1, 0, 0), (0, 0, 1)))

    def __init__(self, apart, dpart, axis=(0, 0, 1), **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        apart : 1-dim. `RectPartition`
            Partition of the angle interval.
        dpart : 2-dim. `RectPartition`
            Partition of the detector parameter rectangle.
        axis : `array-like`, shape ``(3,)``, optional
            Vector defining the fixed rotation axis of this geometry.

        Other Parameters
        ----------------
        det_pos_init : `array-like`, shape ``(3,)``, optional
            Initial position of the detector reference point.
            The default depends on ``axis``, see Notes.
        det_axes_init : 2-tuple of `array-like`'s (shape ``(3,)``), optional
            Initial axes defining the detector orientation. The default
            depends on ``axis``, see Notes.
        translation : `array-like`, shape ``(3,)``, optional
            Global translation of the geometry. This is added last in any
            method that computes an absolute vector, e.g., `det_refpoint`,
            and also shifts the axis of rotation.
            Default: ``(0, 0, 0)``
        check_bounds : bool, optional
            If ``True``, methods computing vectors check input arguments.
            Checks are vectorized and add only a small overhead.
            Default: ``True``

        Notes
        -----
        In the default configuration, the rotation axis is ``(0, 0, 1)``,
        the initial detector reference point position is ``(0, 1, 0)``,
        and the default detector axes are ``[(1, 0, 0), (0, 0, 1)]``.
        If a different ``axis`` is provided, the new default initial
        position and the new default axes are the computed by rotating
        the original ones by a matrix that transforms ``(0, 0, 1)`` to the
        new (normalized) ``axis``. This matrix is calculated with the
        `rotation_matrix_from_to` function. Expressed in code, we have ::

            init_rot = rotation_matrix_from_to((0, 0, 1), axis)
            det_pos_init = init_rot.dot((0, 1, 0))
            det_axes_init[0] = init_rot.dot((1, 0, 0))
            det_axes_init[1] = init_rot.dot((0, 0, 1))

        Examples
        --------
        Initialization with default parameters:

        >>> apart = odl.uniform_partition(0, np.pi, 10)
        >>> dpart = odl.uniform_partition([-1, -1], [1, 1], (20, 20))
        >>> geom = Parallel3dAxisGeometry(apart, dpart)
        >>> geom.det_refpoint(0)
        array([ 0.,  1.,  0.])
        >>> geom.det_point_position(0, [-1, 1])
        array([-1.,  1.,  1.])

        Checking the default orientation:

        >>> e_x, e_y, e_z = np.eye(3)  # standard unit vectors
        >>> np.allclose(geom.axis, e_z)
        True
        >>> np.allclose(geom.det_pos_init, e_y)
        True
        >>> np.allclose(geom.det_axes_init, (e_x, e_z))
        True

        Specifying an axis by default rotates the standard configuration
        to this position:

        >>> geom = Parallel3dAxisGeometry(apart, dpart, axis=(0, 1, 0))
        >>> np.allclose(geom.axis, e_y)
        True
        >>> np.allclose(geom.det_pos_init, -e_z)
        True
        >>> np.allclose(geom.det_axes_init, (e_x, e_y))
        True
        >>> geom = Parallel3dAxisGeometry(apart, dpart, axis=(1, 0, 0))
        >>> np.allclose(geom.axis, e_x)
        True
        >>> np.allclose(geom.det_pos_init, e_y)
        True
        >>> np.allclose(geom.det_axes_init, (-e_z, e_x))
        True

        The initial detector position and axes can also be set explicitly:

        >>> geom = Parallel3dAxisGeometry(
        ...     apart, dpart, det_pos_init=(-1, 0, 0),
        ...     det_axes_init=((0, 1, 0), (0, 0, 1)))
        >>> np.allclose(geom.axis, e_z)
        True
        >>> np.allclose(geom.det_pos_init, -e_x)
        True
        >>> np.allclose(geom.det_axes_init, (e_y, e_z))
        True
        """
        default_axis = self._default_config['axis']
        default_det_pos_init = self._default_config['det_pos_init']
        default_det_axes_init = self._default_config['det_axes_init']

        # Handle initial coordinate system. We need to assign `None` to
        # the vectors first since we want to check that `init_matrix`
        # is not used together with those other parameters.
        det_pos_init = kwargs.pop('det_pos_init', None)
        det_axes_init = kwargs.pop('det_axes_init', None)

        # Store some stuff for repr
        if det_pos_init is not None:
            self._det_pos_init_arg = np.asarray(det_pos_init, dtype=float)
        else:
            self._det_pos_init_arg = None

        if det_axes_init is not None:
            self._det_axes_init_arg = tuple(
                np.asarray(a, dtype=float) for a in det_axes_init)
        else:
            self._det_axes_init_arg = None

        # Compute the transformed system and the transition matrix. We
        # transform only those vectors that were not explicitly given.
        vecs_to_transform = []
        if det_pos_init is None:
            vecs_to_transform.append(default_det_pos_init)
        if det_axes_init is None:
            vecs_to_transform.extend(default_det_axes_init)

        transformed_vecs = transform_system(
            axis, default_axis, vecs_to_transform)
        transformed_vecs = list(transformed_vecs)

        axis = transformed_vecs.pop(0)
        if det_pos_init is None:
            det_pos_init = transformed_vecs.pop(0)
        if det_axes_init is None:
            det_axes_init = (transformed_vecs.pop(0), transformed_vecs.pop(0))

        assert transformed_vecs == []

        # Translate the absolute vectors by the given translation
        translation = np.asarray(kwargs.pop('translation', (0, 0, 0)),
                                 dtype=float)
        det_pos_init += translation

        # Initialize stuff. Normalization of the detector axis happens in
        # the detector class. `check_bounds` is needed for both detector
        # and geometry.
        AxisOrientedGeometry.__init__(self, axis)
        check_bounds = kwargs.get('check_bounds', True)
        detector = Flat2dDetector(dpart, axes=det_axes_init,
                                  check_bounds=check_bounds)
        super(Parallel3dAxisGeometry, self).__init__(
            ndim=3, apart=apart, detector=detector,
            det_pos_init=det_pos_init, translation=translation,
            **kwargs)

        if self.motion_partition.ndim != 1:
            raise ValueError('`apart` has dimension {}, expected 1'
                             ''.format(self.motion_partition.ndim))

    @classmethod
    def frommatrix(cls, apart, dpart, init_matrix, **kwargs):
        """Create an instance of `Parallel3dAxisGeometry` using a matrix.

        This alternative constructor uses a matrix to rotate and
        translate the default configuration. It is most useful when
        the transformation to be applied is already given as a matrix.

        Parameters
        ----------
        apart : 1-dim. `RectPartition`
            Partition of the parameter interval.
        dpart : 2-dim. `RectPartition`
            Partition of the detector parameter set.
        init_matrix : `array_like`, shape ``(3, 3)`` or ``(3, 4)``, optional
            Transformation matrix whose left ``(3, 3)`` block is multiplied
            with the default ``det_pos_init`` and ``det_axes_init`` to
            determine the new vectors. If present, the fourth column acts
            as a translation after the initial transformation.
            The resulting ``det_axes_init`` will be normalized.
        kwargs :
            Further keyword arguments passed to the class constructor.

        Returns
        -------
        geometry : `Parallel3dAxisGeometry`

        Examples
        --------
        Map unit vectors ``e_y -> e_z`` and ``e_z -> -e_y``, keeping the
        right-handedness:

        >>> apart = odl.uniform_partition(0, np.pi, 10)
        >>> dpart = odl.uniform_partition([-1, -1], [1, 1], (20, 20))
        >>> matrix = np.array([[1, 0, 0],
        ...                    [0, 0, -1],
        ...                    [0, 1, 0]])
        >>> geom = Parallel3dAxisGeometry.frommatrix(
        ...     apart, dpart, init_matrix=matrix)
        >>> geom.axis
        array([ 0., -1.,  0.])
        >>> geom.det_pos_init
        array([ 0.,  0.,  1.])
        >>> geom.det_axes_init
        array([[ 1.,  0.,  0.],
               [ 0., -1.,  0.]])

        Adding a translation with a fourth matrix column:

        >>> matrix = np.array([[0, 0, -1, 0],
        ...                    [0, 1, 0, 1],
        ...                    [1, 0, 0, 1]])
        >>> geom = Parallel3dAxisGeometry.frommatrix(apart, dpart, matrix)
        >>> geom.translation
        array([ 0.,  1.,  1.])
        >>> geom.det_pos_init  # (0, 1, 0) + (0, 1, 1)
        array([ 0.,  2.,  1.])
        """
        # Get transformation and translation parts from `init_matrix`
        init_matrix = np.asarray(init_matrix, dtype=float)
        if init_matrix.shape not in ((3, 3), (3, 4)):
            raise ValueError('`matrix` must have shape (3, 3) or (3, 4), '
                             'got array with shape {}'
                             ''.format(init_matrix.shape))
        trafo_matrix = init_matrix[:, :3]
        translation = init_matrix[:, 3:].squeeze()

        # Transform the default vectors
        default_axis = cls._default_config['axis']
        default_det_pos_init = cls._default_config['det_pos_init']
        default_det_axes_init = cls._default_config['det_axes_init']
        vecs_to_transform = (default_det_pos_init,) + default_det_axes_init
        transformed_vecs = transform_system(
            default_axis, None, vecs_to_transform, matrix=trafo_matrix)

        # Use the standard constructor with these vectors
        axis, det_pos, det_axis_0, det_axis_1 = transformed_vecs
        if translation.size != 0:
            kwargs['translation'] = translation

        return cls(apart, dpart, axis,
                   det_pos_init=det_pos,
                   det_axes_init=[det_axis_0, det_axis_1],
                   **kwargs)

    @property
    def det_axes_init(self):
        """Initial axes of the detector."""
        return self.detector.axes

    def det_axes(self, angle):
        """Return the detector axes tuple at ``angle``.

        Parameters
        ----------
        angle : float or `array-like`
            Angle(s) in radians describing the counter-clockwise rotation
            of the detector around `axis`.

        Returns
        -------
        axes : `numpy.ndarray`
            Unit vectors along which the detector is aligned.
            If ``angle`` is a single parameter, the returned array has
            shape ``(2, 3)``, otherwise
            ``broadcast(*angle).shape + (2, 3)``.

        Notes
        -----
        To get an array that enumerates the detector axes in the first
        dimension, move the second-to-last axis to the first position:

            axes = det_axes(angle)
            axes_enumeration = np.moveaxis(deriv, -2, 0)

        Examples
        --------
        Calling the method with a single angle produces a ``(2, 3)`` array
        of vertically stacked vectors:

        >>> apart = odl.uniform_partition(0, np.pi, 10)
        >>> dpart = odl.uniform_partition([-1, -1], [1, 1], (20, 20))
        >>> geom = Parallel3dAxisGeometry(apart, dpart)
        >>> geom.det_axes(0)
        array([[ 1.,  0.,  0.],
               [ 0.,  0.,  1.]])
        >>> np.allclose(geom.det_axes(np.pi / 2), [[0, 1, 0],
        ...                                        [0, 0, 1]])
        True

        The method is vectorized, i.e., it can be called with multiple
        angles at once (or n-dimensional arrays of parameters):

        >>> np.allclose(geom.det_axes([0, np.pi / 2]),
        ...             [[[1, 0, 0],
        ...               [0, 0, 1]],
        ...              [[0, 1, 0],
        ...               [0, 0, 1]]])
        True
        >>> geom.det_axes(np.zeros((4, 5))).shape
        (4, 5, 2, 3)
        """
        # Transpose to take dot along axis 1
        axes = self.rotation_matrix(angle).dot(self.det_axes_init.T)
        # `axes` has shape (a, 3, 2), need to roll the last dimensions
        # to the second to last place
        return np.rollaxis(axes, -1, -2)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.motion_partition, self.det_partition]
        optargs = []

        if not np.allclose(self.axis, self._default_config['axis']):
            optargs.append(['axis', array_str(self.axis), ''])

        if self._det_pos_init_arg is not None:
            optargs.append(['det_pos_init',
                            array_str(self._det_pos_init_arg),
                            None])

        if self._det_axes_init_arg is not None:
            optargs.append(
                ['det_axes_init',
                 tuple(array_str(a) for a in self._det_axes_init_arg),
                 None])

        if not np.array_equal(self.translation, (0, 0, 0)):
            optargs.append(['translation', array_str(self.translation), ''])

        sig_str = signature_string(posargs, optargs, sep=',\n')
        return '{}(\n{}\n)'.format(self.__class__.__name__, indent(sig_str))

    def __getitem__(self, indices):
        """Return self[indices].

        This is defined by ::

            self[indices].partition == self.partition[indices]

        where all other parameters are the same.

        Examples
        --------
        >>> apart = odl.uniform_partition(0, 4, 4)
        >>> dpart = odl.uniform_partition([-1, -1], [1, 1], [20, 20])
        >>> geom = odl.tomo.Parallel3dAxisGeometry(apart, dpart)

        Extract sub-geometry with every second angle:

        >>> geom[::2]
        Parallel3dAxisGeometry(
            nonuniform_partition(
                [ 0.5,  2.5],
                min_pt=0.0, max_pt=4.0
            ),
            uniform_partition([-1., -1.], [ 1.,  1.], (20, 20))
        )
        """
        part = self.partition[indices]
        apart = part.byaxis[0]
        dpart = part.byaxis[1:]

        return Parallel3dAxisGeometry(apart, dpart,
                                      axis=self.axis,
                                      det_pos_init=self._det_pos_init_arg,
                                      det_axes_init=self._det_axes_init_arg,
                                      translation=self.translation)

    # Manually override the abstract method in `Geometry` since it's found
    # first
    rotation_matrix = AxisOrientedGeometry.rotation_matrix


def parallel_beam_geometry(space, num_angles=None, det_shape=None):
    r"""Create default parallel beam geometry from ``space``.

    This is intended for simple test cases where users do not need the full
    flexibility of the geometries, but simply want a geometry that works.

    This default geometry gives a fully sampled sinogram according to the
    Nyquist criterion, which in general results in a very large number of
    samples. In particular, a ``space`` that is not centered at the origin
    can result in very large detectors.

    Parameters
    ----------
    space : `DiscreteLp`
        Reconstruction space, the space of the volumetric data to be projected.
        Needs to be 2d or 3d.
    num_angles : int, optional
        Number of angles.
        Default: Enough to fully sample the data, see Notes.
    det_shape : int or sequence of int, optional
        Number of detector pixels.
        Default: Enough to fully sample the data, see Notes.

    Returns
    -------
    geometry : `ParallelBeamGeometry`
        If ``space`` is 2d, return a `Parallel2dGeometry`.
        If ``space`` is 3d, return a `Parallel3dAxisGeometry`.

    Examples
    --------
    Create a parallel beam geometry from a 2d space:

    >>> space = odl.uniform_discr([-1, -1], [1, 1], (20, 20))
    >>> geometry = parallel_beam_geometry(space)
    >>> geometry.angles.size
    45
    >>> geometry.detector.size
    31

    Notes
    -----
    According to [NW2001]_, pages 72--74, a function
    :math:`f : \mathbb{R}^2 \to \mathbb{R}` that has compact support

    .. math::
        \| x \| > \rho  \implies f(x) = 0,

    and is essentially bandlimited

    .. math::
       \| \xi \| > \Omega \implies \hat{f}(\xi) \approx 0,

    can be fully reconstructed from a parallel beam ray transform
    if (1) the projection angles are sampled with a spacing of
    :math:`\Delta \psi` such that

    .. math::
        \Delta \psi \leq \frac{\pi}{\rho \Omega},

    and (2) the detector is sampled with an interval :math:`\Delta s`
    that satisfies

    .. math::
        \Delta s \leq \frac{\pi}{\Omega}.

    The geometry returned by this function satisfies these conditions exactly.

    If the domain is 3-dimensional, the geometry is "separable", in that each
    slice along the z-dimension of the data is treated as independed 2d data.

    References
    ----------
    .. [NW2001] Natterer, F and Wuebbeling, F.
       *Mathematical Methods in Image Reconstruction*.
       SIAM, 2001.
       https://dx.doi.org/10.1137/1.9780898718324
    """
    # Find maximum distance from rotation axis
    corners = space.domain.corners()[:, :2]
    rho = np.max(np.linalg.norm(corners, axis=1))

    # Find default values according to Nyquist criterion.

    # We assume that the function is bandlimited by a wave along the x or y
    # axis. The highest frequency we can measure is then a standing wave with
    # period of twice the inter-node distance.
    min_side = min(space.partition.cell_sides[:2])
    omega = np.pi / min_side
    num_px_horiz = 2 * int(np.ceil(rho * omega / np.pi)) + 1

    if space.ndim == 2:
        det_min_pt = -rho
        det_max_pt = rho
        if det_shape is None:
            det_shape = num_px_horiz
    elif space.ndim == 3:
        num_px_vert = space.shape[2]
        min_h = space.domain.min_pt[2]
        max_h = space.domain.max_pt[2]
        det_min_pt = [-rho, min_h]
        det_max_pt = [rho, max_h]
        if det_shape is None:
            det_shape = [num_px_horiz, num_px_vert]

    if num_angles is None:
        num_angles = int(np.ceil(omega * rho))

    angle_partition = uniform_partition(0, np.pi, num_angles)
    det_partition = uniform_partition(det_min_pt, det_max_pt, det_shape)

    if space.ndim == 2:
        return Parallel2dGeometry(angle_partition, det_partition)
    elif space.ndim == 3:
        return Parallel3dAxisGeometry(angle_partition, det_partition)
    else:
        raise ValueError('``space.ndim`` must be 2 or 3.')


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
