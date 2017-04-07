# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Parallel beam geometries in 2 and 3 dimensions."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

from odl.discr import uniform_partition
from odl.tomo.geometry.detector import Flat1dDetector, Flat2dDetector
from odl.tomo.geometry.geometry import Geometry, AxisOrientedGeometry
from odl.tomo.util import euler_matrix, rotation_matrix_from_to
from odl.util import signature_string, indent_rows


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
    """

    def __init__(self, ndim, apart, detector, det_pos_init):
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
        """
        super().__init__(ndim, apart, detector)

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
        """Discrete angles given in this geometry."""
        return self.motion_grid.coord_vectors[0]

    def det_refpoint(self, angle):
        """Return the position of the detector ref. point at ``angles``.

        The reference point is given by a rotation of the initial
        position by ``angles``.

        Parameters
        ----------
        angle : float
            Parameter describing the detector rotation, must be
            contained in `motion_params`.

        Returns
        -------
        point : `numpy.ndarray`, shape (`ndim`,)
            The reference point for the given parameter.
        """
        if angle not in self.motion_params:
            raise ValueError('`angle` {} not in the valid range {}'
                             ''.format(angle, self.motion_params))
        return self.rotation_matrix(angle).dot(self.det_pos_init)

    def det_to_src(self, angles, dpar, normalized=True):
        """Direction from a detector location to the source.

        In parallel geometry, this function is independent of the
        detector parameter.

        Since the (virtual) source is infinitely far away, only the
        normalized version is valid.

        Parameters
        ----------
        angles : float
            Euler angles given in radians, must be contained
            in this geometry's `motion_params`
        dpar : float
            Detector parameters, must be contained in this
            geometry's `det_params`
        normalized : bool, optional
            If ``True``, return the normalized version of the vector.
            For parallel geometry, this is the only sensible option.

        Returns
        -------
        vec : `numpy.ndarray`, shape (`ndim`,)
            Unit vector pointing from the detector to the source

        Raises
        ------
        NotImplementedError
            if ``normalized=False`` is given, since this case is not
            well defined.
        """
        if angles not in self.motion_params:
            raise ValueError('`angles` {} not in the valid range {}'
                             ''.format(angles, self.motion_params))
        if dpar not in self.det_params:
            raise ValueError('`dpar` {} not in the valid range '
                             '{}'.format(dpar, self.det_params))
        if not normalized:
            raise NotImplementedError('non-normalized detector to source is '
                                      'not available in parallel case')

        return self.rotation_matrix(angles).dot(self.detector.normal)


class Parallel2dGeometry(ParallelBeamGeometry):

    """Parallel beam geometry in 2d.

    The motion parameter is the counter-clockwise rotation angle around
    the origin, and the detector is a line detector.

    In the standard configuration, the detector is perpendicular to the
    ray direction, its reference point is initially at ``(-1, 0)``, and
    the initial detector axis is ``(0, 1)``.
    """

    _default_config = dict(det_pos_init=(-1, 0), det_axis_init=(0, 1))

    def __init__(self, apart, dpart, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        apart : 1-dim. `RectPartition`
            Partition of the angle interval.
        dpart : 1-dim. `RectPartition`
            Partition of the detector parameter interval.
        det_pos_init : `array-like`, shape ``(2,)``, optional
            Initial position of the detector reference point.
            Default: ``(-1, 0)``.
        det_axis_init : `array-like` (shape ``(2,)``), optional
            Initial axis defining the detector orientation. The default
            depends on ``det_pos_init``, see Notes.
        extra_rot : `array_like`, shape ``(2, 2)``, optional
            Rotation matrix that should be applied at the end to the
            configuration of ``det_pos_init`` and ``det_axis_init``.
            The rotation is extrinsic, i.e., defined in the "world"
            coordinate system.

        Notes
        -----
        In the default configuration, the initial detector reference point
        is ``(-1, 0)``, and the initial detector axis is ``(0, 1)``. If a
        different ``det_pos_init`` is chosen, the new default axis is
        given as a rotation of the original one by a matrix that transforms
        ``(1, 0)`` to the new (normalized) ``det_pos_init``. This matrix
        is calculated with the `rotation_matrix_from_to` function.
        Expressed in code, we have ::

            init_rot = rotation_matrix_from_to((1, 0), det_pos_init)
            det_axis_init = init_rot.dot((0, 1))

        If ``det_pos_init == [0, 0]``, no rotation is performed.

        Examples
        --------
        Initialization with default parameters:

        >>> e_x, e_y = np.eye(2)  # standard unit vectors
        >>> apart = odl.uniform_partition(0, np.pi, 10)
        >>> dpart = odl.uniform_partition(-1, 1, 20)
        >>> geom = Parallel2dGeometry(apart, dpart)
        >>> np.allclose(geom.det_pos_init, -e_x)
        True
        >>> np.allclose(geom.det_axis_init, e_y)
        True

        Specifying an initial detector position by default rotates the
        standard configuration to this position:

        >>> geom = Parallel2dGeometry(apart, dpart, det_pos_init=(0, 1))
        >>> np.allclose(geom.det_pos_init, e_y)
        True
        >>> np.allclose(geom.det_axis_init, e_x)
        True
        >>> geom = Parallel2dGeometry(apart, dpart, det_pos_init=(1, 0))
        >>> np.allclose(geom.det_pos_init, e_x)
        True
        >>> np.allclose(geom.det_axis_init, -e_y)
        True

        The initial detector axis can also be set explicitly:

        >>> geom = Parallel2dGeometry(
        ...     apart, dpart, det_pos_init=(0, 1), det_axis_init=(1, 0))
        >>> np.allclose(geom.det_pos_init, e_y)
        True
        >>> np.allclose(geom.det_axis_init, e_x)
        True

        A matrix can be given to perform a final rotation. This is most
        useful to rotate non-standard ``det_axis_init``, or if full
        control over the rotation is desired:

        >>> rot_matrix = np.array([[-1, 0],
        ...                        [0, 1]])
        >>> geom = Parallel2dGeometry(apart, dpart, extra_rot=rot_matrix)
        >>> np.allclose(geom.det_pos_init, e_x)
        True
        >>> np.allclose(geom.det_axis_init, e_y)
        True
        """
        def_det_pos = self._default_config['det_pos_init']
        def_det_axis = self._default_config['det_axis_init']

        det_pos_init = np.array(kwargs.pop('det_pos_init', def_det_pos),
                                dtype=float)

        det_axis_init = kwargs.pop('det_axis_init', None)
        extra_rot = np.asarray(kwargs.pop('extra_rot', np.eye(2)), dtype=float)

        if extra_rot.shape != (2, 2):
            raise ValueError('`extra_rot` must have shape (2, 2), got {}'
                             ''.format(extra_rot.shape))
        if abs(np.linalg.det(extra_rot)) < 1e-4:
            raise ValueError('`extra_rot` is almost singular')
        self.__extra_rotation = extra_rot

        if kwargs:
            raise TypeError('got an unexpected keyword argument {!r}'
                            ''.format(kwargs.popitem()[0]))

        if (np.linalg.norm(np.asarray(det_pos_init) - def_det_pos) < 1e-4 or
                np.linalg.norm(det_pos_init) == 0):
            # Vector close to default is mapped to default (due to
            # instability otherwise). We also take no rotation if det_pos_init
            # is zero.
            init_rot = np.eye(2)
        else:
            # Rotation due to non-standard det_pos_init
            init_rot = rotation_matrix_from_to(def_det_pos, det_pos_init)

        if det_axis_init is None:
            det_axis_init = init_rot.dot(def_det_axis)

        # Extra rotation of everything
        det_pos_init = self.extra_rotation.dot(det_pos_init)
        det_axis_init = self.extra_rotation.dot(det_axis_init)

        detector = Flat1dDetector(part=dpart, axis=det_axis_init)
        super().__init__(ndim=2, apart=apart, detector=detector,
                         det_pos_init=det_pos_init)

        if self.motion_partition.ndim != 1:
            raise ValueError('`apart` dimension {}, expected 1'
                             ''.format(self.motion_partition.ndim))

    @property
    def det_axis_init(self):
        """Detector axis at angle 0."""
        return self.detector.axis

    def det_axis(self, angle):
        """Return the detector axis at ``angle``."""
        return self.rotation_matrix(angle).dot(self.det_axis_init)

    @property
    def extra_rotation(self):
        """Rotation matrix to the initial detector configuration.

        This rotation is applied after the initial definition of detector
        position and axes.
        """
        return self.__extra_rotation

    def rotation_matrix(self, angle):
        """Return the rotation matrix for ``angle``.

        For an angle ``phi``, the matrix is given by ::

            rot(phi) = [[cos(phi), -sin(phi)],
                        [sin(phi), cos(phi)]]

        Parameters
        ----------
        angle : float
            Rotation angle given in radians, must be contained in
            this geometry's `motion_params`

        Returns
        -------
        rot : `numpy.ndarray`, shape (2, 2)
            The rotation matrix mapping the standard basis vectors in
            the fixed ("lab") coordinate system to the basis vectors of
            the local coordinate system of the detector reference point,
            expressed in the fixed system
        """
        if angle not in self.motion_params:
            raise ValueError('`angle` {} not in the valid range {}'
                             ''.format(angle, self.motion_params))
        return euler_matrix(angle)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.motion_partition, self.det_partition]
        optargs = []

        if not np.allclose(self.extra_rotation, np.eye(self.ndim)):
            inv_rot = np.linalg.inv(self.extra_rotation)
            orig_det_pos = inv_rot.dot(self.det_pos_init)
            orig_det_axis = inv_rot.dot(self.det_axis_init)
        else:
            orig_det_pos = self.det_pos_init
            orig_det_axis = self.det_axis_init

        # TODO: change array printing from tolist() to new array_str
        def_init_pos = self._default_config['det_pos_init']
        if not np.allclose(orig_det_pos, def_init_pos):
            optargs.append(('det_pos_init', orig_det_pos.tolist(), None))
            if np.linalg.norm(orig_det_pos) == 0:
                init_rot = np.eye(self.ndim)
            else:
                init_rot = rotation_matrix_from_to(def_init_pos, orig_det_pos)
            orig_det_axis = init_rot.T.dot(orig_det_axis)

        def_init_axis = self._default_config['det_axis_init']
        if not np.allclose(orig_det_axis, def_init_axis):
            det_axis_init = orig_det_axis.tolist()
            optargs.append(('det_axis_init', det_axis_init, None))

        if not np.allclose(self.extra_rotation, np.eye(self.ndim)):
            optargs.append(('extra_rot', self.extra_rotation.tolist(), None))

        sig_str = signature_string(posargs, optargs, sep=[',\n', ',\n', ',\n'])
        return '{}(\n{}\n)'.format(self.__class__.__name__,
                                   indent_rows(sig_str))


class Parallel3dEulerGeometry(ParallelBeamGeometry):

    """Parallel beam geometry in 3d.

    The motion parameters are two or three Euler angles, and the detector
    is flat and two-dimensional.

    In the standard configuration, the detector reference point starts
    at ``(-1, 0, 0)``, and the initial detector axes are
    ``[(0, 1, 0), (0, 0, 1)]``.
    """

    _default_config = dict(det_pos_init=(-1, 0, 0),
                           det_axes_init=((0, 1, 0), (0, 0, 1)))

    def __init__(self, apart, dpart, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        apart : 2- or 3-dim. `RectPartition`
            Partition of the angle parameter set
        dpart : 2-dim. `RectPartition`
            Partition of the detector parameter interval
        det_pos_init : `array-like`, shape ``(3,)``, optional
            Initial position of the detector reference point.
            Default: ``(-1, 0, 0)``
        det_axes_init : 2-tuple of `array-like`'s (shape ``(3,)``), optional
            Initial axes defining the detector orientation. The default
            depends on ``det_pos_init``, see Notes.
        extra_rot : `array_like`, shape ``(3, 3)``, optional
            Rotation matrix that should be applied at the end to the
            configuration of ``det_pos_init`` and ``det_axes_init``.
            The rotation is extrinsic, i.e., defined in the "world"
            coordinate system.

        Notes
        -----
        In the default configuration, the initial detector reference point
        is ``(-1, 0, 0)``, and the initial detector axes are
        ``[(0, 1, 0), (0, 0, 1)]``. If a different ``det_pos_init`` is
        chosen, the new default axes are given as a rotation of the original
        ones by a matrix that transforms ``(-1, 0, 0)`` to the new
        (normalized) ``det_pos_init``. This matrix is calculated with the
        `rotation_matrix_from_to` function. Expressed in code, we have ::

            init_rot = rotation_matrix_from_to((-1, 0, 0), det_pos_init)
            det_axes_init[0] = init_rot.dot((0, 1, 0))
            det_axes_init[1] = init_rot.dot((0, 0, 1))

        Examples
        --------
        Initialization with default parameters and 2 Euler angles:

        >>> e_x, e_y, e_z = np.eye(3)  # standard unit vectors
        >>> apart = odl.uniform_partition([0, 0], [np.pi, 2 * np.pi],
        ...                               (10, 20))
        >>> dpart = odl.uniform_partition([-1, -1], [1, 1], (20, 20))
        >>> geom = Parallel3dEulerGeometry(apart, dpart)
        >>> np.allclose(geom.det_pos_init, -e_x)
        True
        >>> np.allclose(geom.det_axes_init, (e_y, e_z))
        True

        Specifying an initial detector position by default rotates the
        standard configuration to this position:

        >>> geom = Parallel3dEulerGeometry(apart, dpart,
        ...                                det_pos_init=(0, 1, 0))
        >>> np.allclose(geom.det_pos_init, e_y)
        True
        >>> np.allclose(geom.det_axes_init, (e_x, e_z))
        True
        >>> geom = Parallel3dEulerGeometry(apart, dpart,
        ...                                det_pos_init=(0, 0, 1))
        >>> np.allclose(geom.det_pos_init, e_z)
        True
        >>> np.allclose(geom.det_axes_init, (e_y, e_x))
        True

        The initial detector axes can also be set explicitly:

        >>> geom = Parallel3dEulerGeometry(
        ...     apart, dpart, det_pos_init=(0, 1, 0),
        ...     det_axes_init=((-1, 0, 0), (0, 0, -1)))
        >>> np.allclose(geom.det_pos_init, e_y)
        True
        >>> np.allclose(geom.det_axes_init, (-e_x, -e_z))
        True

        A matrix can be given to perform a final rotation. This is most
        useful to rotate non-standard ``det_axes_init``, or if full
        control over the rotation is desired:

        >>> rot_matrix = np.array([[1, 0, 0],
        ...                        [0, 0, -1],
        ...                        [0, 1, 0]])
        >>> geom = Parallel3dEulerGeometry(apart, dpart, extra_rot=rot_matrix)
        >>> np.allclose(geom.det_pos_init, -e_x)  # default
        True
        >>> np.allclose(geom.det_axes_init, (e_z, -e_y))
        True
        """
        def_det_pos = self._default_config['det_pos_init']
        def_det_axes = self._default_config['det_axes_init']

        det_pos_init = np.asarray(kwargs.pop('det_pos_init', def_det_pos),
                                  dtype=float)
        det_axes_init = kwargs.pop('det_axes_init', None)
        extra_rot = np.asarray(kwargs.pop('extra_rot', np.eye(3)))

        if extra_rot.shape != (3, 3):
            raise ValueError('`extra_rot` must have shape (3, 3), got {}'
                             ''.format(extra_rot.shape))
        if abs(np.linalg.det(extra_rot)) < 1e-4:
            raise ValueError('`extra_rot` is almost singular')
        self.__extra_rotation = extra_rot

        if kwargs:
            raise TypeError('got an unexpected keyword argument {!r}'
                            ''.format(kwargs.popitem()[0]))

        if (np.linalg.norm(np.asarray(det_pos_init) - def_det_pos) < 1e-4 or
                np.linalg.norm(det_pos_init) < 1e-4):
            # Vector close to default is mapped to default (due to
            # instability otherwise). We also take no rotation if det_pos_init
            # is close to zero.
            init_rot = np.eye(3)
        else:
            # Rotation due to non-standard det_pos_init
            init_rot = rotation_matrix_from_to(def_det_pos, det_pos_init)

        if det_axes_init is None:
            det_axes_init = [init_rot.dot(a) for a in def_det_axes]

        # Extra rotation of everything
        det_pos_init = self.extra_rotation.dot(det_pos_init)
        det_axes_init = [self.extra_rotation.dot(a) for a in det_axes_init]

        detector = Flat2dDetector(part=dpart, axes=det_axes_init)
        super().__init__(ndim=3, apart=apart, detector=detector,
                         det_pos_init=det_pos_init)

        if self.motion_partition.ndim not in (2, 3):
            raise ValueError('`apart` has dimension {}, expected '
                             '2 or 3'.format(self.motion_partition.ndim))

    @property
    def det_axes_init(self):
        """Initial axes of the detector."""
        return self.detector.axes

    def det_axes(self, angles):
        """Return the detector axes tuple at ``angle``."""
        return tuple(self.rotation_matrix(angles).dot(axis)
                     for axis in self.det_axes_init)

    @property
    def extra_rotation(self):
        """Rotation matrix to the initial detector configuration.

        This rotation is applied after the initial definition of detector
        position and axes.
        """
        return self.__extra_rotation

    def rotation_matrix(self, angles):
        """Matrix defining the intrinsic rotation for ``angles``.

        Parameters
        ----------
        angles : `array-like`
            Angles in radians defining the rotation, must be contained
            in this geometry's ``motion_params``

        Returns
        -------
        rot : `numpy.ndarray`, shape ``(3, 3)``
            Rotation matrix from the initial configuration of detector
            position and axes (all angles zero) to the configuration at
            ``angles``. The rotation is extrinsic, i.e., expressed in the
            "world" coordinate system.
        """
        if angles not in self.motion_params:
            raise ValueError('`angles` {} not in the valid range {}'
                             ''.format(angles, self.motion_params))
        return euler_matrix(*angles)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.motion_partition, self.det_partition]
        optargs = []

        if not np.allclose(self.extra_rotation, np.eye(self.ndim)):
            inv_rot = np.linalg.inv(self.extra_rotation)
            orig_det_pos = inv_rot.dot(self.det_pos_init)
            orig_det_axes = [inv_rot.dot(a) for a in self.det_axes_init]
        else:
            orig_det_pos = self.det_pos_init
            orig_det_axes = self.det_axes_init

        # TODO: change array printing from tolist() to new array_str
        def_init_pos = self._default_config['det_pos_init']
        if not np.allclose(orig_det_pos, def_init_pos):
            optargs.append(('det_pos_init', orig_det_pos.tolist(), None))
            if np.linalg.norm(orig_det_pos) == 0:
                init_rot = np.eye(self.ndim)
            else:
                init_rot = rotation_matrix_from_to(def_init_pos, orig_det_pos)
            orig_det_axes = [init_rot.T.dot(a) for a in orig_det_axes]

        def_init_axes = self._default_config['det_axes_init']
        if not np.allclose(orig_det_axes, def_init_axes):
            det_axes_init = tuple(a.tolist() for a in orig_det_axes)
            optargs.append(('det_axes_init', det_axes_init, None))

        if not np.allclose(self.extra_rotation, np.eye(self.ndim)):
            optargs.append(('extra_rot', self.extra_rotation.tolist(), None))

        sig_str = signature_string(posargs, optargs, sep=[',\n', ',\n', ',\n'])
        return '{}(\n{}\n)'.format(self.__class__.__name__,
                                   indent_rows(sig_str))


class Parallel3dAxisGeometry(ParallelBeamGeometry, AxisOrientedGeometry):

    """Parallel beam geometry in 3d with single rotation axis.

    The motion parameter is the rotation angle around the specified
    axis, and the detector is a flat 2d detector perpendicular to the
    ray direction.

    In the standard configuration, the rotation axis is ``(0, 0, 1)``,
    the detector reference point starts at ``(-1, 0, 0)``, and the
    initial detector axes are ``[(0, 1, 0), (0, 0, 1)]``.
    """

    _default_config = dict(axis=(0, 0, 1),
                           det_pos_init=(-1, 0, 0),
                           det_axes_init=((0, 1, 0), (0, 0, 1)))

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
        det_pos_init : `array-like`, shape ``(3,)``, optional
            Initial position of the detector reference point.
            The default depends on ``axis``, see Notes.
        det_axes_init : 2-tuple of `array-like`'s (shape ``(3,)``), optional
            Initial axes defining the detector orientation. The default
            depends on ``axis``, see Notes.
        extra_rot : `array_like`, shape ``(3, 3)``, optional
            Rotation matrix that should be applied at the end to the
            configuration of ``det_pos_init`` and ``det_axes_init``.
            The rotation is extrinsic, i.e., defined in the "world"
            coordinate system.

        Notes
        -----
        In the default configuration, the rotation axis is ``(0, 0, 1)``,
        the initial detector reference point position is ``(-1, 0, 0)``,
        and the default detector axes are ``[(0, 1, 0), (0, 0, 1)]``.
        If a different ``axis`` is provided, the new default initial
        position and the new default axes are the computed by rotating
        the original ones by a matrix that transforms ``(0, 0, 1)`` to the
        new (normalized) ``axis``. This matrix is calculated with the
        `rotation_matrix_from_to` function. Expressed in code, we have ::

            init_rot = rotation_matrix_from_to((0, 0, 1), axis)
            det_pos_init = init_rot.dot((-1, 0, 0))
            det_axes_init[0] = init_rot.dot((0, 1, 0))
            det_axes_init[1] = init_rot.dot((0, 0, 1))

        Examples
        --------
        Initialization with default parameters:

        >>> e_x, e_y, e_z = np.eye(3)  # standard unit vectors
        >>> apart = odl.uniform_partition(0, np.pi, 10)
        >>> dpart = odl.uniform_partition([-1, -1], [1, 1], (20, 20))
        >>> geom = Parallel3dAxisGeometry(apart, dpart)
        >>> np.allclose(geom.axis, e_z)
        True
        >>> np.allclose(geom.det_pos_init, -e_x)
        True
        >>> np.allclose(geom.det_axes_init, (e_y, e_z))
        True

        Specifying an axis by default rotates the standard configuration
        to this position:

        >>> geom = Parallel3dAxisGeometry(apart, dpart, axis=(0, 1, 0))
        >>> np.allclose(geom.axis, e_y)
        True
        >>> np.allclose(geom.det_pos_init, -e_x)
        True
        >>> np.allclose(geom.det_axes_init, (-e_z, e_y))
        True
        >>> geom = Parallel3dAxisGeometry(apart, dpart, axis=(1, 0, 0))
        >>> np.allclose(geom.axis, e_x)
        True
        >>> np.allclose(geom.det_pos_init, e_z)
        True
        >>> np.allclose(geom.det_axes_init, (e_y, e_x))
        True

        The initial detector position and axes can also be set explicitly:

        >>> geom = Parallel3dAxisGeometry(apart, dpart, axis=(0, 1, 0),
        ...                               det_pos_init=(1, 0, 0))
        >>> np.allclose(geom.axis, e_y)
        True
        >>> np.allclose(geom.det_pos_init, e_x)
        True
        >>> np.allclose(geom.det_axes_init, (-e_z, e_y))  # as above
        True
        >>> geom = Parallel3dAxisGeometry(
        ...     apart, dpart, axis=(0, 1, 0),
        ...     det_axes_init=((0, 1, 0), (0, 0, 1)))
        >>> np.allclose(geom.axis, e_y)
        True
        >>> np.allclose(geom.det_pos_init, -e_x)
        True
        >>> np.allclose(geom.det_axes_init, (e_y, e_z))
        True

        A matrix can be given to perform a final rotation. This is most
        useful to rotate non-standard ``det_axes_init``, or if full
        control over the rotation is desired:

        >>> rot_matrix = np.array([[1, 0, 0],
        ...                        [0, 0, -1],
        ...                        [0, 1, 0]])
        >>> geom = Parallel3dAxisGeometry(apart, dpart, extra_rot=rot_matrix)
        >>> np.allclose(geom.axis, -e_y)
        True
        >>> np.allclose(geom.det_pos_init, -e_x)  # default
        True
        >>> np.allclose(geom.det_axes_init, (e_z, -e_y))
        True
        """
        def_axis = self._default_config['axis']
        def_det_pos = self._default_config['det_pos_init']
        def_det_axes = self._default_config['det_axes_init']

        det_pos_init = kwargs.pop('det_pos_init', None)
        det_axes_init = kwargs.pop('det_axes_init', None)
        extra_rot = np.asarray(kwargs.pop('extra_rot', np.eye(3)))

        if extra_rot.shape != (3, 3):
            raise ValueError('`extra_rot` must have shape (3, 3), got {}'
                             ''.format(extra_rot.shape))
        if abs(np.linalg.det(extra_rot)) < 1e-4:
            raise ValueError('`extra_rot` is almost singular')
        self.__extra_rotation = extra_rot

        if kwargs:
            raise TypeError('got an unexpected keyword argument {!r}'
                            ''.format(kwargs.popitem()[0]))

        if np.linalg.norm(np.asarray(axis) - def_axis) < 1e-4:
            # Vector close to default is mapped to default (due to
            # instability otherwise). We also take no rotation if det_pos_init
            # is close to zero.
            init_rot = np.eye(3)
        else:
            # Rotation due to non-standard det_pos_init
            init_rot = rotation_matrix_from_to(def_axis, axis)

        if det_pos_init is None:
            det_pos_init = init_rot.dot(def_det_pos)

        if det_axes_init is None:
            det_axes_init = [init_rot.dot(a) for a in def_det_axes]

        # Extra rotation of everything
        det_pos_init = self.extra_rotation.dot(det_pos_init)
        det_axes_init = [self.extra_rotation.dot(a) for a in det_axes_init]

        axis = self.extra_rotation.dot(axis)
        AxisOrientedGeometry.__init__(self, axis)
        detector = Flat2dDetector(part=dpart, axes=det_axes_init)
        super().__init__(ndim=3, apart=apart, detector=detector,
                         det_pos_init=det_pos_init)

        if self.motion_partition.ndim != 1:
            raise ValueError('`apart` has dimension {}, expected 1'
                             ''.format(self.motion_partition.ndim))

    @property
    def det_axes_init(self):
        """Initial axes of the detector."""
        return self.detector.axes

    def det_axes(self, angles):
        """Return the detector axes tuple at ``angle``."""
        return tuple(self.rotation_matrix(angles).dot(axis)
                     for axis in self.det_axes_init)

    @property
    def extra_rotation(self):
        """Rotation matrix to the initial detector configuration.

        This rotation is applied after the initial definition of axis,
        detector position and detector axes.
        """
        return self.__extra_rotation

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.motion_partition, self.det_partition]
        optargs = []

        if not np.allclose(self.extra_rotation, np.eye(self.ndim)):
            inv_rot = np.linalg.inv(self.extra_rotation)
            orig_axis = inv_rot.dot(self.axis)
            orig_det_pos = inv_rot.dot(self.det_pos_init)
            orig_det_axes = [inv_rot.dot(a) for a in self.det_axes_init]
        else:
            orig_axis = self.axis
            orig_det_pos = self.det_pos_init
            orig_det_axes = self.det_axes_init

        def_axis = self._default_config['axis']
        if not np.allclose(orig_axis, def_axis):
            optargs.append(('axis', orig_axis.tolist(), None))
            init_rot = rotation_matrix_from_to(def_axis, orig_axis)
            orig_det_pos = init_rot.T.dot(orig_det_pos)
            orig_det_axes = [init_rot.T.dot(a) for a in orig_det_axes]

        def_init_pos = self._default_config['det_pos_init']
        if not np.allclose(orig_det_pos, def_init_pos):
            optargs.append(('det_pos_init', orig_det_pos.tolist(), None))

        def_init_axes = self._default_config['det_axes_init']
        if not np.allclose(orig_det_axes, def_init_axes):
            det_axes_init = tuple(a.tolist() for a in orig_det_axes)
            optargs.append(('det_axes_init', det_axes_init, None))

        if not np.allclose(self.extra_rotation, np.eye(self.ndim)):
            optargs.append(('extra_rot', self.extra_rotation.tolist(), None))

        sig_str = signature_string(posargs, optargs, sep=[',\n', ',\n', ',\n'])
        return '{}(\n{}\n)'.format(self.__class__.__name__,
                                   indent_rows(sig_str))

    # Fix for bug in ABC thinking this is abstract
    rotation_matrix = AxisOrientedGeometry.rotation_matrix


def parallel_beam_geometry(space, angles=None, det_shape=None):
    """Create default parallel beam geometry from ``space``.

    This is intended for simple test cases where users do not need the full
    flexibility of the geometries, but simply want a geometry that works.

    This default geometry gives a fully sampled sinogram according to the
    Nyquist criterion, which in general results in a very large number of
    samples.

    Parameters
    ----------
    space : `DiscreteLp`
        Reconstruction space, the space of the volumetric data to be projected.
        Needs to be 2d or 3d.
    angles : int, optional
        Number of angles.
        Default: Enough to fully sample the data, see Notes.
    det_shape : int or sequence of int, optional
        Number of detector pixels.
        Default: Enough to fully sample the data, see Notes.

    Returns
    -------
    geometry : `ParallelBeamGeometry`
        If ``space`` is 2d, returns a `Parallel2dGeometry`.
        If ``space`` is 3d, returns a `Parallel3dAxisGeometry`.

    Examples
    --------
    Create geometry from 2d space and check the number of data points:

    >>> space = odl.uniform_discr([-1, -1], [1, 1], [20, 20])
    >>> geometry = parallel_beam_geometry(space)
    >>> geometry.angles.size
    45
    >>> geometry.detector.size
    29

    Notes
    -----
    According to `Mathematical Methods in Image Reconstruction`_ (page 72), for
    a function :math:`f : \\mathbb{R}^2 \\to \\mathbb{R}` that has compact
    support

    .. math::
        \| x \| > \\rho  \implies f(x) = 0,

    and is essentially bandlimited

    .. math::
       \| \\xi \| > \\Omega \implies \\hat{f}(\\xi) \\approx 0,

    then, in order to fully reconstruct the function from a parallel beam ray
    transform the function should be sampled at an angular interval
    :math:`\\Delta \psi` such that

    .. math::
        \\Delta \psi \leq \\frac{\\pi}{\\rho \\Omega},

    and the detector should be sampled with an interval :math:`Delta s` that
    satisfies

    .. math::
        \\Delta s \leq \\frac{\\pi}{\\Omega}.

    The geometry returned by this function satisfies these conditions exactly.

    If the domain is 3-dimensional, the geometry is "separable", in that each
    slice along the z-dimension of the data is treated as independed 2d data.

    References
    ----------
    .. _Mathematical Methods in Image Reconstruction: \
http://dx.doi.org/10.1137/1.9780898718324
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

    if det_shape is None:
        if space.ndim == 2:
            det_shape = int(2 * rho * omega / np.pi) + 1
        elif space.ndim == 3:
            width = int(2 * rho * omega / np.pi) + 1
            height = space.shape[2]
            det_shape = [width, height]

    if angles is None:
        angles = int(omega * rho) + 1

    # Define angles
    angle_partition = uniform_partition(0, np.pi, angles)

    if space.ndim == 2:
        det_partition = uniform_partition(-rho, rho, det_shape)
        return Parallel2dGeometry(angle_partition, det_partition)
    elif space.ndim == 3:
        min_h = space.domain.min_pt[2]
        max_h = space.domain.max_pt[2]

        det_partition = uniform_partition([-rho, min_h],
                                          [rho, max_h],
                                          det_shape)
        return Parallel3dAxisGeometry(angle_partition, det_partition)
    else:
        raise ValueError('``space.ndim`` must be 2 or 3.')


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
