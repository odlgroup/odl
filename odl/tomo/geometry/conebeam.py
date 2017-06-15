# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Cone beam geometries in 2 and 3 dimensions."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

from odl.tomo.geometry.detector import Flat1dDetector, Flat2dDetector
from odl.tomo.geometry.geometry import (
    DivergentBeamGeometry, AxisOrientedGeometry)
from odl.tomo.util.utility import (
    transform_system, euler_matrix, rotation_matrix_from_to)
from odl.util import signature_string, indent_rows


__all__ = ('FanFlatGeometry',
           'CircularConeFlatGeometry', 'HelicalConeFlatGeometry',)


class FanFlatGeometry(DivergentBeamGeometry):

    """Fan beam (2d cone beam) geometry with flat 1d detector.

    The source moves on a circle with radius ``src_radius``, and the
    detector reference point is opposite to the source, i.e. at maximum
    distance, on a circle with radius ``det_radius``. One of the two
    radii can be chosen as 0, which corresponds to a stationary source
    or detector, respectively.

    The motion parameter is the 1d rotation angle parameterizing source
    and detector positions simultaneously.

    In the standard configuration, the detector is perpendicular to the
    ray direction, its reference point is initially at ``(0, 1)``, and
    the initial detector axis is ``(1, 0)``.

    For details, check `the online docs
    <https://odlgroup.github.io/odl/guide/geometry_guide.html>`_.
    """

    _default_config = dict(src_to_det_init=(0, 1), det_axis_init=(1, 0))

    def __init__(self, apart, dpart, src_radius, det_radius,
                 src_to_det_init=(0, 1), **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        apart : 1-dim. `RectPartition`
            Partition of the angle interval.
        dpart : 1-dim. `RectPartition`
            Partition of the detector parameter interval.
        src_radius : nonnegative float
            Radius of the source circle.
        det_radius : nonnegative float
            Radius of the detector circle.
        src_to_det_init : `array-like` (shape ``(2,)``), optional
            Initial state of the vector pointing from source to detector
            reference point. The zero vector is not allowed.
            Default: ``(0, 1)``.
        det_axis_init : `array-like` (shape ``(2,)``), optional
            Initial axis defining the detector orientation. The default
            depends on ``src_to_det_init``, see Notes.

        Other Parameters
        ----------------
        init_matrix : `array_like`, shape ``(2, 2)``, optional
            Transformation matrix that should be applied to the default
            configuration of ``src_to_det_init`` and ``det_axis_init``.
            The resulting ``src_to_det_init`` and ``det_axis_init`` will
            be normalized.

            This option cannot be used together with any of the parameters
            ``src_to_det_init`` or ``det_axis_init``.

        translation : `array-like`, shape ``(2,)``, optional
            Global translation of the geometry. This is added last in any
            method that computes an absolute vector, e.g., `det_refpoint`,
            and also shifts the center of rotation.

        Notes
        -----
        In the default configuration, the initial source-to-detector vector
        is ``(0, 1)``, and the initial detector axis is ``(1, 0)``. If a
        different ``src_to_det_init`` is chosen, the new default axis is
        given as a rotation of the original one by a matrix that transforms
        ``(0, 1)`` to the new (normalized) ``src_to_det_init``. This matrix
        is calculated with the `rotation_matrix_from_to` function.
        Expressed in code, we have ::

            init_rot = rotation_matrix_from_to((0, 1), src_to_det_init)
            det_axis_init = init_rot.dot((1, 0))

        Examples
        --------
        Initialization with default parameters and some radii:

        >>> apart = odl.uniform_partition(0, 2 * np.pi, 10)
        >>> dpart = odl.uniform_partition(-1, 1, 20)
        >>> geom = FanFlatGeometry(apart, dpart, src_radius=1, det_radius=5)
        >>> np.allclose(geom.src_position(0), [0, -1])
        True
        >>> np.allclose(geom.det_refpoint(0), [0, 5])
        True
        >>> np.allclose(geom.det_point_position(0, 1), [1, 5])
        True

        Checking the default orientation:

        >>> e_x, e_y = np.eye(2)  # standard unit vectors
        >>> np.allclose(geom.src_to_det_init, e_y)
        True
        >>> np.allclose(geom.det_axis_init, e_x)
        True

        Specifying an initial detector position by default rotates the
        standard configuration to this position:

        >>> geom = FanFlatGeometry(apart, dpart, src_radius=1, det_radius=5,
        ...                        src_to_det_init=(1, 0))
        >>> np.allclose(geom.src_to_det_init, e_x)
        True
        >>> np.allclose(geom.det_axis_init, -e_y)
        True
        >>> geom = FanFlatGeometry(apart, dpart, src_radius=1, det_radius=5,
        ...                        src_to_det_init=(0, -1))
        >>> np.allclose(geom.src_to_det_init, -e_y)
        True
        >>> np.allclose(geom.det_axis_init, -e_x)
        True

        The initial detector axis can also be set explicitly:

        >>> geom = FanFlatGeometry(
        ...     apart, dpart, src_radius=1, det_radius=5,
        ...     src_to_det_init=(1, 0), det_axis_init=(0, 1))
        >>> np.allclose(geom.src_to_det_init, e_x)
        True
        >>> np.allclose(geom.det_axis_init, e_y)
        True

        A matrix can be given to perform a final rotation. This is most
        useful to rotate non-standard ``det_axis_init``, or if full
        control over the rotation is desired:

        >>> rot_matrix = np.array([[1, 0],
        ...                        [0, -1]])
        >>> geom = FanFlatGeometry(apart, dpart, src_radius=1, det_radius=5,
        ...                        init_matrix=rot_matrix)
        >>> np.allclose(geom.src_to_det_init, -e_y)
        True
        >>> np.allclose(geom.det_axis_init, e_x)
        True
        """
        default_src_to_det_init = self._default_config['src_to_det_init']
        default_det_axis_init = self._default_config['det_axis_init']

        # Handle initial coordinate system. We need to assign `None` to
        # the vectors first since we want to check that `init_matrix`
        # is not used together with those other parameters.
        det_axis_init = kwargs.pop('det_axis_init', None)
        init_matrix = kwargs.pop('init_matrix', None)
        if (init_matrix is not None and
                (not np.allclose(src_to_det_init, default_src_to_det_init) or
                 det_axis_init is not None)):
            raise TypeError('`init_matrix` cannot be combined with '
                            '`axis`, `src_to_det_init` or `det_axis_init`')

        # Store some stuff for repr
        if init_matrix is not None:
            self._init_pattern = 'matrix'
            self.__init_matrix = np.asarray(init_matrix, dtype=float)
        else:
            self._init_pattern = 'vectors'
            self.__init_matrix = np.eye(2)

        if src_to_det_init is not None:
            self._src_to_det_init_arg = np.asarray(src_to_det_init,
                                                   dtype=float)
        else:
            self._src_to_det_init_arg = None

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
            src_to_det_init, default_src_to_det_init, vecs_to_transform,
            matrix=init_matrix)

        src_to_det_init = transformed_vecs[0]
        if det_axis_init is None:
            det_axis_init = transformed_vecs[1]

        # Check and normalize `src_to_det_init`. Detector axes are
        # normalized in the detector class.
        if np.array_equiv(src_to_det_init, 0):
            raise ValueError('`src_to_det_init` cannot be the zero vector')
        else:
            src_to_det_init /= np.linalg.norm(src_to_det_init)

        # Initialize stuff
        self.__src_to_det_init = src_to_det_init
        detector = Flat1dDetector(dpart, det_axis_init)
        translation = kwargs.pop('translation', None)
        super().__init__(ndim=2, motion_part=apart, detector=detector,
                         translation=translation)

        self.__src_radius = float(src_radius)
        if self.src_radius < 0:
            raise ValueError('source circle radius {} is negative'
                             ''.format(src_radius))
        self.__det_radius = float(det_radius)
        if self.det_radius < 0:
            raise ValueError('detector circle radius {} is negative'
                             ''.format(det_radius))

        if self.src_radius == 0 and self.det_radius == 0:
            raise ValueError('source and detector circle radii cannot both be '
                             '0')

        if self.motion_partition.ndim != 1:
            raise ValueError('`apart` has dimension {}, expected 1'
                             ''.format(self.motion_partition.ndim))

        # Make sure there are no leftover kwargs
        if kwargs:
            raise TypeError('got unexpected keyword arguments {}'
                            ''.format(kwargs))

    @property
    def src_radius(self):
        """Source circle radius of this geometry."""
        return self.__src_radius

    @property
    def det_radius(self):
        """Detector circle radius of this geometry."""
        return self.__det_radius

    @property
    def src_to_det_init(self):
        """Initial source-to-detector unit vector."""
        return self.__src_to_det_init

    @property
    def det_axis_init(self):
        """Detector axis at angle 0."""
        return self.detector.axis

    @property
    def init_matrix(self):
        """Matrix mapping default to initial configuration.

        This matrix is different from the identity matrix only if
        ``init_matrix`` was provided during initialization.
        """
        return self.__init_matrix

    def det_axis(self, angle):
        """Return the detector axis at ``angle``."""
        return self.rotation_matrix(angle).dot(self.det_axis_init)

    @property
    def angles(self):
        """Discrete angles given in this geometry."""
        return self.motion_grid.coord_vectors[0]

    def src_position(self, angle):
        """Return the source position at ``angle``.

        For an angle ``phi``, the source position is given by::

            src(phi) = translation +
                       rot_matrix(phi) * (-src_rad * src_to_det_init)

        where ``src_to_det_init`` is the initial unit vector pointing
        from source to detector.

        Parameters
        ----------
        angle : float
            Rotation angle given in radians, must be contained in
            this geometry's `motion_params`.

        Returns
        -------
        point : `numpy.ndarray`, shape ``(2,)``
            Source position corresponding to the given angle.

        Examples
        --------
        With default arguments, the source starts at ``src_rad * (-e_y)``
        and rotates to ``src_rad * e_x`` at 90 degrees:

        >>> apart = odl.uniform_partition(0, 2 * np.pi, 10)
        >>> dpart = odl.uniform_partition(-1, 1, 20)
        >>> geom = FanFlatGeometry(apart, dpart, src_radius=2, det_radius=5)
        >>> np.allclose(geom.src_position(0), [0, -2])
        True
        >>> np.allclose(geom.src_position(np.pi / 2), [2, 0])
        True
        """
        if angle not in self.motion_params:
            raise ValueError('`angle` {} is not in the valid range {}'
                             ''.format(angle, self.motion_params))

        # Initial vector from 0 to the source. It can be computed this way
        # since source and detector are at maximum distance, i.e. the
        # connecting line passes the origin.
        origin_to_src_init = -self.src_radius * self.src_to_det_init
        return (self.translation +
                self.rotation_matrix(angle).dot(origin_to_src_init))

    def det_refpoint(self, angle):
        """Return the detector reference point position at ``angle``.

        For an angle ``phi``, the detector position is given by ::

            det_ref(phi) = translation +
                           rot_matrix(phi) * (det_rad * src_to_det_init)

        where ``src_to_det_init`` is the initial unit vector pointing
        from source to detector.

        Parameters
        ----------
        angle : float
            Rotation angle given in radians, must be contained in
            this geometry's `motion_params`

        Returns
        -------
        point : `numpy.ndarray`, shape (2,)
            Detector reference point corresponding to the given angle

        See Also
        --------
        rotation_matrix

        Examples
        --------
        With default arguments, the detector starts at ``det_rad * e_y``
        and rotates to ``det_rad * (-e_x)`` at 90 degrees:

        >>> apart = odl.uniform_partition(0, 2 * np.pi, 10)
        >>> dpart = odl.uniform_partition(-1, 1, 20)
        >>> geom = FanFlatGeometry(apart, dpart, src_radius=2, det_radius=5)
        >>> np.allclose(geom.det_refpoint(0), [0, 5])
        True
        >>> np.allclose(geom.det_refpoint(np.pi / 2), [-5, 0])
        True
        """
        if angle not in self.motion_params:
            raise ValueError('`angle` {} is not in the valid range {}'
                             ''.format(angle, self.motion_params))

        # Initial vector from 0 to the detector. It can be computed this way
        # since source and detector are at maximum distance, i.e. the
        # connecting line passes the origin.
        origin_to_det_init = self.det_radius * self.src_to_det_init
        return (self.translation +
                self.rotation_matrix(angle).dot(origin_to_det_init))

    def rotation_matrix(self, angle):
        """Return the rotation matrix for ``angle``.

        For an angle ``phi``, the matrix is given by ::

            rot(phi) = [[cos(phi), -sin(phi)],
                        [sin(phi), cos(phi)]]

        Parameters
        ----------
        angle : float
            Rotation angle given in radians, must be contained in
            this geometry's `motion_params`.

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
            raise ValueError('`angle` {} not in the valid range {}'
                             ''.format(angle, self.motion_params))
        return euler_matrix(angle)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.motion_partition, self.det_partition]
        optargs = [('src_radius', self.src_radius, -1),
                   ('det_radius', self.det_radius, -1)]

        if self._init_pattern == 'vectors':
            if not np.allclose(self.src_to_det_init,
                               self._default_config['src_to_det_init']):
                optargs.append(
                    ['src_to_det_init', self.src_to_det_init.tolist(), None])

            if self._det_axis_init_arg is not None:
                optargs.append(
                    ['det_axis_init', self._det_axis_init_arg.tolist(), None])

        elif self._init_pattern == 'matrix':
            if not np.allclose(self.init_matrix, np.eye(2)):
                optargs.append(
                    ['init_matrix', self.init_matrix.tolist(), None])

        else:
            raise RuntimeError('bad `_init_pattern`')

        if not np.array_equal(self.translation, (0, 0)):
            optargs.append(['translation', self.translation.tolist(), None])

        sig_str = signature_string(posargs, optargs, sep=',\n')
        return '{}(\n{}\n)'.format(self.__class__.__name__,
                                   indent_rows(sig_str))


class HelicalConeFlatGeometry(DivergentBeamGeometry, AxisOrientedGeometry):

    """Cone beam geometry with helical source curve and flat detector.

    The source moves along a spiral oriented along a fixed ``axis``, with
    radius ``src_radius`` in the azimuthal plane and a given ``pitch``.
    The detector reference point is opposite to the source, i.e. in
    the point at distance ``src_rad + det_rad`` on the line in the
    azimuthal plane through the source point and ``axis``.

    The motion parameter is the 1d rotation angle parameterizing source
    and detector positions simultaneously.

    In the standard configuration, the rotation axis is ``(0, 0, 1)``,
    the initial source-to-detector vector is ``(0, 1, 0)``, and the
    initial detector axes are ``[(1, 0, 0), (0, 0, 1)]``.

    See Also
    --------
    CircularConeFlatGeometry : Case with zero pitch
    """

    _default_config = dict(axis=(0, 0, 1),
                           src_to_det_init=(0, 1, 0),
                           det_axes_init=((1, 0, 0), (0, 0, 1)))

    def __init__(self, apart, dpart, src_radius, det_radius, pitch,
                 axis=(0, 0, 1), **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        apart : 1-dim. `RectPartition`
            Partition of the angle interval.
        dpart : 2-dim. `RectPartition`
            Partition of the detector parameter rectangle.
        src_radius : nonnegative float
            Radius of the source circle.
        det_radius : nonnegative float
            Radius of the detector circle.
        pitch : float
            Constant vertical distance that a point on the helix
            traverses when increasing the angle parameter by ``2 * pi``.
        axis : `array-like`, shape ``(3,)``, optional
            Vector defining the fixed rotation axis of this geometry.

        Other Parameters
        ----------------
        pitch_offset : float, optional
            Offset along the ``axis`` at ``angle=0``. Default: 0.
        src_to_det_init : `array-like`, shape ``(2,)``, optional
            Initial state of the vector pointing from source to detector
            reference point. The zero vector is not allowed.
            The default depends on ``axis``, see Notes.
        det_axes_init : 2-tuple of `array-like`'s (shape ``(2,)``), optional
            Initial axes defining the detector orientation. The default
            depends on ``axis``, see Notes.
        extra_rot : `array_like`, shape ``(3, 3)``, optional
            Rotation matrix that should be applied at the end to the
            configuration of ``src_to_det_init`` and ``det_axes_init``.
            The rotation is extrinsic, i.e., defined in the "world"
            coordinate system.

        Notes
        -----
        In the default configuration, the rotation axis is ``(0, 0, 1)``,
        the initial source-to-detector direction is ``(0, 1, 0)``,
        and the default detector axes are ``[(1, 0, 0), (0, 0, 1)]``.
        If a different ``axis`` is provided, the new default initial
        position and the new default axes are the computed by rotating
        the original ones by a matrix that transforms ``(0, 0, 1)`` to the
        new (normalized) ``axis``. This matrix is calculated with the
        `rotation_matrix_from_to` function. Expressed in code, we have ::

            init_rot = rotation_matrix_from_to((0, 0, 1), axis)
            src_to_det_init = init_rot.dot((0, 1, 0))
            det_axes_init[0] = init_rot.dot((1, 0, 0))
            det_axes_init[1] = init_rot.dot((0, 0, 1))

        Examples
        --------
        Initialization with default parameters and some (arbitrary)
        choices for pitch and radii:

        >>> e_x, e_y, e_z = np.eye(3)  # standard unit vectors
        >>> apart = odl.uniform_partition(0, 4 * np.pi, 10)
        >>> dpart = odl.uniform_partition([-1, -1], [1, 1], (20, 20))
        >>> geom = HelicalConeFlatGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10, pitch=2)
        >>> geom.src_position(0)
        array([ 0., -5.,  0.])
        >>> geom.det_refpoint(0)
        array([ 0., 10.,  0.])
        >>> np.allclose(geom.src_position(2 * np.pi),
        ...     geom.src_position(0) + (0, 0, 2))  # z shift due to pitch
        True
        >>> np.allclose(geom.axis, e_z)
        True
        >>> np.allclose(geom.src_to_det_init, e_y)
        True
        >>> np.allclose(geom.det_axes_init, (e_x, e_z))
        True

        Specifying an axis by default rotates the standard configuration
        to this position:

        >>> geom = HelicalConeFlatGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10, pitch=2,
        ...     axis=(0, 1, 0))
        >>> np.allclose(geom.axis, e_y)
        True
        >>> np.allclose(geom.src_to_det_init, -e_z)
        True
        >>> np.allclose(geom.det_axes_init, (e_x, e_y))
        True
        >>> geom = HelicalConeFlatGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10, pitch=2,
        ...     axis=(1, 0, 0))
        >>> np.allclose(geom.axis, e_x)
        True
        >>> np.allclose(geom.src_to_det_init, e_y)
        True
        >>> np.allclose(geom.det_axes_init, (-e_z, e_x))
        True

        The initial source-to-detector vector and the detector axes can
        also be set explicitly:

        >>> geom = HelicalConeFlatGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10, pitch=2,
        ...     src_to_det_init=(-1, 0, 0),
        ...     det_axes_init=((0, 1, 0), (0, 0, 1)))
        >>> np.allclose(geom.axis, e_z)
        True
        >>> np.allclose(geom.src_to_det_init, -e_x)
        True
        >>> np.allclose(geom.det_axes_init, (e_y, e_z))
        True

        A matrix can be given to perform a final rotation. This is most
        useful to rotate non-standard ``det_axes_init``, or if full
        control over the rotation is desired:

        >>> rot_matrix = np.array([[1, 0, 0],
        ...                        [0, 0, -1],
        ...                        [0, 1, 0]])
        >>> geom = HelicalConeFlatGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10, pitch=2,
        ...     extra_rot=rot_matrix)
        >>> np.allclose(geom.axis, -e_y)
        True
        >>> np.allclose(geom.src_to_det_init, e_z)
        True
        >>> np.allclose(geom.det_axes_init, (e_x, -e_y))
        True
        """
        default_axis = self._default_config['axis']
        default_src_to_det = self._default_config['src_to_det_init']
        default_det_axes = self._default_config['det_axes_init']

        src_to_det_init = kwargs.pop('src_to_det_init', None)
        det_axes_init = kwargs.pop('det_axes_init', None)
        extra_rot = np.asarray(kwargs.pop('extra_rot', np.eye(3)))
        if extra_rot.shape != (3, 3):
            raise ValueError('`extra_rot` must have shape (3, 3), got {}'
                             ''.format(extra_rot.shape))
        if abs(np.linalg.det(extra_rot)) < 1e-4:
            raise ValueError('`extra_rot` is almost singular')
        self.__extra_rotation = extra_rot

        if np.allclose(axis, default_axis, rtol=1e-3):
            # Vector close to default is mapped to default (due to
            # instability otherwise)
            init_rot = np.eye(3)
        else:
            # Rotation due to non-standard src_to_det_init
            init_rot = rotation_matrix_from_to(default_axis, axis)

        if src_to_det_init is None:
            src_to_det_init = init_rot.dot(default_src_to_det)
        if np.linalg.norm(src_to_det_init) <= 1e-10:
            raise ValueError('initial source to detector vector {} is too '
                             'close to zero'.format(src_to_det_init))

        if det_axes_init is None:
            det_axes_init = [init_rot.dot(a) for a in default_det_axes]

        # Extra rotation of everything
        src_to_det_init = self.extra_rotation.dot(src_to_det_init)
        det_axes_init = [self.extra_rotation.dot(a) for a in det_axes_init]

        axis = self.extra_rotation.dot(axis)
        AxisOrientedGeometry.__init__(self, axis)

        self.__src_to_det_init = (np.array(src_to_det_init) /
                                  np.linalg.norm(src_to_det_init))

        detector = Flat2dDetector(dpart, det_axes_init)
        super().__init__(ndim=3, motion_part=apart, detector=detector)

        self.__pitch = float(pitch)
        self.__pitch_offset = float(kwargs.pop('pitch_offset', 0))
        self.__src_radius = float(src_radius)
        if self.src_radius < 0:
            raise ValueError('source circle radius {} is negative'
                             ''.format(src_radius))
        self.__det_radius = float(det_radius)
        if self.det_radius < 0:
            raise ValueError('detector circle radius {} is negative'
                             ''.format(det_radius))

        if self.src_radius == 0 and self.det_radius == 0:
            raise ValueError('source and detector circle radii cannot both be '
                             '0')

        if kwargs:
            raise TypeError('got an unexpected keyword argument {!r}'
                            ''.format(kwargs.popitem()[0]))

    @property
    def src_radius(self):
        """Source circle radius of this geometry."""
        return self.__src_radius

    @property
    def det_radius(self):
        """Detector circle radius of this geometry."""
        return self.__det_radius

    @property
    def pitch(self):
        """Constant vertical distance traversed in a full rotation."""
        return self.__pitch

    @property
    def src_to_det_init(self):
        """Initial state of the vector pointing from source to detector
        reference point."""
        return self.__src_to_det_init

    @property
    def det_axes_init(self):
        """Initial axes defining the detector orientation."""
        return self.detector.axes

    def det_axes(self, angles):
        """Return the detector axes tuple at ``angle``."""
        return tuple(self.rotation_matrix(angles).dot(axis)
                     for axis in self.det_axes_init)

    @property
    def pitch_offset(self):
        """Vertical offset at ``angle=0``."""
        return self.__pitch_offset

    @property
    def extra_rotation(self):
        """Rotation matrix to the initial detector configuration.

        This rotation is applied after the initial definition of axis,
        source-to-detector vector and detector axes.
        """
        return self.__extra_rotation

    @property
    def angles(self):
        """Discrete angles given in this geometry."""
        return self.motion_grid.coord_vectors[0]

    def det_refpoint(self, angle):
        """Return the detector reference point position at ``angle``.

        For an angle ``phi``, the detector position is given by::

            ref(phi) = det_rad * rot_matrix(phi) * src_to_det_init +
                       (pitch_offset + pitch * phi) * axis

        where ``src_to_det_init`` is the initial unit vector pointing
        from source to detector.

        Parameters
        ----------
        angle : float
            Rotation angle given in radians, must be contained in
            this geometry's `motion_params`

        Returns
        -------
        point : `numpy.ndarray`, shape (3,)
            Detector reference point corresponding to the given angle

        See Also
        --------
        rotation_matrix
        """
        angle = float(angle)
        if angle not in self.motion_params:
            raise ValueError('`angle` {} is not in the valid range {}'
                             ''.format(angle, self.motion_params))

        # Initial vector from 0 to the detector. It can be computed this way
        # since source and detector are at maximum distance, i.e. the
        # connecting line passes the origin.
        origin_to_det_init = self.det_radius * self.src_to_det_init
        circle_component = self.rotation_matrix(angle).dot(origin_to_det_init)

        # Increment along the rotation axis according to pitch and pitch_offset
        pitch_component = self.axis * (self.pitch_offset +
                                       self.pitch * angle / (2 * np.pi))

        return circle_component + pitch_component

    def src_position(self, angle):
        """Return the source position at ``angle``.

        For an angle ``phi``, the source position is given by::

            src(phi) = -src_rad * rot_matrix(phi) * src_to_det_init +
                       (pitch_offset + pitch * phi) * axis

        where ``src_to_det_init`` is the initial unit vector pointing
        from source to detector.

        Parameters
        ----------
        angle : float
            Rotation angle given in radians, must be contained in
            this geometry's `motion_params`

        Returns
        -------
        point : `numpy.ndarray`, shape (3,)
            Detector reference point corresponding to the given angle

        See Also
        --------
        rotation_matrix
        """
        angle = float(angle)
        if angle not in self.motion_params:
            raise ValueError('`angle` {} is not in the valid range {}'
                             ''.format(angle, self.motion_params))

        # Initial vector from 0 to the source. It can be computed this way
        # since source and detector are at maximum distance, i.e. the
        # connecting line passes the origin.
        origin_to_src_init = -self.src_radius * self.src_to_det_init
        circle_component = self.rotation_matrix(angle).dot(origin_to_src_init)

        # Increment by pitch
        pitch_component = self.axis * (self.pitch_offset +
                                       self.pitch * angle / (np.pi * 2))

        return circle_component + pitch_component

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.motion_partition, self.det_partition]
        optargs = [('src_radius', self.src_radius, -1),
                   ('det_radius', self.det_radius, -1),
                   ('pitch', self.pitch, 0)  # 0 for CircularConeFlatGeometry
                   ]

        if not np.allclose(self.extra_rotation, np.eye(3)):
            inv_rot = np.linalg.inv(self.extra_rotation)
            orig_axis = inv_rot.dot(self.axis)
            orig_src_to_det = inv_rot.dot(self.src_to_det_init)
            orig_det_axes = [inv_rot.dot(a) for a in self.det_axes_init]
        else:
            orig_axis = self.axis
            orig_src_to_det = self.src_to_det_init
            orig_det_axes = self.det_axes_init

        default_axis = self._default_config['axis']
        if not np.allclose(orig_axis, default_axis):
            optargs.append(('axis', orig_axis.tolist(), None))
            init_rot = rotation_matrix_from_to(default_axis, orig_axis)
            orig_src_to_det = init_rot.T.dot(orig_src_to_det)
            orig_det_axes = [init_rot.T.dot(a) for a in orig_det_axes]

        optargs.append(('pitch_offset', self.pitch_offset, 0))

        default_src_to_det = self._default_config['src_to_det_init']
        if not np.allclose(orig_src_to_det, default_src_to_det):
            optargs.append(('src_to_det_init', orig_src_to_det.tolist(), None))

        default_init_axes = self._default_config['det_axes_init']
        if not np.allclose(orig_det_axes, default_init_axes):
            det_axes_init = tuple(a.tolist() for a in orig_det_axes)
            optargs.append(('det_axes_init', det_axes_init, None))

        if not np.allclose(self.extra_rotation, np.eye(3)):
            optargs.append(('extra_rot', self.extra_rotation.tolist(), None))

        sig_str = signature_string(posargs, optargs, sep=[',\n', ',\n', ',\n'])
        return '{}(\n{}\n)'.format(self.__class__.__name__,
                                   indent_rows(sig_str))

    # Fix for bug in ABC thinking this is abstract
    rotation_matrix = AxisOrientedGeometry.rotation_matrix


class CircularConeFlatGeometry(HelicalConeFlatGeometry):

    """Cone beam geometry with circular source curve and flat detector.

    The source moves along a circle with radius ``src_radius`` in the
    plane perpendicular to a fixed ``axis``. The detector reference
    point is opposite to the source, i.e. in the same plane on a circle
    with radius ``det_rad`` at maximum distance to the source. This
    implies that it lies on the line through the source point and
    the intersection of the ``axis`` with the azimuthal plane.

    The motion parameter is the 1d rotation angle parameterizing source
    and detector positions simultaneously.

    In the standard configuration, the rotation axis is ``(0, 0, 1)``,
    the initial source-to-detector vector is ``(0, 1, 0)``, and the
    initial detector axes are ``[(1, 0, 0), (0, 0, 1)]``.

    See Also
    --------
    HelicalConeFlatGeometry : General case with motion in z direction
    """

    def __init__(self, apart, dpart, src_radius, det_radius, axis=(0, 0, 1),
                 **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        apart : 1-dim. `RectPartition`
            Partition of the angle interval.
        dpart : 2-dim. `RectPartition`
            Partition of the detector parameter rectangle.
        src_radius : nonnegative float
            Radius of the source circle.
        det_radius : nonnegative float
            Radius of the detector circle.
        axis : `array-like`, shape ``(3,)``, optional
            Vector defining the fixed rotation axis of this geometry.

        Other Parameters
        ----------------
        src_to_det_init : `array-like`, shape ``(2,)``, optional
            Initial state of the vector pointing from source to detector
            reference point. The zero vector is not allowed.
            The default depends on ``axis``, see Notes.
        det_axes_init : 2-tuple of `array-like`'s (shape ``(2,)``), optional
            Initial axes defining the detector orientation. The default
            depends on ``axis``, see Notes.
        extra_rot : `array_like`, shape ``(3, 3)``, optional
            Rotation matrix that should be applied at the end to the
            configuration of ``src_to_det_init`` and ``det_axes_init``.
            The rotation is extrinsic, i.e., defined in the "world"
            coordinate system.

        Notes
        -----
        In the default configuration, the rotation axis is ``(0, 0, 1)``,
        the initial source-to-detector direction is ``(0, 1, 0)``,
        and the default detector axes are ``[(1, 0, 0), (0, 0, 1)]``.
        If a different ``axis`` is provided, the new default initial
        position and the new default axes are the computed by rotating
        the original ones by a matrix that transforms ``(0, 0, 1)`` to the
        new (normalized) ``axis``. This matrix is calculated with the
        `rotation_matrix_from_to` function. Expressed in code, we have ::

            init_rot = rotation_matrix_from_to((0, 0, 1), axis)
            src_to_det_init = init_rot.dot((0, 1, 0))
            det_axes_init[0] = init_rot.dot((1, 0, 0))
            det_axes_init[1] = init_rot.dot((0, 0, 1))

        Examples
        --------
        Initialization with default parameters and some (arbitrary)
        choices for the radii:

        >>> e_x, e_y, e_z = np.eye(3)  # standard unit vectors
        >>> apart = odl.uniform_partition(0, 4 * np.pi, 10)
        >>> dpart = odl.uniform_partition([-1, -1], [1, 1], (20, 20))
        >>> geom = CircularConeFlatGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10)
        >>> geom.src_position(0)
        array([ 0., -5.,  0.])
        >>> geom.det_refpoint(0)
        array([ 0., 10.,  0.])
        >>> np.allclose(geom.src_position(2 * np.pi), geom.src_position(0))
        True
        >>> np.allclose(geom.axis, e_z)
        True
        >>> np.allclose(geom.src_to_det_init, e_y)
        True
        >>> np.allclose(geom.det_axes_init, (e_x, e_z))
        True

        Specifying an axis by default rotates the standard configuration
        to this position:

        >>> geom = CircularConeFlatGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10,
        ...     axis=(0, 1, 0))
        >>> np.allclose(geom.axis, e_y)
        True
        >>> np.allclose(geom.src_to_det_init, -e_z)
        True
        >>> np.allclose(geom.det_axes_init, (e_x, e_y))
        True
        >>> geom = CircularConeFlatGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10,
        ...     axis=(1, 0, 0))
        >>> np.allclose(geom.axis, e_x)
        True
        >>> np.allclose(geom.src_to_det_init, e_y)
        True
        >>> np.allclose(geom.det_axes_init, (-e_z, e_x))
        True

        The initial source-to-detector vector and the detector axes can
        also be set explicitly:

        >>> geom = CircularConeFlatGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10,
        ...     src_to_det_init=(-1, 0, 0),
        ...     det_axes_init=((0, 1, 0), (0, 0, 1)))
        >>> np.allclose(geom.axis, e_z)
        True
        >>> np.allclose(geom.src_to_det_init, -e_x)
        True
        >>> np.allclose(geom.det_axes_init, (e_y, e_z))
        True

        A matrix can be given to perform a final rotation. This is most
        useful to rotate non-standard ``det_axes_init``, or if full
        control over the rotation is desired:

        >>> rot_matrix = np.array([[1, 0, 0],
        ...                        [0, 0, -1],
        ...                        [0, 1, 0]])
        >>> geom = CircularConeFlatGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10,
        ...     extra_rot=rot_matrix)
        >>> np.allclose(geom.axis, -e_y)
        True
        >>> np.allclose(geom.src_to_det_init, e_z)
        True
        >>> np.allclose(geom.det_axes_init, (e_x, -e_y))
        True
        """
        # For a better error message
        for key in ('pitch', 'pitch_offset'):
            if key in kwargs:
                raise TypeError('got an unexpected keyword argument {!r}'
                                ''.format(key))

        super().__init__(apart, dpart, src_radius, det_radius, pitch=0,
                         axis=axis, **kwargs)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
