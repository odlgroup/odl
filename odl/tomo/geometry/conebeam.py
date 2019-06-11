# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Cone beam geometries in 2 and 3 dimensions."""

from __future__ import print_function, division, absolute_import
import numpy as np

from odl.discr import uniform_partition
from odl.tomo.geometry.detector import (
    Flat1dDetector, Flat2dDetector,
    CircularDetector, CylindricalDetector, SphericalDetector)
from odl.tomo.geometry.geometry import (
    DivergentBeamGeometry, AxisOrientedGeometry)
from odl.tomo.util.utility import (
    euler_matrix, transform_system, is_inside_bounds)
from odl.util import signature_string, indent, array_str


__all__ = ('FanBeamGeometry', 'ConeBeamGeometry',
           'cone_beam_geometry', 'helical_geometry')


class FanBeamGeometry(DivergentBeamGeometry):

    """Fan beam (2d cone beam) geometry.

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
                 det_curvature_radius=None, src_to_det_init=(0, 1), **kwargs):
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
            Radius of the detector circle. Must be nonzero if ``src_radius``
            is zero.
        det_curvature_radius : nonnegative float, optional
            Radius of the detector curvature.
            If ``None``, a flat detector is used, otherwise must be positive.
        src_to_det_init : `array-like` (shape ``(2,)``), optional
            Initial state of the vector pointing from source to detector
            reference point. The zero vector is not allowed.

        Other Parameters
        ----------------
        det_axis_init : `array-like` (shape ``(2,)``), optional
            Initial axis defining the detector orientation. The default
            depends on ``src_to_det_init``, see Notes.
        translation : `array-like`, shape ``(2,)``, optional
            Global translation of the geometry. This is added last in any
            method that computes an absolute vector, e.g., `det_refpoint`,
            and also shifts the center of rotation.
        check_bounds : bool, optional
            If ``True``, methods computing vectors check input arguments.
            Checks are vectorized and add only a small overhead.
            Default: ``True``

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
        >>> geom = FanBeamGeometry(apart, dpart, src_radius=1, det_radius=5)
        >>> geom.src_position(0)
        array([ 0., -1.])
        >>> geom.det_refpoint(0)
        array([ 0.,  5.])
        >>> geom.det_point_position(0, 1)  # (0, 5) + 1 * (1, 0)
        array([ 1.,  5.])

        Checking the default orientation:

        >>> geom.src_to_det_init
        array([ 0.,  1.])
        >>> geom.det_axis_init
        array([ 1.,  0.])

        Specifying curvature of the detector:

        >>> apart = odl.uniform_partition(0, 2 * np.pi, 10)
        >>> dpart = odl.uniform_partition(-np.pi / 2, np.pi / 2, 10)
        >>> geom = FanBeamGeometry(apart, dpart, src_radius=1, det_radius=5,
        ...                        det_curvature_radius=10)
        >>> geom.src_position(0)
        array([ 0., -1.])
        >>> geom.det_refpoint(0)
        array([ 0.,  5.])
        >>> # (0, 5) + 10 * (sin(pi/6), cos(pi/6) - 1)
        >>> np.round(geom.det_point_position(0, np.pi / 6), 2)
        array([ 5.  ,  3.66])

        Specifying an initial detector position by default rotates the
        standard configuration to this position:

        >>> e_x, e_y = np.eye(2)  # standard unit vectors
        >>> geom = FanBeamGeometry(apart, dpart, src_radius=1, det_radius=5,
        ...                        src_to_det_init=(1, 0))
        >>> np.allclose(geom.src_to_det_init, e_x)
        True
        >>> np.allclose(geom.det_axis_init, -e_y)
        True
        >>> geom = FanBeamGeometry(apart, dpart, src_radius=1, det_radius=5,
        ...                        src_to_det_init=(0, -1))
        >>> np.allclose(geom.src_to_det_init, -e_y)
        True
        >>> np.allclose(geom.det_axis_init, -e_x)
        True

        The initial detector axis can also be set explicitly:

        >>> geom = FanBeamGeometry(
        ...     apart, dpart, src_radius=1, det_radius=5,
        ...     src_to_det_init=(1, 0), det_axis_init=(0, 1))
        >>> np.allclose(geom.src_to_det_init, e_x)
        True
        >>> np.allclose(geom.det_axis_init, e_y)
        True
        """
        default_src_to_det_init = self._default_config['src_to_det_init']
        default_det_axis_init = self._default_config['det_axis_init']

        # Handle the initial coordinate system. We need to assign `None` to
        # the vectors first in order to signalize to the `transform_system`
        # utility that they should be transformed from default since they
        # were not explicitly given.
        det_axis_init = kwargs.pop('det_axis_init', None)

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
            src_to_det_init, default_src_to_det_init, vecs_to_transform)
        transformed_vecs = list(transformed_vecs)

        src_to_det_init = transformed_vecs.pop(0)
        if det_axis_init is None:
            det_axis_init = transformed_vecs.pop(0)
        assert transformed_vecs == []

        # Check and normalize `src_to_det_init`. Detector axes are
        # normalized in the detector class.
        if np.array_equiv(src_to_det_init, 0):
            raise ValueError('`src_to_det_init` cannot be the zero vector')
        else:
            src_to_det_init /= np.linalg.norm(src_to_det_init)

        # Initialize stuff
        self.__src_to_det_init = src_to_det_init
        # `check_bounds` is needed for both detector and geometry
        check_bounds = kwargs.get('check_bounds', True)
        if det_curvature_radius is None:
            detector = Flat1dDetector(dpart, axis=det_axis_init,
                                      check_bounds=check_bounds)
        else:
            detector = CircularDetector(dpart,
                                        radius=det_curvature_radius,
                                        axis=det_axis_init,
                                        check_bounds=check_bounds)
        translation = kwargs.pop('translation', None)
        super(FanBeamGeometry, self).__init__(
            ndim=2, motion_part=apart, detector=detector,
            translation=translation, **kwargs)

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

    @classmethod
    def frommatrix(cls, apart, dpart, src_radius, det_radius, init_matrix,
                   det_curvature_radius=None, **kwargs):
        """Create an instance of `FanBeamGeometry` using a matrix.

        This alternative constructor uses a matrix to rotate and
        translate the default configuration. It is most useful when
        the transformation to be applied is already given as a matrix.

        Parameters
        ----------
        apart : 1-dim. `RectPartition`
            Partition of the angle interval.
        dpart : 1-dim. `RectPartition`
            Partition of the detector parameter interval.
        src_radius : nonnegative float
            Radius of the source circle.
        det_radius : nonnegative float
            Radius of the detector circle. Must be nonzero if ``src_radius``
            is zero.
        init_matrix : `array_like`, shape ``(2, 2)`` or ``(2, 3)``, optional
            Transformation matrix whose left ``(2, 2)`` block is multiplied
            with the default ``det_pos_init`` and ``det_axis_init`` to
            determine the new vectors. If present, the third column acts
            as a translation after the initial transformation.
            The resulting ``det_axis_init`` will be normalized.
        det_curvature_radius : nonnegative float, optional
            Radius of the detector curvature.
            If ``None``, flat detector is used, otherwise must be positive.
        kwargs :
            Further keyword arguments passed to the class constructor.

        Returns
        -------
        geometry : `FanBeamGeometry`

        Examples
        --------
        Mirror the second unit vector, creating a left-handed system:

        >>> apart = odl.uniform_partition(0, np.pi, 10)
        >>> dpart = odl.uniform_partition(-1, 1, 20)
        >>> matrix = np.array([[1, 0],
        ...                    [0, -1]])
        >>> geom = FanBeamGeometry.frommatrix(
        ...     apart, dpart, src_radius=1, det_radius=5, init_matrix=matrix)
        >>> geom.det_refpoint(0)
        array([ 0., -5.])
        >>> geom.det_axis_init
        array([ 1.,  0.])
        >>> geom.translation
        array([ 0.,  0.])

        Adding a translation with a third matrix column:

        >>> matrix = np.array([[1, 0, 1],
        ...                    [0, -1, 1]])
        >>> geom = FanBeamGeometry.frommatrix(
        ...     apart, dpart, src_radius=1, det_radius=5, init_matrix=matrix)
        >>> geom.translation
        array([ 1.,  1.])
        >>> geom.det_refpoint(0)  # (0, -5) + (1, 1)
        array([ 1., -4.])
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
        default_src_to_det_init = cls._default_config['src_to_det_init']
        default_det_axis_init = cls._default_config['det_axis_init']
        vecs_to_transform = [default_det_axis_init]
        transformed_vecs = transform_system(
            default_src_to_det_init, None, vecs_to_transform,
            matrix=trafo_matrix)

        # Use the standard constructor with these vectors
        src_to_det, det_axis = transformed_vecs
        if translation.size != 0:
            kwargs['translation'] = translation

        return cls(apart, dpart, src_radius, det_radius, det_curvature_radius,
                   src_to_det, det_axis_init=det_axis, **kwargs)

    @property
    def src_radius(self):
        """Source circle radius of this geometry."""
        return self.__src_radius

    @property
    def det_radius(self):
        """Detector circle radius of this geometry."""
        return self.__det_radius

    @property
    def det_curvature_radius(self):
        """Detector curve radius of this geometry."""
        return getattr(self.detector, 'radius', None)

    @property
    def src_to_det_init(self):
        """Initial source-to-detector unit vector."""
        return self.__src_to_det_init

    @property
    def det_axis_init(self):
        """Detector axis at angle 0."""
        return self.detector.axis

    def det_axis(self, angle):
        """Return the detector axis at ``angle``."""
        return self.rotation_matrix(angle).dot(self.det_axis_init)

    @property
    def angles(self):
        """Discrete angles given in this geometry."""
        return self.motion_grid.coord_vectors[0]

    def src_position(self, angle):
        """Return the source position at ``angle``.

        For an angle ``phi``, the source position is given by ::

            src(phi) = translation +
                       rot_matrix(phi) * (-src_rad * src_to_det_init)

        where ``src_to_det_init`` is the initial unit vector pointing
        from source to detector.

        Parameters
        ----------
        angle : float or `array-like`
            Angle(s) in radians describing the counter-clockwise
            rotation of source and detector.

        Returns
        -------
        pos : `numpy.ndarray`
            Vector(s) pointing from the origin to the source.
            If ``angle`` is a single parameter, the returned array has
            shape ``(2,)``, otherwise ``angle.shape + (2,)``.

        See Also
        --------
        det_refpoint

        Examples
        --------
        With default arguments, the source starts at ``src_rad * (-e_y)``
        and rotates to ``src_rad * e_x`` at 90 degrees:

        >>> apart = odl.uniform_partition(0, 2 * np.pi, 10)
        >>> dpart = odl.uniform_partition(-1, 1, 20)
        >>> geom = FanBeamGeometry(apart, dpart, src_radius=2, det_radius=5)
        >>> geom.src_position(0)
        array([ 0., -2.])
        >>> np.allclose(geom.src_position(np.pi / 2), [2, 0])
        True

        The method is vectorized, i.e., it can be called with multiple
        angles at once:

        >>> points = geom.src_position([0, np.pi / 2])
        >>> np.allclose(points[0], [0, -2])
        True
        >>> np.allclose(points[1], [2, 0])
        True
        """
        squeeze_out = (np.shape(angle) == ())
        angle = np.array(angle, dtype=float, copy=False, ndmin=1)

        # Initial vector from the rotation center to the source. It can be
        # computed this way since source and detector are at maximum distance,
        # i.e. the connecting line passes the origin.
        center_to_src_init = -self.src_radius * self.src_to_det_init
        pos_vec = (self.translation[None, :]
                   + self.rotation_matrix(angle).dot(center_to_src_init))
        if squeeze_out:
            pos_vec = pos_vec.squeeze()

        return pos_vec

    def det_refpoint(self, angle):
        """Return the detector reference point position at ``angle``.

        For an angle ``phi``, the detector position is given by ::

            det_ref(phi) = translation +
                           rot_matrix(phi) * (det_rad * src_to_det_init)

        where ``src_to_det_init`` is the initial unit vector pointing
        from source to detector.

        Parameters
        ----------
        angle : float or `array-like`
            Angle(s) in radians describing the counter-clockwise
            rotation of source and detector.

        Returns
        -------
        point : `numpy.ndarray`
            Vector(s) pointing from the origin to the detector reference
            point. If ``angle`` is a single parameter, the returned array
            has shape ``(2,)``, otherwise ``angle.shape + (2,)``.

        See Also
        --------
        src_position

        Examples
        --------
        With default arguments, the detector starts at ``det_rad * e_y``
        and rotates to ``det_rad * (-e_x)`` at 90 degrees:

        >>> apart = odl.uniform_partition(0, 2 * np.pi, 10)
        >>> dpart = odl.uniform_partition(-1, 1, 20)
        >>> geom = FanBeamGeometry(apart, dpart, src_radius=2, det_radius=5)
        >>> geom.det_refpoint(0)
        array([ 0.,  5.])
        >>> np.allclose(geom.det_refpoint(np.pi / 2), [-5, 0])
        True

        The method is vectorized, i.e., it can be called with multiple
        angles at once:

        >>> points = geom.det_refpoint([0, np.pi / 2])
        >>> np.allclose(points[0], [0, 5])
        True
        >>> np.allclose(points[1], [-5, 0])
        True
        """
        squeeze_out = (np.shape(angle) == ())
        angle = np.array(angle, dtype=float, copy=False, ndmin=1)

        # Initial vector from the rotation center to the detector. It can be
        # computed this way since source and detector are at maximum distance,
        # i.e. the connecting line passes the origin.
        center_to_det_init = self.det_radius * self.src_to_det_init
        refpt = (self.translation[None, :]
                 + self.rotation_matrix(angle).dot(center_to_det_init))
        if squeeze_out:
            refpt = refpt.squeeze()

        return refpt

    def rotation_matrix(self, angle):
        """Return the rotation matrix for ``angle``.

        For an angle ``phi``, the matrix is given by ::

            rot(phi) = [[cos(phi), -sin(phi)],
                        [sin(phi), cos(phi)]]

        Parameters
        ----------
        angle : float or `array-like`
            Angle(s) in radians describing the counter-clockwise
            rotation of source and detector.

        Returns
        -------
        rot : `numpy.ndarray`
            The rotation matrix (or matrices) mapping vectors at the
            initial state to the ones in the state defined by ``angle``.
            The rotation is extrinsic, i.e., defined in the "world"
            coordinate system.
            If ``angle`` is a single parameter, the returned array has
            shape ``(2, 2)``, otherwise ``angle.shape + (2, 2)``.
        """
        squeeze_out = (np.shape(angle) == ())
        angle = np.array(angle, dtype=float, copy=False, ndmin=1)
        if (self.check_bounds
                and not is_inside_bounds(angle, self.motion_params)):
            raise ValueError('`angle` {} not in the valid range {}'
                             ''.format(angle, self.motion_params))

        matrix = euler_matrix(angle)
        if squeeze_out:
            matrix = matrix.squeeze()

        return matrix

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.motion_partition, self.det_partition]
        optargs = [('src_radius', self.src_radius, -1),
                   ('det_radius', self.det_radius, -1)]

        if not np.allclose(self.src_to_det_init,
                           self._default_config['src_to_det_init']):
            optargs.append(
                ['src_to_det_init', array_str(self.src_to_det_init), ''])

        if self._det_axis_init_arg is not None:
            optargs.append(
                ['det_axis_init', array_str(self._det_axis_init_arg), ''])

        if not np.array_equal(self.translation, (0, 0)):
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
        >>> dpart = odl.uniform_partition(-1, 1, 20)
        >>> geom = odl.tomo.FanBeamGeometry(apart, dpart, 50, 100)

        Extract sub-geometry with every second angle:

        >>> geom[::2, :]
        FanBeamGeometry(
            nonuniform_partition(
                [ 0.5,  2.5],
                min_pt=0.0, max_pt=4.0
            ),
            uniform_partition(-1.0, 1.0, 20),
            src_radius=50.0,
            det_radius=100.0
        )
        """
        part = self.partition[indices]
        apart = part.byaxis[0]
        dpart = part.byaxis[1]

        return FanBeamGeometry(apart, dpart,
                               src_radius=self.src_radius,
                               det_radius=self.det_radius,
                               det_curvature_radius=self.det_curvature_radius,
                               src_to_det_init=self.src_to_det_init,
                               det_axis_init=self._det_axis_init_arg,
                               translation=self.translation)


class ConeBeamGeometry(DivergentBeamGeometry, AxisOrientedGeometry):

    """Cone beam geometry with circular/helical source curve.

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

    For details, check `the online docs
    <https://odlgroup.github.io/odl/guide/geometry_guide.html>`_.
    """

    _default_config = dict(axis=(0, 0, 1),
                           src_to_det_init=(0, 1, 0),
                           det_axes_init=((1, 0, 0), (0, 0, 1)))

    def __init__(self, apart, dpart, src_radius, det_radius,
                 det_curvature_radius=None, pitch=0, axis=(0, 0, 1), **kwargs):
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
            Radius of the detector circle. Must be nonzero if ``src_radius``
            is zero.
        det_curvature_radius :  2-tuple of nonnegative floats, optional
            Radius or radii of the detector curvature.
            If ``None``, a flat detector is used.
            If ``(r, None)`` or ``(r, float('inf'))``, a cylindrical
            detector is used.
            If ``(r1, r2)``, a spherical detector is used.
        pitch : float, optional
            Constant distance along ``axis`` that a point on the helix
            traverses when increasing the angle parameter by ``2 * pi``.
            The default case ``pitch=0`` results in a circular cone
            beam geometry.
        axis : `array-like`, shape ``(3,)``, optional
            Vector defining the fixed rotation axis of this geometry.

        Other Parameters
        ----------------
        offset_along_axis : float, optional
            Scalar offset along the ``axis`` at ``angle=0``, i.e., the
            translation along the axis at angle 0 is
            ``offset_along_axis * axis``.
            Default: 0.
        src_to_det_init : `array-like`, shape ``(3,)``, optional
            Initial state of the vector pointing from source to detector
            reference point. The zero vector is not allowed.
            The default depends on ``axis``, see Notes.
        det_axes_init : length-2-sequence of `array-like`'s, optional
            Initial axes defining the detector orientation, provided as
            arrays with shape ``(3,)``.
            Default: depends on ``axis``, see Notes.
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

        >>> apart = odl.uniform_partition(0, 4 * np.pi, 10)
        >>> dpart = odl.uniform_partition([-1, -1], [1, 1], (20, 20))
        >>> geom = ConeBeamGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10, pitch=2)
        >>> geom.src_position(0)
        array([ 0., -5.,  0.])
        >>> geom.det_refpoint(0)
        array([ 0., 10.,  0.])
        >>> np.allclose(geom.src_position(2 * np.pi),
        ...             geom.src_position(0) + (0, 0, 2))  # z shift by pitch
        True

        Checking the default orientation:

        >>> geom.axis
        array([ 0.,  0.,  1.])
        >>> geom.src_to_det_init
        array([ 0.,  1.,  0.])
        >>> geom.det_axes_init
        array([[ 1.,  0.,  0.],
               [ 0.,  0.,  1.]])

        Specifying curvature of the cylindrical detector:

        >>> apart = odl.uniform_partition(0, 4 * np.pi, 10)
        >>> dpart = odl.uniform_partition(
        ...     [-np.pi / 2, -1], [np.pi / 2, 1], (20, 20))
        >>> geom = ConeBeamGeometry(apart, dpart,
        ...     src_radius=5, det_radius=10, det_curvature_radius=(10, None))
        >>> # (10*sin(pi/2), 10*cos(pi/2), 1)
        >>> np.round(geom.det_point_position(0, [ np.pi / 2, 1] ), 2)
        array([ 10., 0., 1.])

        Specifying curvature of the spherical detector:

        >>> apart = odl.uniform_partition(0, 4 * np.pi, 10)
        >>> dpart = odl.uniform_partition([-np.pi / 2, -np.pi / 4],
        ...                               [ np.pi / 2,  np.pi / 4], (20, 20))
        >>> geom = ConeBeamGeometry(apart, dpart,
        ...     src_radius=5, det_radius=10, det_curvature_radius=(10, 10))
        >>> # 10*( cos(pi/4), 0, sin(pi/4))
        >>> np.round(geom.det_point_position(0, [ np.pi / 2, np.pi / 4] ), 2)
        array([ 7.07, 0.  , 7.07])

        Specifying an axis by default rotates the standard configuration
        to this position:

        >>> e_x, e_y, e_z = np.eye(3)  # standard unit vectors
        >>> geom = ConeBeamGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10, pitch=2,
        ...     axis=(0, 1, 0))
        >>> np.allclose(geom.axis, e_y)
        True
        >>> np.allclose(geom.src_to_det_init, -e_z)
        True
        >>> np.allclose(geom.det_axes_init, (e_x, e_y))
        True
        >>> geom = ConeBeamGeometry(
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

        >>> geom = ConeBeamGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10, pitch=2,
        ...     src_to_det_init=(-1, 0, 0),
        ...     det_axes_init=((0, 1, 0), (0, 0, 1)))
        >>> np.allclose(geom.axis, e_z)
        True
        >>> np.allclose(geom.src_to_det_init, -e_x)
        True
        >>> np.allclose(geom.det_axes_init, (e_y, e_z))
        True
        """
        default_axis = self._default_config['axis']
        default_src_to_det_init = self._default_config['src_to_det_init']
        default_det_axes_init = self._default_config['det_axes_init']

        # Handle initial coordinate system. We need to assign `None` to
        # the vectors first since we want to check that `init_matrix`
        # is not used together with those other parameters.
        src_to_det_init = kwargs.pop('src_to_det_init', None)
        det_axes_init = kwargs.pop('det_axes_init', None)

        # Store some stuff for repr
        if src_to_det_init is not None:
            self._src_to_det_init_arg = np.asarray(src_to_det_init,
                                                   dtype=float)
        else:
            self._src_to_det_init_arg = None

        if det_axes_init is not None:
            self._det_axes_init_arg = tuple(
                np.asarray(a, dtype=float) for a in det_axes_init)
        else:
            self._det_axes_init_arg = None

        # Compute the transformed system and the transition matrix. We
        # transform only those vectors that were not explicitly given.
        vecs_to_transform = []
        if src_to_det_init is None:
            vecs_to_transform.append(default_src_to_det_init)
        if det_axes_init is None:
            vecs_to_transform.extend(default_det_axes_init)

        transformed_vecs = transform_system(
            axis, default_axis, vecs_to_transform)
        transformed_vecs = list(transformed_vecs)

        axis = transformed_vecs.pop(0)
        if src_to_det_init is None:
            src_to_det_init = transformed_vecs.pop(0)
        if det_axes_init is None:
            det_axes_init = (transformed_vecs.pop(0), transformed_vecs.pop(0))
        assert transformed_vecs == []

        # Check and normalize `src_to_det_init`. Detector axes are
        # normalized in the detector class.
        if np.linalg.norm(src_to_det_init) == 0:
            raise ValueError('`src_to_det_init` cannot be zero')
        else:
            src_to_det_init /= np.linalg.norm(src_to_det_init)

        # Get stuff out of kwargs, otherwise upstream code complains
        # about unknown parameters (rightly so)
        self.__pitch = float(pitch)
        self.__offset_along_axis = float(kwargs.pop('offset_along_axis', 0))
        self.__src_radius = float(src_radius)

        # Initialize stuff
        self.__src_to_det_init = src_to_det_init
        AxisOrientedGeometry.__init__(self, axis)
        # `check_bounds` is needed for both detector and geometry
        check_bounds = kwargs.get('check_bounds', True)
        if det_curvature_radius is None:
            detector = Flat2dDetector(dpart, axes=det_axes_init,
                                      check_bounds=check_bounds)
        elif len(det_curvature_radius) == 2:
            if det_curvature_radius[0] == det_curvature_radius[1]:
                detector = SphericalDetector(dpart,
                                             radius=det_curvature_radius[0],
                                             axes=det_axes_init,
                                             check_bounds=check_bounds)
            elif (det_curvature_radius[1] is None
                  or det_curvature_radius[1] == float('inf')):
                detector = CylindricalDetector(dpart,
                                               radius=det_curvature_radius[0],
                                               axes=det_axes_init,
                                               check_bounds=check_bounds)
            else:
                raise NotImplementedError('Curved detector with different '
                                          'curvature radii')
        else:
            raise ValueError('det_curvature_radius {} must be a 2-tuple'
                             ''.format(det_curvature_radius))
        super(ConeBeamGeometry, self).__init__(
            ndim=3, motion_part=apart, detector=detector, **kwargs)

        # Check parameters
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

    @classmethod
    def frommatrix(cls, apart, dpart, src_radius, det_radius, init_matrix,
                   det_curvature_radius=None, pitch=0, **kwargs):
        """Create an instance of `ConeBeamGeometry` using a matrix.

        This alternative constructor uses a matrix to rotate and
        translate the default configuration. It is most useful when
        the transformation to be applied is already given as a matrix.

        Parameters
        ----------
        apart : 1-dim. `RectPartition`
            Partition of the parameter interval.
        dpart : 2-dim. `RectPartition`
            Partition of the detector parameter set.
        src_radius : nonnegative float
            Radius of the source circle.
        det_radius : nonnegative float
            Radius of the detector circle. Must be nonzero if ``src_radius``
            is zero.
        init_matrix : `array_like`, shape ``(3, 3)`` or ``(3, 4)``, optional
            Transformation matrix whose left ``(3, 3)`` block is multiplied
            with the default ``det_pos_init`` and ``det_axes_init`` to
            determine the new vectors. If present, the fourth column acts
            as a translation after the initial transformation.
            The resulting ``det_axes_init`` will be normalized.
        det_curvature_radius :  2-tuple of nonnegative floats, optional
            Radius or radii of the detector curvature.
            If ``None``, a flat detector is used.
            If ``(r, None)`` or ``(r, float('inf'))``, a cylindrical
            detector is used.
            If ``(r1, r2)``, a spherical detector is used.
        pitch : float, optional
            Constant distance along the rotation axis that a point on the
            helix traverses when increasing the angle parameter by
            ``2 * pi``. The default case ``pitch=0`` results in a circular
            cone beam geometry.
        kwargs :
            Further keyword arguments passed to the class constructor.

        Returns
        -------
        geometry : `ConeBeamGeometry`

        Examples
        --------
        Map unit vectors ``e_y -> e_z`` and ``e_z -> -e_y``, keeping the
        right-handedness:

        >>> apart = odl.uniform_partition(0, 2 * np.pi, 10)
        >>> dpart = odl.uniform_partition([-1, -1], [1, 1], (20, 20))
        >>> matrix = np.array([[1, 0, 0],
        ...                    [0, 0, -1],
        ...                    [0, 1, 0]])
        >>> geom = ConeBeamGeometry.frommatrix(
        ...     apart, dpart, src_radius=5, det_radius=10, pitch=2,
        ...     init_matrix=matrix)
        >>> geom.axis
        array([ 0., -1.,  0.])
        >>> geom.src_to_det_init
        array([ 0.,  0.,  1.])
        >>> geom.det_axes_init
        array([[ 1.,  0.,  0.],
               [ 0., -1.,  0.]])

        Adding a translation with a fourth matrix column:

        >>> matrix = np.array([[0, 0, -1, 0],
        ...                    [0, 1, 0, 1],
        ...                    [1, 0, 0, 1]])
        >>> geom = ConeBeamGeometry.frommatrix(
        ...     apart, dpart, src_radius=5, det_radius=10, pitch=2,
        ...     init_matrix=matrix)
        >>> geom.translation
        array([ 0.,  1.,  1.])
        >>> geom.det_refpoint(0)  # (0, 10, 0) + (0, 1, 1)
        array([  0.,  11.,   1.])
        """
        for key in ('axis', 'src_to_det_init', 'det_axes_init', 'translation'):
            if key in kwargs:
                raise TypeError('got unknown keyword argument {!r}'
                                ''.format(key))

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
        default_src_to_det_init = cls._default_config['src_to_det_init']
        default_det_axes_init = cls._default_config['det_axes_init']
        vecs_to_transform = (default_src_to_det_init,) + default_det_axes_init
        transformed_vecs = transform_system(
            default_axis, None, vecs_to_transform, matrix=trafo_matrix)

        # Use the standard constructor with these vectors
        axis, src_to_det, det_axis_0, det_axis_1 = transformed_vecs
        if translation.size == 0:
            pass
        else:
            kwargs['translation'] = translation

        return cls(apart, dpart, src_radius, det_radius,
                   det_curvature_radius=det_curvature_radius,
                   pitch=pitch,
                   axis=axis,
                   src_to_det_init=src_to_det,
                   det_axes_init=[det_axis_0, det_axis_1],
                   **kwargs)

    @property
    def src_radius(self):
        """Source circle radius of this geometry."""
        return self.__src_radius

    @property
    def det_radius(self):
        """Detector circle radius of this geometry."""
        return self.__det_radius

    @property
    def det_curvature_radius(self):
        """Detector curve radius of this geometry."""
        return getattr(self.detector, 'radius', None)

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

    @property
    def offset_along_axis(self):
        """Scalar offset along ``axis`` at ``angle=0``."""
        return self.__offset_along_axis

    @property
    def angles(self):
        """Discrete angles given in this geometry."""
        return self.motion_grid.coord_vectors[0]

    def det_axes(self, angle):
        """Return the detector axes tuple at ``angle``.

        Parameters
        ----------
        angles : float or `array-like`
            Angle(s) in radians describing the counter-clockwise rotation
            of the detector around `axis`.

        Returns
        -------
        axes : `numpy.ndarray`
            Unit vectors along which the detector is aligned.
            If ``angle`` is a single parameter, the returned array has
            shape ``(2, 3)``, otherwise
            ``broadcast(*angles).shape + (2, 3)``.

        Notes
        -----
        To get an array that enumerates the detector axes in the first
        dimension, move the second-to-last axis to the first position:

            axes = det_axes(angle)
            axes_enumeration = np.moveaxis(deriv, -2, 0)
        """
        # Transpose to take dot along axis 1
        axes = self.rotation_matrix(angle).dot(self.det_axes_init.T)
        # `axes` has shape (a, 3, 2), need to roll the last dimensions
        # to the second-to-last place
        return np.rollaxis(axes, -1, -2)

    def det_refpoint(self, angle):
        """Return the detector reference point position at ``angle``.

        For an angle ``phi``, the detector position is given by ::

            det_ref(phi) = translation +
                           rot_matrix(phi) * (det_rad * src_to_det_init) +
                           (offset_along_axis + pitch * phi) * axis

        where ``src_to_det_init`` is the initial unit vector pointing
        from source to detector.

        Parameters
        ----------
        angle : float or `array-like`
            Angle(s) in radians describing the counter-clockwise
            rotation of the detector.

        Returns
        -------
        refpt : `numpy.ndarray`
            Vector(s) pointing from the origin to the detector reference
            point. If ``angle`` is a single parameter, the returned array
            has shape ``(3,)``, otherwise ``angle.shape + (3,)``.

        See Also
        --------
        src_position

        Examples
        --------
        With default arguments, the detector starts at ``det_rad * e_y``
        and rotates to ``det_rad * (-e_x) + pitch/4 * e_z`` at
        90 degrees:

        >>> apart = odl.uniform_partition(0, 4 * np.pi, 10)
        >>> dpart = odl.uniform_partition([-1, -1], [1, 1], (20, 20))
        >>> geom = ConeBeamGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10, pitch=2)
        >>> geom.det_refpoint(0)
        array([  0.,  10.,   0.])
        >>> np.allclose(geom.det_refpoint(np.pi / 2), [-10, 0, 0.5])
        True

        The method is vectorized, i.e., it can be called with multiple
        angles at once (or an n-dimensional array of angles):

        >>> points = geom.det_refpoint([0, np.pi / 2])
        >>> np.allclose(points[0], [0, 10, 0])
        True
        >>> np.allclose(points[1], [-10, 0, 0.5])
        True
        >>> geom.det_refpoint(np.zeros((4, 5))).shape
        (4, 5, 3)
        """
        squeeze_out = (np.shape(angle) == ())
        angle = np.array(angle, dtype=float, copy=False, ndmin=1)
        rot_matrix = self.rotation_matrix(angle)
        extra_dims = angle.ndim

        # Initial vector from center of rotation to detector.
        # It can be computed this way since source and detector are at
        # maximum distance, i.e. the connecting line passes the center.
        center_to_det_init = self.det_radius * self.src_to_det_init
        # `circle_component` has shape (a, ndim)
        circle_component = rot_matrix.dot(center_to_det_init)

        # Increment along the rotation axis according to pitch and
        # offset_along_axis
        # `shift_along_axis` has shape angles.shape
        shift_along_axis = (self.offset_along_axis
                            + self.pitch * angle / (2 * np.pi))
        # Create outer product of `shift_along_axis` and `axis`, resulting
        # in shape (a, ndim)
        pitch_component = np.multiply.outer(shift_along_axis, self.axis)

        # Broadcast translation along extra dimensions
        transl_slc = (None,) * extra_dims + (slice(None),)
        refpt = (self.translation[transl_slc]
                 + circle_component
                 + pitch_component)
        if squeeze_out:
            refpt = refpt.squeeze()

        return refpt

    def src_position(self, angle):
        """Return the source position at ``angle``.

        For an angle ``phi``, the source position is given by ::

            src(phi) = translation +
                       rot_matrix(phi) * (-src_rad * src_to_det_init) +
                       (offset_along_axis + pitch * phi) * axis

        where ``src_to_det_init`` is the initial unit vector pointing
        from source to detector.

        Parameters
        ----------
        angle : float or `array-like`
            Angle(s) in radians describing the counter-clockwise
            rotation of the detector.

        Returns
        -------
        pos : `numpy.ndarray`, shape (3,) or (num_angles, 3)
            Vector(s) pointing from the origin to the source position.
            If ``angle`` is a single parameter, the returned array has
            shape ``(3,)``, otherwise ``angle.shape + (3,)``.

        See Also
        --------
        det_refpoint

        Examples
        --------
        With default arguments, the source starts at ``src_rad * (-e_y)``
        and rotates to ``src_rad * e_x + pitch/4 * e_z`` at
        90 degrees:

        >>> apart = odl.uniform_partition(0, 4 * np.pi, 10)
        >>> dpart = odl.uniform_partition([-1, -1], [1, 1], (20, 20))
        >>> geom = ConeBeamGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10, pitch=2)
        >>> geom.src_position(0)
        array([ 0., -5.,  0.])
        >>> np.allclose(geom.src_position(np.pi / 2), [5, 0, 0.5])
        True

        The method is vectorized, i.e., it can be called with multiple
        angles at once:

        >>> points = geom.src_position([0, np.pi / 2])
        >>> np.allclose(points[0], [0, -5, 0])
        True
        >>> np.allclose(points[1], [5, 0, 0.5])
        True
        >>> geom.src_position(np.zeros((4, 5))).shape
        (4, 5, 3)
        """
        squeeze_out = (np.shape(angle) == ())
        angle = np.array(angle, dtype=float, copy=False, ndmin=1)
        rot_matrix = self.rotation_matrix(angle)
        extra_dims = angle.ndim

        # Initial vector from center of rotation to source.
        # It can be computed this way since source and detector are at
        # maximum distance, i.e. the connecting line passes the center.
        center_to_src_init = -self.src_radius * self.src_to_det_init
        # `circle_component` has shape (a, ndim)
        circle_component = rot_matrix.dot(center_to_src_init)

        # Increment along the rotation axis according to pitch and
        # offset_along_axis
        # `shift_along_axis` has shape angles.shape
        shift_along_axis = (self.offset_along_axis
                            + self.pitch * angle / (2 * np.pi))
        # Create outer product of `shift_along_axis` and `axis`, resulting
        # in shape (a, ndim)
        pitch_component = np.multiply.outer(shift_along_axis, self.axis)

        # Broadcast translation along extra dimensions
        transl_slc = (None,) * extra_dims + (slice(None),)
        refpt = (self.translation[transl_slc]
                 + circle_component
                 + pitch_component)
        if squeeze_out:
            refpt = refpt.squeeze()

        return refpt

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.motion_partition, self.det_partition]
        optargs = [('src_radius', self.src_radius, -1),
                   ('det_radius', self.det_radius, -1),
                   ('pitch', self.pitch, 0)
                   ]

        if not np.allclose(self.axis, self._default_config['axis']):
            optargs.append(['axis', array_str(self.axis), ''])

        optargs.append(['offset_along_axis', self.offset_along_axis, 0])

        if self._src_to_det_init_arg is not None:
            optargs.append(['src_to_det_init',
                            array_str(self._src_to_det_init_arg),
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
        >>> geom = odl.tomo.ConeBeamGeometry(apart, dpart, 50, 100, pitch=2)

        Extract sub-geometry with every second angle:

        >>> geom[::2]
        ConeBeamGeometry(
            nonuniform_partition(
                [ 0.5,  2.5],
                min_pt=0.0, max_pt=4.0
            ),
            uniform_partition([-1., -1.], [ 1.,  1.], (20, 20)),
            src_radius=50.0,
            det_radius=100.0,
            pitch=2.0
        )
        """
        part = self.partition[indices]
        apart = part.byaxis[0]
        dpart = part.byaxis[1:]

        return ConeBeamGeometry(apart, dpart,
                                src_radius=self.src_radius,
                                det_radius=self.det_radius,
                                det_curvature_radius=self.det_curvature_radius,
                                pitch=self.pitch,
                                axis=self.axis,
                                offset_along_axis=self.offset_along_axis,
                                src_to_det_init=self._src_to_det_init_arg,
                                det_axes_init=self._det_axes_init_arg,
                                translation=self.translation)

    # Manually override the abstract method in `Geometry` since it's found
    # first
    rotation_matrix = AxisOrientedGeometry.rotation_matrix


def cone_beam_geometry(space, src_radius, det_radius, num_angles=None,
                       short_scan=False, det_shape=None):
    r"""Create a default fan or cone beam geometry from ``space``.

    This function is intended for simple test cases where users do not
    need the full flexibility of the geometries, but simply wants a
    geometry that works.

    The geometry returned by this function has equidistant angles
    that lie (strictly) between 0 and either ``2 * pi`` (full scan)
    or ``pi + fan_angle`` (short scan).
    The detector is centered around 0, and its size is chosen such that
    the whole ``space`` is covered with lines.

    The number of angles and detector elements is chosen such that
    the resulting sinogram is fully sampled according to the
    Nyquist criterion, which in general results in a very large number of
    samples. In particular, a ``space`` that is not centered at the origin
    can result in very large detectors since the latter is always
    origin-centered.

    Parameters
    ----------
    space : `DiscreteLp`
        Reconstruction space, the space of the volumetric data to be
        projected. Must be 2- or 3-dimensional.
    src_radius : nonnegative float
        Radius of the source circle. Must be larger than the radius of
        the smallest vertical cylinder containing ``space.domain``,
        i.e., the source must be outside the volume for all rotations.
    det_radius : nonnegative float
        Radius of the detector circle.
    short_scan : bool, optional
        Use the minimum required angular range ``[0, pi + fan_angle]``.
        For ``True``, the `parker_weighting` should be used in FBP.
        By default, the range ``[0, 2 * pi]`` is used.
    num_angles : int, optional
        Number of angles.
        Default: Enough to fully sample the data, see Notes.
    det_shape : int or sequence of ints, optional
        Number of detector pixels.
        Default: Enough to fully sample the data, see Notes.

    Returns
    -------
    geometry : `DivergentBeamGeometry`
        Projection geometry with equidistant angles and zero-centered
        detector as determined by sampling criteria.

            - If ``space`` is 2D, the result is a `FanBeamGeometry`.
            - If ``space`` is 3D, the result is a `ConeBeamGeometry`.

    Examples
    --------
    Create a fan beam geometry from a 2d space:

    >>> space = odl.uniform_discr([-1, -1], [1, 1], (20, 20))
    >>> geometry = cone_beam_geometry(space, src_radius=5, det_radius=5)
    >>> geometry.angles.size
    78
    >>> geometry.detector.size
    57

    For a short scan geometry (from 0 to ``pi + fan_angle``), the
    ``short_scan`` flag can be set, resulting in a smaller number of
    angles:

    >>> geometry = cone_beam_geometry(space, src_radius=5, det_radius=5,
    ...                               short_scan=True)
    >>> geometry.angles.size
    46

    If the source is close to the object, the detector becomes larger due
    to more magnification:

    >>> geometry = cone_beam_geometry(space, src_radius=3, det_radius=9)
    >>> geometry.angles.size
    80
    >>> geometry.detector.size
    105

    Notes
    -----
    According to [NW2001]_, pages 75--76, a function
    :math:`f : \mathbb{R}^2 \to \mathbb{R}` that has compact support

    .. math::
        \| x \| > \rho  \implies f(x) = 0,

    and is essentially bandlimited

    .. math::
       \| \xi \| > \Omega \implies \hat{f}(\xi) \approx 0,

    can be fully reconstructed from a fan beam ray transform with
    source-detector distance :math:`r` (assuming all detector
    points have the same distance to the source) if (1) the projection
    angles are sampled with a spacing of :math:`\Delta \psi` such that

    .. math::
        \Delta \psi \leq \frac{r + \rho}{r}\, \frac{\pi}{\rho \Omega},

    and (2) the detector is sampled with an angular interval
    :math:`\Delta \alpha` that satisfies

    .. math::
        \Delta \alpha \leq \frac{\pi}{r \Omega}.

    For a flat detector, the angular interval is smallest in the center
    of the fan and largest at the boundaries. The worst-case relation
    between the linear and angular sampling intervals are

    .. math::
        \Delta s = R \Delta \alpha, \quad R^2 = r^2 + (w / 2)^2,

    where :math:`w` is the width of the detector.
    Thus, to satisfy the angular detector condition one can choose

    .. math::
        \Delta s \leq \frac{\pi \sqrt{r^2 + (w / 2)^2}}{r \Omega}.

    The geometry returned by this function satisfies these conditions exactly.

    If the domain is 3-dimensional, a circular cone beam geometry is
    created with the third coordinate axis as rotation axis. This does,
    of course, not yield complete data, but is equivalent to the
    2D fan beam case in the :math:`z = 0` slice. The vertical size of
    the detector is chosen such that it covers the object vertically
    with rays, using a containing cuboid
    :math:`[-\rho, \rho]^2 \times [z_{\mathrm{min}}, z_{\mathrm{min}}]`
    to compute the cone angle.

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

    # Compute minimum width of the detector to cover the object. The relation
    # used here is (w/2)/(rs+rd) = rho/rs since both are equal to tan(alpha),
    # where alpha is the half fan angle.
    rs = float(src_radius)
    if (rs <= rho):
        raise ValueError('source too close to the object, resulting in '
                         'infinite detector for full coverage')
    rd = float(det_radius)
    r = src_radius + det_radius
    w = 2 * rho * (rs + rd) / rs

    # Compute minimum number of pixels given the constraint on the
    # sampling interval and the computed width
    rb = np.hypot(r, w / 2)  # length of the boundary ray to the flat detector
    num_px_horiz = 2 * int(np.ceil(w * omega * r / (2 * np.pi * rb))) + 1

    if space.ndim == 2:
        det_min_pt = -w / 2
        det_max_pt = w / 2
        if det_shape is None:
            det_shape = num_px_horiz
    elif space.ndim == 3:
        # Compute number of vertical pixels required to cover the object,
        # using the same sampling interval vertically as horizontally.
        # The reasoning is the same as for the computation of w.

        # Minimum distance of the containing cuboid edges to the source
        dist = rs - rho
        # Take angle of the rays going through the top and bottom corners
        # in that edge
        half_cone_angle = max(np.arctan(abs(space.partition.min_pt[2]) / dist),
                              np.arctan(abs(space.partition.max_pt[2]) / dist))
        h = 2 * np.sin(half_cone_angle) * (rs + rd)

        # Use the vertical spacing from the reco space, corrected for
        # magnification at the "back" of the object, i.e., where it is
        # minimal
        min_mag = (rs + rd) / (rs + rho)
        delta_h = min_mag * space.cell_sides[2]
        num_px_vert = int(np.ceil(h / delta_h))
        h = num_px_vert * delta_h  # make multiple of spacing

        det_min_pt = [-w / 2, -h / 2]
        det_max_pt = [w / 2, h / 2]
        if det_shape is None:
            det_shape = [num_px_horiz, num_px_vert]

    fan_angle = 2 * np.arctan(rho / rs)
    if short_scan:
        max_angle = min(np.pi + fan_angle, 2 * np.pi)
    else:
        max_angle = 2 * np.pi

    if num_angles is None:
        num_angles = int(np.ceil(max_angle * omega * rho / np.pi
                                 * r / (r + rho)))

    angle_partition = uniform_partition(0, max_angle, num_angles)
    det_partition = uniform_partition(det_min_pt, det_max_pt, det_shape)

    if space.ndim == 2:
        return FanBeamGeometry(angle_partition, det_partition,
                               src_radius, det_radius)
    elif space.ndim == 3:
        return ConeBeamGeometry(angle_partition, det_partition,
                                src_radius, det_radius)
    else:
        raise ValueError('``space.ndim`` must be 2 or 3.')


def helical_geometry(space, src_radius, det_radius, num_turns,
                     n_pi=1, num_angles=None, det_shape=None):
    """Create a default helical geometry from ``space``.

    This function is intended for simple test cases where users do not
    need the full flexibility of the geometries, but simply wants a
    geometry that works.

    The geometry returned by this function has equidistant angles
    that lie (strictly) between 0 and ``2 * pi * num_turns``.
    The detector is centered around 0, and its size is chosen such that
    the whole ``space`` is covered with lines.

    The number of angles and detector elements is chosen such that
    the resulting sinogram is fully sampled according to the
    Nyquist criterion, which in general results in a very large number of
    samples. In particular, a ``space`` that is not centered at the origin
    can result in very large detectors since the latter is always
    origin-centered.

    Parameters
    ----------
    space : `DiscreteLp`
        Reconstruction space, the space of the volumetric data to be
        projected. Must be 3-dimensional.
    src_radius : nonnegative float
        Radius of the source circle. Must be larger than the radius of
        the smallest vertical cylinder containing ``space.domain``,
        i.e., the source must be outside the volume for all rotations.
    det_radius : nonnegative float
        Radius of the detector circle.
    num_turns : positive float
        Total number of helical turns.
    num_angles : int, optional
        Number of angles.
        Default: Enough to fully sample the data, see Notes.
    n_pi : odd int, optional
        Total number of half rotations to include in the window. Values larger
        than 1 should be used if the pitch is much smaller than the detector
        height.
    det_shape : int or sequence of ints, optional
        Number of detector pixels.
        Default: Enough to fully sample the data, see Notes.

    Returns
    -------
    geometry : `ConeBeamGeometry`
        Projection geometry with equidistant angles and zero-centered
        detector as determined by sampling criteria.

    Examples
    --------
    Create a helical beam geometry from space:

    >>> space = odl.uniform_discr([-1, -1, -1], [1, 1, 1], (20, 20, 20))
    >>> geometry = helical_geometry(space, src_radius=5, det_radius=5,
    ...                             num_turns=3)
    >>> geometry.angles.size
    234
    >>> geometry.detector.shape
    (57, 9)

    Notes
    -----
    In the "fan beam direction", the sampling exactly follows the
    two-dimensional case see `cone_beam_geometry` for a description.

    In the "axial direction", e.g. along the [0, 0, 1] axis, the geometry is
    sampled according to two criteria. First, the bounds of the detector
    are chosen to satisfy the tuy condition.
    See `[TSS1998]`_ for a full description.

    Second, the sampling rate is selected according to the nyquist criterion
    to give a full sampling. This is done by sampling such that the pixel
    size is half of the size of the projection of the smallest voxel onto the
    detector.

    References
    ----------
    [TSS1998] Tam, K C, Samarasekera, S and Sauer, F.
    *Exact cone beam CT with a spiral scan*.
    Physics in Medicine & Biology 4 (1998), p 1015.

    .. _[TSS1998]: https://dx.doi.org/10.1088/0031-9155/43/4/028
    """
    # Find maximum distance from rotation axis
    corners = space.domain.corners()[:, :2]
    rho = np.max(np.linalg.norm(corners, axis=1))

    offset_along_axis = space.partition.min_pt[2]
    pitch = space.partition.extent[2] / num_turns

    # Find default values according to Nyquist criterion.

    # We assume that the function is bandlimited by a wave along the x or y
    # axis. The highest frequency we can measure is then a standing wave with
    # period of twice the inter-node distance.
    min_side = min(space.partition.cell_sides[:2])
    omega = np.pi / min_side

    # Compute minimum width of the detector to cover the object. The relation
    # used here is (w/2)/(rs+rd) = rho/rs since both are equal to tan(alpha),
    # where alpha is the half fan angle.
    rs = float(src_radius)
    if (rs <= rho):
        raise ValueError('source too close to the object, resulting in '
                         'infinite detector for full coverage')
    rd = float(det_radius)
    r = rs + rd
    w = 2 * rho * (rs + rd) / rs

    # Compute minimum number of pixels given the constraint on the
    # sampling interval and the computed width
    rb = np.hypot(r, w / 2)  # length of the boundary ray to the flat detector
    num_px_horiz = 2 * int(np.ceil(w * omega * r / (2 * np.pi * rb))) + 1

    # Compute lower and upper bound needed to fully sample the object.
    # In particular, since in a helical geometry several turns are used,
    # this is selected so that the field of view of two opposing projections,
    # separated by theta = 180 deg, overlap, but as little as possible.
    # See `tam_danielson_window` for more information.
    h_axis = (pitch / (2 * np.pi) *
              (1 + (-rho / src_radius) ** 2) *
              (n_pi * np.pi / 2.0 - np.arctan(-rho / src_radius)))
    h = 2 * h_axis * (rs + rd) / rs

    # Compute number of pixels
    min_mag = r / rs
    dh = 0.5 * space.partition.cell_sides[2] * min_mag
    num_px_vert = int(np.ceil(h / dh))

    det_min_pt = [-w / 2, -h / 2]
    det_max_pt = [w / 2, h / 2]
    if det_shape is None:
        det_shape = [num_px_horiz, num_px_vert]

    max_angle = 2 * np.pi * num_turns

    if num_angles is None:
        num_angles = int(np.ceil(max_angle * omega * rho / np.pi
                                 * r / (r + rho)))

    angle_partition = uniform_partition(0, max_angle, num_angles)
    det_partition = uniform_partition(det_min_pt, det_max_pt, det_shape)

    return ConeBeamGeometry(angle_partition, det_partition,
                            src_radius, det_radius,
                            offset_along_axis=offset_along_axis,
                            pitch=pitch)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
