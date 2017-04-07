# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Cone beam geometries in 3 dimensions."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

from odl.tomo.geometry.detector import Flat2dDetector
from odl.tomo.geometry.geometry import (
    DivergentBeamGeometry, AxisOrientedGeometry)
from odl.tomo.util.utility import perpendicular_vector, rotation_matrix_from_to
from odl.util import signature_string, indent_rows


__all__ = ('CircularConeFlatGeometry', 'HelicalConeFlatGeometry',)


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
    the initial source-to-detector vector is ``(-1, 0, 0)``, and the
    initial detector axes are ``[(0, 1, 0), (0, 0, 1)]``.

    See Also
    --------
    CircularConeFlatGeometry : Case with zero pitch
    """

    _default_config = dict(axis=(0, 0, 1),
                           src_to_det_init=(-1, 0, 0),
                           det_axes_init=((0, 1, 0), (0, 0, 1)))

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
        the initial source-to-detector direction is ``(-1, 0, 0)``,
        and the default detector axes are ``[(0, 1, 0), (0, 0, 1)]``.
        If a different ``axis`` is provided, the new default initial
        position and the new default axes are the computed by rotating
        the original ones by a matrix that transforms ``(0, 0, 1)`` to the
        new (normalized) ``axis``. This matrix is calculated with the
        `rotation_matrix_from_to` function. Expressed in code, we have ::

            init_rot = rotation_matrix_from_to((0, 0, 1), axis)
            src_to_det_init = init_rot.dot((-1, 0, 0))
            det_axes_init[0] = init_rot.dot((0, 1, 0))
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
        array([ 5.,  0.,  0.])
        >>> geom.det_refpoint(0)
        array([-10.,   0.,   0.])
        >>> np.allclose(geom.src_position(2 * np.pi),
        ...     geom.src_position(0) + (0, 0, 2))  # z shift due to pitch
        True
        >>> np.allclose(geom.axis, e_z)
        True
        >>> np.allclose(geom.src_to_det_init, -e_x)
        True
        >>> np.allclose(geom.det_axes_init, (e_y, e_z))
        True

        Specifying an axis by default rotates the standard configuration
        to this position:

        >>> geom = HelicalConeFlatGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10, pitch=2,
        ...     axis=(0, 1, 0))
        >>> np.allclose(geom.axis, e_y)
        True
        >>> np.allclose(geom.src_to_det_init, -e_x)
        True
        >>> np.allclose(geom.det_axes_init, (-e_z, e_y))
        True
        >>> geom = HelicalConeFlatGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10, pitch=2,
        ...     axis=(1, 0, 0))
        >>> np.allclose(geom.axis, e_x)
        True
        >>> np.allclose(geom.src_to_det_init, e_z)
        True
        >>> np.allclose(geom.det_axes_init, (e_y, e_x))
        True

        The initial source-to-detector vector and the detector axes can
        also be set explicitly:

        >>> geom = HelicalConeFlatGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10, pitch=2,
        ...     axis=(0, 1, 0), src_to_det_init=(1, 0, 0))
        >>> np.allclose(geom.axis, e_y)
        True
        >>> np.allclose(geom.src_to_det_init, e_x)
        True
        >>> np.allclose(geom.det_axes_init, (-e_z, e_y))  # as above
        True
        >>> geom = HelicalConeFlatGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10, pitch=2,
        ...     axis=(0, 1, 0), det_axes_init=((0, 1, 0), (0, 0, 1)))
        >>> np.allclose(geom.axis, e_y)
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
        >>> np.allclose(geom.src_to_det_init, -e_x)  # default
        True
        >>> np.allclose(geom.det_axes_init, (e_z, -e_y))
        True
        """
        def_axis = self._default_config['axis']
        def_src_to_det = self._default_config['src_to_det_init']
        def_det_axes = self._default_config['det_axes_init']

        src_to_det_init = kwargs.pop('src_to_det_init', None)
        det_axes_init = kwargs.pop('det_axes_init', None)
        extra_rot = np.asarray(kwargs.pop('extra_rot', np.eye(3)))
        if extra_rot.shape != (3, 3):
            raise ValueError('`extra_rot` must have shape (3, 3), got {}'
                             ''.format(extra_rot.shape))
        if abs(np.linalg.det(extra_rot)) < 1e-4:
            raise ValueError('`extra_rot` is almost singular')
        self.__extra_rotation = extra_rot

        if np.allclose(axis, def_axis, rtol=1e-3):
            # Vector close to default is mapped to default (due to
            # instability otherwise)
            init_rot = np.eye(3)
        else:
            # Rotation due to non-standard src_to_det_init
            init_rot = rotation_matrix_from_to(def_axis, axis)

        if src_to_det_init is None:
            src_to_det_init = init_rot.dot(def_src_to_det)
        if np.linalg.norm(src_to_det_init) <= 1e-10:
            raise ValueError('initial source to detector vector {} is too '
                             'close to zero'.format(src_to_det_init))

        if det_axes_init is None:
            det_axes_init = [init_rot.dot(a) for a in def_det_axes]

        # Extra rotation of everything
        src_to_det_init = self.extra_rotation.dot(src_to_det_init)
        det_axes_init = [self.extra_rotation.dot(a) for a in det_axes_init]

        axis = self.extra_rotation.dot(axis)
        AxisOrientedGeometry.__init__(self, axis)

        self.__src_to_det_init = (np.array(src_to_det_init) /
                                  np.linalg.norm(src_to_det_init))

        src_to_det_init = kwargs.pop('src_to_det_init',
                                     perpendicular_vector(self.axis))

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

        def_axis = self._default_config['axis']
        if not np.allclose(orig_axis, def_axis):
            optargs.append(('axis', orig_axis.tolist(), None))
            init_rot = rotation_matrix_from_to(def_axis, orig_axis)
            orig_src_to_det = init_rot.T.dot(orig_src_to_det)
            orig_det_axes = [init_rot.T.dot(a) for a in orig_det_axes]

        optargs.append(('pitch_offset', self.pitch_offset, 0))

        def_src_to_det = self._default_config['src_to_det_init']
        if not np.allclose(orig_src_to_det, def_src_to_det):
            optargs.append(('src_to_det_init', orig_src_to_det.tolist(), None))

        def_init_axes = self._default_config['det_axes_init']
        if not np.allclose(orig_det_axes, def_init_axes):
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
    the initial source-to-detector vector is ``(-1, 0, 0)``, and the
    initial detector axes are ``[(0, 1, 0), (0, 0, 1)]``.

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
        the initial source-to-detector direction is ``(-1, 0, 0)``,
        and the default detector axes are ``[(0, 1, 0), (0, 0, 1)]``.
        If a different ``axis`` is provided, the new default initial
        position and the new default axes are the computed by rotating
        the original ones by a matrix that transforms ``(0, 0, 1)`` to the
        new (normalized) ``axis``. This matrix is calculated with the
        `rotation_matrix_from_to` function. Expressed in code, we have ::

            init_rot = rotation_matrix_from_to((0, 0, 1), axis)
            src_to_det_init = init_rot.dot((-1, 0, 0))
            det_axes_init[0] = init_rot.dot((0, 1, 0))
            det_axes_init[1] = init_rot.dot((0, 0, 1))

        Examples
        --------
        Initialization with default parameters and some (arbitrary)
        choices for pitch and radii:

        >>> e_x, e_y, e_z = np.eye(3)  # standard unit vectors
        >>> apart = odl.uniform_partition(0, 2 * np.pi, 10)
        >>> dpart = odl.uniform_partition([-1, -1], [1, 1], (20, 20))
        >>> geom = CircularConeFlatGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10)
        >>> geom.src_position(0)
        array([ 5.,  0.,  0.])
        >>> geom.det_refpoint(0)
        array([-10.,   0.,   0.])
        >>> np.allclose(geom.src_position(2 * np.pi), geom.src_position(0))
        True
        >>> np.allclose(geom.axis, e_z)
        True
        >>> np.allclose(geom.src_to_det_init, -e_x)
        True
        >>> np.allclose(geom.det_axes_init, (e_y, e_z))
        True

        Specifying an axis by default rotates the standard configuration
        to this position:

        >>> geom = CircularConeFlatGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10,
        ...     axis=(0, 1, 0))
        >>> np.allclose(geom.axis, e_y)
        True
        >>> np.allclose(geom.src_to_det_init, -e_x)
        True
        >>> np.allclose(geom.det_axes_init, (-e_z, e_y))
        True
        >>> geom = CircularConeFlatGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10,
        ...     axis=(1, 0, 0))
        >>> np.allclose(geom.axis, e_x)
        True
        >>> np.allclose(geom.src_to_det_init, e_z)
        True
        >>> np.allclose(geom.det_axes_init, (e_y, e_x))
        True

        The initial source-to-detector vector and the detector axes can
        also be set explicitly:

        >>> geom = CircularConeFlatGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10,
        ...     axis=(0, 1, 0), src_to_det_init=(1, 0, 0))
        >>> np.allclose(geom.axis, e_y)
        True
        >>> np.allclose(geom.src_to_det_init, e_x)
        True
        >>> np.allclose(geom.det_axes_init, (-e_z, e_y))  # as above
        True
        >>> geom = CircularConeFlatGeometry(
        ...     apart, dpart, src_radius=5, det_radius=10,
        ...     axis=(0, 1, 0), det_axes_init=((0, 1, 0), (0, 0, 1)))
        >>> np.allclose(geom.axis, e_y)
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
        >>> np.allclose(geom.src_to_det_init, -e_x)  # default
        True
        >>> np.allclose(geom.det_axes_init, (e_z, -e_y))
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
