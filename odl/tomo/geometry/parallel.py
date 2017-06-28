# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Parallel beam geometries in 2 or 3 dimensions."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

from odl.discr import uniform_partition, nonuniform_partition
from odl.tomo.geometry.detector import Flat1dDetector, Flat2dDetector
from odl.tomo.geometry.geometry import Geometry, AxisOrientedGeometry
from odl.tomo.util import euler_matrix, transform_system
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
            Additional parameters passed on to the ``__init__`` method
            of `Geometry`.
        """
        super().__init__(ndim, apart, detector, **kwargs)

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
        if self.motion_partition.ndim == 1:
            return self.motion_grid.coord_vectors[0]
        else:
            return self.motion_grid.points()

    def det_refpoint(self, angle):
        """Return the position of the detector ref. point at ``angles``.

        The reference point is given by a rotation of the initial
        position by ``angles``.

        For an angle ``phi``, the detector position is given by ::

            det_ref(phi) = translation +
                           rot_matrix(phi) * (det_pos_init - translation)

        where ``det_pos_init`` is the detector reference point at initial
        state.

        Parameters
        ----------
        angle : float
            Parameter describing the detector rotation, must be
            contained in `motion_params`.

        Returns
        -------
        point : `numpy.ndarray`, shape (`ndim`,)
            The reference point for the given parameter.

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
        if angle not in self.motion_params:
            raise ValueError('`angle` {} not in the valid range {}'
                             ''.format(angle, self.motion_params))
        rot_part = self.rotation_matrix(angle).dot(
            self.det_pos_init - self.translation)
        return self.translation + rot_part

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
        det_axis_init : `array-like` (shape ``(2,)``), optional
            Initial axis defining the detector orientation. The default
            depends on ``det_pos_init``, see Notes.
        translation : `array-like`, shape ``(2,)``, optional
            Global translation of the geometry. This is added last in any
            method that computes an absolute vector, e.g., `det_refpoint`,
            and also shifts the center of rotation.

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
        # the detector class.
        detector = Flat1dDetector(part=dpart, axis=det_axis_init)
        super().__init__(ndim=2, apart=apart, detector=detector,
                         det_pos_init=det_pos_init, translation=translation)

        if self.motion_partition.ndim != 1:
            raise ValueError('`apart` dimension {}, expected 1'
                             ''.format(self.motion_partition.ndim))

        # Make sure there are no leftover kwargs
        if kwargs:
            raise TypeError('got unexpected keyword arguments {}'
                            ''.format(kwargs))

    @classmethod
    def frommatrix(cls, apart, dpart, init_matrix):
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

        Returns
        -------
        geometry : `Parallel2dGeometry`
            The resulting geometry.

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
        if translation.size == 0:
            kwargs = {}
        else:
            kwargs = {'translation': translation}

        return cls(apart, dpart, det_pos,
                   det_axis_init=det_axis, **kwargs)

    @property
    def det_axis_init(self):
        """Detector axis at angle 0."""
        return self.detector.axis

    def det_axis(self, angle):
        """Return the detector axis at ``angle``."""
        return self.rotation_matrix(angle).dot(self.det_axis_init)

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
            expressed in the fixed system.
        """
        if angle not in self.motion_params:
            raise ValueError('`angle` {} not in the valid range {}'
                             ''.format(angle, self.motion_params))
        return euler_matrix(angle)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.motion_partition, self.det_partition]
        optargs = []

        if not np.allclose(self.det_pos_init - self.translation,
                           self._default_config['det_pos_init']):
            optargs.append(
                ['det_pos_init', self.det_pos_init.tolist(), None])

        if self._det_axis_init_arg is not None:
            optargs.append(
                ['det_axis_init', self._det_axis_init_arg.tolist(), None])

        if not np.array_equal(self.translation, (0, 0)):
            optargs.append(
                ['translation', self.translation.tolist(), None])

        sig_str = signature_string(posargs, optargs, sep=',\n')
        return '{}(\n{}\n)'.format(self.__class__.__name__,
                                   indent_rows(sig_str))


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
        det_axes_init : 2-tuple of `array-like`'s (shape ``(3,)``), optional
            Initial axes defining the detector orientation. The default
            depends on ``det_pos_init``, see Notes.
        translation : `array-like`, shape ``(3,)``, optional
            Global translation of the geometry. This is added last in any
            method that computes an absolute vector, e.g., `det_refpoint`,
            and also shifts the center of rotation.

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

        # Initialize stuff. Normalization of the detector axes happens in
        # the detector class.
        detector = Flat2dDetector(part=dpart, axes=det_axes_init)
        super().__init__(ndim=3, apart=apart, detector=detector,
                         det_pos_init=det_pos_init, translation=translation)

        if self.motion_partition.ndim not in (2, 3):
            raise ValueError('`apart` has dimension {}, expected '
                             '2 or 3'.format(self.motion_partition.ndim))

        # Make sure there are no leftover kwargs
        if kwargs:
            raise TypeError('got unexpected keyword arguments {}'
                            ''.format(kwargs))

    @classmethod
    def frommatrix(cls, apart, dpart, init_matrix):
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

        Returns
        -------
        geometry : `Parallel3dEulerGeometry`
            The resulting geometry.

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
        (array([ 0.,  0.,  1.]), array([-1.,  0.,  0.]))

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
        if translation.size == 0:
            kwargs = {}
        else:
            kwargs = {'translation': translation}

        return cls(apart, dpart, det_pos,
                   det_axes_init=[det_axis_0, det_axis_1],
                   **kwargs)

    @property
    def det_axes_init(self):
        """Initial axes of the detector."""
        return self.detector.axes

    def det_axes(self, angles):
        """Return the detector axes tuple at ``angle``."""
        return tuple(self.rotation_matrix(angles).dot(axis)
                     for axis in self.det_axes_init)

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

        if not np.allclose(self.det_pos_init - self.translation,
                           self._default_config['det_pos_init']):
            optargs.append(
                ['det_pos_init', self.det_pos_init.tolist(), None])

        if self._det_axes_init_arg is not None:
            optargs.append(
                ['det_axes_init',
                 tuple(a.tolist() for a in self._det_axes_init_arg),
                 None])

        if not np.array_equal(self.translation, (0, 0, 0)):
            optargs.append(['translation', self.translation.tolist(), None])

        sig_str = signature_string(posargs, optargs, sep=',\n')
        return '{}(\n{}\n)'.format(self.__class__.__name__,
                                   indent_rows(sig_str))


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
        # the detector class.
        AxisOrientedGeometry.__init__(self, axis)
        detector = Flat2dDetector(dpart, det_axes_init)
        super().__init__(ndim=3, apart=apart, detector=detector,
                         det_pos_init=det_pos_init, translation=translation)

        if self.motion_partition.ndim != 1:
            raise ValueError('`apart` has dimension {}, expected 1'
                             ''.format(self.motion_partition.ndim))

        # Make sure there are no leftover kwargs
        if kwargs:
            raise TypeError('got unexpected keyword arguments {}'
                            ''.format(kwargs))

    @classmethod
    def frommatrix(cls, apart, dpart, init_matrix):
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

        Returns
        -------
        geometry : `Parallel3dAxisGeometry`
            The resulting geometry.

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
        (array([ 1.,  0.,  0.]), array([ 0., -1.,  0.]))

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
        if translation.size == 0:
            kwargs = {}
        else:
            kwargs = {'translation': translation}

        return cls(apart, dpart, axis,
                   det_pos_init=det_pos,
                   det_axes_init=[det_axis_0, det_axis_1],
                   **kwargs)

    @property
    def det_axes_init(self):
        """Initial axes of the detector."""
        return self.detector.axes

    def det_axes(self, angles):
        """Return the detector axes tuple at ``angle``."""
        return tuple(self.rotation_matrix(angles).dot(axis)
                     for axis in self.det_axes_init)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.motion_partition, self.det_partition]
        optargs = []

        if not np.allclose(self.axis, self._default_config['axis']):
            optargs.append(['axis', self.axis.tolist(), None])

        if self._det_pos_init_arg is not None:
            optargs.append(['det_pos_init',
                            self._det_pos_init_arg.tolist(),
                            None])

        if self._det_axes_init_arg is not None:
            optargs.append(
                ['det_axes_init',
                 tuple(a.tolist() for a in self._det_axes_init_arg),
                 None])

        if not np.array_equal(self.translation, (0, 0, 0)):
            optargs.append(['translation', self.translation.tolist(), None])

        sig_str = signature_string(posargs, optargs, sep=',\n')
        return '{}(\n{}\n)'.format(self.__class__.__name__,
                                   indent_rows(sig_str))

    # Manually override the abstract method in `Geometry` since it's found
    # first
    rotation_matrix = AxisOrientedGeometry.rotation_matrix


def parallel_beam_geometry(space, num_angles=None, det_shape=None):
    """Create default parallel beam geometry from ``space``.

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
    Create geometry from 2d space and check the number of data points:

    >>> space = odl.uniform_discr([-1, -1], [1, 1], (20, 20))
    >>> geometry = parallel_beam_geometry(space)
    >>> geometry.angles.size
    45
    >>> geometry.detector.size
    29

    Notes
    -----
    According to `Mathematical Methods in Image Reconstruction`_
    (pages 72--74), for a function
    :math:`f : \\mathbb{R}^2 \\to \\mathbb{R}` that has compact support

    .. math::
        \| x \| > \\rho  \implies f(x) = 0,

    and is essentially bandlimited

    .. math::
       \| \\xi \| > \\Omega \implies \\hat{f}(\\xi) \\approx 0,

    then, in order to fully reconstruct the function from a parallel beam ray
    transform, the function should be sampled at an angular interval
    :math:`\\Delta \psi` such that

    .. math::
        \\Delta \psi \leq \\frac{\\pi}{\\rho \\Omega},

    and the detector should be sampled with an interval :math:`\\Delta s`
    that satisfies

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
    num_px_horiz = int(np.ceil(2 * rho * omega / np.pi))

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

    angle_partition = nonuniform_partition(
        np.linspace(0, np.pi, num_angles, endpoint=False))

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
