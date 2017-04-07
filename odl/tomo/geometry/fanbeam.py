# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Fan beam geometries in 2 dimensions."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

from odl.tomo.geometry.detector import Flat1dDetector
from odl.tomo.geometry.geometry import DivergentBeamGeometry
from odl.tomo.util.utility import euler_matrix, rotation_matrix_from_to
from odl.util import signature_string, indent_rows


__all__ = ('FanFlatGeometry',)


class FanFlatGeometry(DivergentBeamGeometry):

    """Abstract 2d fan beam geometry with flat 1d detector.

    The source moves on a circle with radius ``src_radius``, and the
    detector reference point is opposite to the source, i.e. at maximum
    distance, on a circle with radius ``det_radius``. One of the two
    radii can be chosen as 0, which corresponds to a stationary source
    or detector, respectively.

    The motion parameter is the 1d rotation angle parameterizing source
    and detector positions simultaneously.

    In the standard configuration, the source and detector start on the
    first coodinate axis with vector ``(1, 0)`` from source to detector,
    and the initial detector axis is ``(0, 1)``.
    """

    _default_config = dict(src_to_det_init=(-1, 0), det_axis_init=(0, 1))

    def __init__(self, apart, dpart, src_radius, det_radius, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        apart : 1-dim. `RectPartition`
            Partition of the angle interval
        dpart : 1-dim. `RectPartition`
            Partition of the detector parameter interval
        src_radius : nonnegative float
            Radius of the source circle
        det_radius : nonnegative float
            Radius of the detector circle
        src_to_det_init : `array-like` (shape ``(2,)``), optional
            Initial state of the vector pointing from source to detector
            reference point. The zero vector is not allowed.
            Default: ``(-1, 0)``.
        det_axis_init : `array-like` (shape ``(2,)``), optional
            Initial axis defining the detector orientation. The default
            depends on ``src_to_det_init``, see Notes.
        extra_rot : `array_like`, shape ``(2, 2)``, optional
            Rotation matrix that should be applied at the end to the
            configuration of ``src_to_det_init`` and ``det_axis_init``.
            The rotation is extrinsic, i.e., defined in the "world"
            coordinate system.

        Notes
        -----
        In the default configuration, the initial source-to-detector vector
        is ``(1, 0)``, and the initial detector axis is ``(0, 1)``. If a
        different ``src_to_det_init`` is chosen, the new default axis is
        given as a rotation of the original one by a matrix that transforms
        ``(1, 0)`` to the new (normalized) ``src_to_det_init``. This matrix
        is calculated with the `rotation_matrix_from_to` function.
        Expressed in code, we have ::

            init_rot = rotation_matrix_from_to((1, 0), src_to_det_init)
            det_axis_init = init_rot.dot((0, 1))

        Examples
        --------
        Initialization with default parameters and some radii:

        >>> e_x, e_y = np.eye(2)  # standard unit vectors
        >>> apart = odl.uniform_partition(0, 2 * np.pi, 10)
        >>> dpart = odl.uniform_partition(-1, 1, 20)
        >>> geom = FanFlatGeometry(apart, dpart, src_radius=1, det_radius=5)
        >>> np.allclose(geom.src_to_det_init, -e_x)
        True
        >>> np.allclose(geom.det_axis_init, e_y)
        True

        Specifying an initial detector position by default rotates the
        standard configuration to this position:

        >>> geom = FanFlatGeometry(apart, dpart, src_radius=1, det_radius=5,
        ...                        src_to_det_init=(0, 1))
        >>> np.allclose(geom.src_to_det_init, e_y)
        True
        >>> np.allclose(geom.det_axis_init, e_x)
        True
        >>> geom = FanFlatGeometry(apart, dpart, src_radius=1, det_radius=5,
        ...                        src_to_det_init=(1, 0))
        >>> np.allclose(geom.src_to_det_init, e_x)
        True
        >>> np.allclose(geom.det_axis_init, -e_y)
        True

        The initial detector axis can also be set explicitly:

        >>> geom = FanFlatGeometry(
        ...     apart, dpart, src_radius=1, det_radius=5,
        ...     src_to_det_init=(0, 1), det_axis_init=(1, 0))
        >>> np.allclose(geom.src_to_det_init, e_y)
        True
        >>> np.allclose(geom.det_axis_init, e_x)
        True

        A matrix can be given to perform a final rotation. This is most
        useful to rotate non-standard ``det_axis_init``, or if full
        control over the rotation is desired:

        >>> rot_matrix = np.array([[-1, 0],
        ...                        [0, 1]])
        >>> geom = FanFlatGeometry(apart, dpart, src_radius=1, det_radius=5,
        ...                        extra_rot=rot_matrix)
        >>> np.allclose(geom.src_to_det_init, e_x)
        True
        >>> np.allclose(geom.det_axis_init, e_y)
        True
        """
        def_src_to_det = self._default_config['src_to_det_init']
        def_det_axis = self._default_config['det_axis_init']

        self.__src_radius, src_radius_in = float(src_radius), src_radius
        if self.src_radius < 0:
            raise ValueError('`src_radius` must be nonnegative, got {}'
                             ''.format(src_radius_in))
        self.__det_radius, det_radius_in = float(det_radius), det_radius
        if det_radius < 0:
            raise ValueError('`det_radius` must be nonnegative, got {}'
                             ''.format(det_radius_in))
        if self.src_radius == 0 and self.det_radius == 0:
            raise ValueError('source and detector radii cannot both be 0')

        src_to_det_init = np.asarray(kwargs.pop('src_to_det_init',
                                                def_src_to_det), dtype=float)
        if np.linalg.norm(src_to_det_init) <= 1e-6:
            raise ValueError('`src_to_det_init` {} too close '
                             'to zero'.format(src_to_det_init))

        det_axis_init = kwargs.pop('det_axis_init', None)
        extra_rot = np.asarray(kwargs.pop('extra_rot', np.eye(2)))
        if extra_rot.shape != (2, 2):
            raise ValueError('`extra_rot` must have shape (2, 2), got {}'
                             ''.format(extra_rot.shape))
        if abs(np.linalg.det(extra_rot)) < 1e-4:
            raise ValueError('`extra_rot` is almost singular')
        self.__extra_rotation = extra_rot

        if kwargs:
            raise TypeError('got an unexpected keyword argument {!r}'
                            ''.format(kwargs.popitem()[0]))

        if np.linalg.norm(src_to_det_init - def_src_to_det) < 1e-4:
            # Vector close to default is mapped to default (due to
            # instability otherwise)
            init_rot = np.eye(2)
        else:
            # Rotation due to non-standard src_to_det_init
            init_rot = rotation_matrix_from_to(def_src_to_det, src_to_det_init)

        if det_axis_init is None:
            det_axis_init = init_rot.dot(def_det_axis)

        # Extra rotation of everything
        src_to_det_init = self.extra_rotation.dot(src_to_det_init)
        det_axis_init = self.extra_rotation.dot(det_axis_init)

        self.__src_to_det_init = (src_to_det_init /
                                  np.linalg.norm(src_to_det_init))

        detector = Flat1dDetector(dpart, det_axis_init)
        super().__init__(ndim=2, motion_part=apart, detector=detector)

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

    @property
    def angles(self):
        """Discrete angles given in this geometry."""
        return self.motion_grid.coord_vectors[0]

    def src_position(self, angle):
        """Return the source position at ``angle``.

        For an angle ``phi``, the source position is given by::

            src(phi) = -src_rad * rot_matrix(phi) * src_to_det_init

        where ``src_to_det_init`` is the initial unit vector pointing
        from source to detector.

        Parameters
        ----------
        angle : float
            Rotation angle given in radians, must be contained in
            this geometry's `motion_params`

        Returns
        -------
        point : `numpy.ndarray`, shape ``(2,)``
            Source position corresponding to the given angle
        """
        if angle not in self.motion_params:
            raise ValueError('`angle` {} is not in the valid range {}'
                             ''.format(angle, self.motion_params))

        # Initial vector from 0 to the source. It can be computed this way
        # since source and detector are at maximum distance, i.e. the
        # connecting line passes the origin.
        origin_to_src_init = -self.src_radius * self.src_to_det_init
        return self.rotation_matrix(angle).dot(origin_to_src_init)

    def det_refpoint(self, angle):
        """Return the detector reference point position at ``angle``.

        For an angle ``phi``, the detector position is given by::

            ref(phi) = det_rad * rot_matrix(phi) * src_to_det_init

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
        """
        if angle not in self.motion_params:
            raise ValueError('`angle` {} is not in the valid range {}'
                             ''.format(angle, self.motion_params))

        # Initial vector from 0 to the detector. It can be computed this way
        # since source and detector are at maximum distance, i.e. the
        # connecting line passes the origin.
        origin_to_det_init = self.det_radius * self.src_to_det_init
        return self.rotation_matrix(angle).dot(origin_to_det_init)

    def rotation_matrix(self, angle):
        """Return the rotation matrix for ``angle``.

        For an angle ``phi``, the matrix is given by::

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
        angle = float(angle)
        if angle not in self.motion_params:
            raise ValueError('`angle` {} not in the valid range {}'
                             ''.format(angle, self.motion_params))
        return euler_matrix(angle)

    # TODO: back projection weighting function?

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.motion_partition, self.det_partition]
        optargs = [('src_radius', self.src_radius, -1),
                   ('det_radius', self.det_radius, -1)]

        if not np.allclose(self.extra_rotation, np.eye(self.ndim)):
            inv_rot = np.linalg.inv(self.extra_rotation)
            orig_src_to_det = inv_rot.dot(self.src_to_det_init)
            orig_det_axis = inv_rot.dot(self.det_axis_init)
        else:
            orig_src_to_det = self.src_to_det_init
            orig_det_axis = self.det_axis_init

        src_to_det_init = self._default_config['src_to_det_init']
        if not np.allclose(orig_src_to_det, src_to_det_init):
            optargs.append(('src_to_det_init', orig_src_to_det.tolist(), None))
            init_rot = rotation_matrix_from_to(src_to_det_init,
                                               orig_src_to_det)
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


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
