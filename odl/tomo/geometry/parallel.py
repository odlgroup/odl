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
from odl.tomo.util.utility import euler_matrix, perpendicular_vector


__all__ = ('ParallelGeometry', 'Parallel2dGeometry', 'Parallel3dEulerGeometry',
           'Parallel3dAxisGeometry', 'parallel_beam_geometry')


class ParallelGeometry(Geometry):

    """Abstract parallel beam geometry in 2 or 3 dimensions.

    Parallel geometries are characterized by a virtual source at
    infinity, such that a unit vector from a detector point towards
    the source (`det_to_src`) is independent of the location on the
    detector.
    """

    def __init__(self, ndim, apart, detector, det_init_pos):
        """Initialize a new instance.

        Parameters
        ----------
        ndim : {2, 3}
            Number of dimensions of this geometry, i.e. dimensionality
            of the physical space in which this geometry is embedded
        apart : `RectPartition`
            Partition of the angle set
        detector : `Detector`
            The detector to use in this geometry
        det_init_pos : `array-like`
            Initial position of the detector reference point
        """
        super().__init__(ndim, apart, detector)

        if self.ndim not in (2, 3):
            raise ValueError('number of dimensions is {}, expected 2 or 3'
                             ''.format(ndim))

        self._det_init_pos = np.asarray(det_init_pos, dtype='float64')
        if self._det_init_pos.shape != (self.ndim,):
            raise ValueError('initial detector position has shape {}, '
                             'expected ({},)'
                             ''.format(self._det_init_pos.shape, self.ndim))

    @property
    def angles(self):
        """Discrete angles given in this geometry."""
        return self.motion_grid.coord_vectors[0]

    def det_refpoint(self, angles):
        """Return the position of the detector ref. point at ``angles``.

        The reference point is given by a rotation of the initial
        position by ``angles``.

        Parameters
        ----------
        angles : float
            Parameters describing the detector rotation, must be
            contained in `motion_params`.

        Returns
        -------
        point : `numpy.ndarray`, shape (`ndim`,)
            The reference point for the given parameters
        """
        if angles not in self.motion_params:
            raise ValueError('`angles` {} not in the valid range {}'
                             ''.format(angles, self.motion_params))
        return self.rotation_matrix(angles).dot(self._det_init_pos)

    def det_to_src(self, angles, dpar, normalized=True):
        """Direction from a detector location to the source.

        In parallel geometry, this function is independent of the
        detector parameter.

        Since the (virtual) source is infinitely far away, only the
        normalized version is valid.

        Parameters
        ----------
        angles : `array-like`
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


class Parallel2dGeometry(ParallelGeometry):

    """Parallel beam geometry in 2d.

    The motion parameter is the counter-clockwise rotation angle around
    the origin, and the detector is a line detector perpendicular to the
    ray direction.

    In the standard configuration, the detector reference point starts
    at ``(1, 0)``, and the initial detector axis is ``(0, 1)``.
    """

    def __init__(self, apart, dpart, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        apart : 1-dim. `RectPartition`
            Partition of the angle interval
        dpart : 1-dim. `RectPartition`
            Partition of the detector parameter interval
        det_init_pos : `array-like`, shape ``(2,)``, optional
            Initial position of the detector reference point. The zero
            vector is only allowed if ``det_init_axis`` is explicitly
            given.
            Default: ``(1, 0)``.
        det_init_axis : `array-like` (shape ``(2,)``), optional
            Initial axis defining the detector orientation.
            By default, a normalized `perpendicular_vector` to
            ``det_init_pos`` is used, which is only valid if
            ``det_init_axis`` is not zero.
        """
        self._det_init_pos = kwargs.pop('det_init_pos', (1.0, 0.0))
        self._det_init_axis = kwargs.pop('det_init_axis', None)

        if self.det_init_axis is None:
            if np.linalg.norm(self.det_init_pos) <= 1e-10:
                raise ValueError('`det_init_pos` {} is close to '
                                 'zero. This is only allowed for explicit '
                                 'det_init_axis'
                                 ''.format(self.det_init_pos))

            det_init_axis = perpendicular_vector(self.det_init_pos)

        detector = Flat1dDetector(part=dpart, axis=det_init_axis)
        super().__init__(ndim=2, apart=apart, detector=detector,
                         det_init_pos=self.det_init_pos)

        if self.motion_partition.ndim != 1:
            raise ValueError('`apart` dimension {}, expected 1'
                             ''.format(self.motion_partition.ndim))

    @property
    def det_init_pos(self):
        """Position of the detector reference point at angle=0."""
        return self._det_init_pos

    @property
    def det_init_axis(self):
        """Direction of the detector extent at angle=0."""
        return self._det_init_axis

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
        if angle not in self.motion_params:
            raise ValueError('`angle` {} not in the valid range {}'
                             ''.format(angle, self.motion_params))
        return euler_matrix(angle)

    def __repr__(self):
        """Return ``repr(self)``."""

        inner_fstr = '\n    {!r},\n    {!r}'

        if not np.allclose(self.det_init_pos, [1, 0]):
            inner_fstr += ',\n    det_init_pos={det_init_pos!r}'

        if not np.allclose(self.det_init_pos,
                           perpendicular_vector(self.det_init_pos)):
            inner_fstr += ',\n    det_init_axis={det_init_axis!r}'

        inner_str = inner_fstr.format(self.motion_partition,
                                      self.det_partition,
                                      det_init_pos=self._det_init_pos,
                                      det_init_axis=self.det_init_axis)
        return '{}({})'.format(self.__class__.__name__, inner_str)


class Parallel3dEulerGeometry(ParallelGeometry):

    """Parallel beam geometry in 3d.

    The motion parameters are two or three Euler angles, and the detector
    is flat and two-dimensional.

    In the standard configuration, the detector reference point starts
    at ``(1, 0, 0)``, and the initial detector axes are
    ``[(0, 1, 0), (0, 0, 1)]``.
    """

    def __init__(self, apart, dpart, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        apart : 2- or 3-dim. `RectPartition`
            Partition of the angle parameter set
        dpart : 2-dim. `RectPartition`
            Partition of the detector parameter interval
        det_init_pos : `array-like`, shape ``(3,)``, optional
            Initial position of the detector reference point. The zero
            vector is only allowed if ``det_init_axes`` is explicitly
            given.
            Default: ``(1, 0, 0)``
        det_init_axes : 2-tuple of `array-like`'s (shape ``(3,)``), optional
            Initial axes defining the detector orientation.
            By default, a normalized `perpendicular_vector` to
            ``det_init_pos`` is taken as first axis, and the normalized
            cross product of these two as second.
        """
        det_init_pos = kwargs.pop('det_init_pos', (1.0, 0.0, 0.0))
        det_init_axes = kwargs.pop('det_init_axes', None)

        if det_init_axes is None:
            if np.linalg.norm(det_init_pos) <= 1e-10:
                raise ValueError('`det_init_pos` {} is close to '
                                 'zero. This is only allowed for explicit '
                                 '`det_init_axes`.'.format(det_init_pos))

            det_init_axis_0 = perpendicular_vector(det_init_pos)
            det_init_axis_1 = np.cross(det_init_pos, det_init_axis_0)
            det_init_axes = (det_init_axis_0, det_init_axis_1)

        detector = Flat2dDetector(part=dpart, axes=det_init_axes)
        super().__init__(ndim=3, apart=apart, detector=detector,
                         det_init_pos=det_init_pos)

        if self.motion_partition.ndim not in (2, 3):
            raise ValueError('`apart` has dimension {}, expected '
                             '2 or 3'.format(self.motion_partition.ndim))

    def rotation_matrix(self, angles):
        """Matrix defining the detector rotation at ``angles``.

        Parameters
        ----------
        angles : `array-like`
            Angles in radians defining the rotation, must be contained
            in this geometry's ``motion_params``

        Returns
        -------
        rot : `numpy.ndarray`, shape ``(3, 3)``
            The rotation matrix mapping the standard basis vectors in
            the fixed ("lab") coordinate system to the basis vectors of
            the local coordinate system of the detector reference point,
            expressed in the fixed system.
        """
        if angles not in self.motion_params:
            raise ValueError('`angles` {} not in the valid range {}'
                             ''.format(angles, self.motion_params))
        return euler_matrix(*angles)

    def __repr__(self):
        """Return ``repr(self)``."""
        # TODO: det_init_axes
        inner_fstr = '\n    {!r},\n    {!r}'

        if not np.allclose(self.origin_to_det, [1, 0]):
            inner_fstr += ',\n    det_init_pos={det_init_pos!r}'

        inner_str = inner_fstr.format(self.motion_partition,
                                      self.det_partition,
                                      det_init_pos=self._det_init_pos)
        return '{}({})'.format(self.__class__.__name__, inner_str)


class Parallel3dAxisGeometry(ParallelGeometry, AxisOrientedGeometry):

    """Parallel beam geometry in 3d with single rotation axis.

    The motion parameter is the rotation angle around the specified
    axis, and the detector is a flat 2d detector perpendicular to the
    ray direction.

    In the standard configuration, the rotation axis is ``(0, 0, 1)``,
    the detector reference point starts at ``(1, 0, 0)``, and the
    initial detector axes are ``[(0, 1, 0), (0, 0, 1)]``.
    """

    def __init__(self, apart, dpart, axis=[0, 0, 1], **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        apart : 1-dim. `RectPartition`
            Partition of the angle interval
        dpart : 2-dim. `RectPartition`
            Partition of the detector parameter interval
        axis : `array-like`, shape ``(3,)``, optional
            Fixed rotation axis defined by a 3-element vector
        det_init_pos : `array-like`, shape ``(3,)``, optional
            Initial position of the detector reference point. The zero
            vector is only allowed if ``det_init_axes`` is explicitly
            given.
            By default, a `perpendicular_vector` to ``axis`` is used.
        det_init_axes : 2-tuple of `array-like`'s (shape ``(3,)``), optional
            Initial axes defining the detector orientation.
            By default, the normalized cross product of ``axis`` and
            ``det_init_pos`` is used as first axis and ``axis`` as second.
        """
        AxisOrientedGeometry.__init__(self, axis)

        det_init_pos = kwargs.pop('det_init_pos', perpendicular_vector(axis))
        det_init_axes = kwargs.pop('det_init_axes', None)

        if det_init_axes is None:
            if np.linalg.norm(det_init_pos) <= 1e-10:
                raise ValueError('initial detector position {} is close to '
                                 'zero. This is only allowed for explicit '
                                 '`det_init_axes`.'.format(det_init_pos))

            det_init_axis_0 = np.cross(self.axis, det_init_pos)
            det_init_axis_0 /= np.linalg.norm(det_init_axis_0)
            det_init_axes = (det_init_axis_0, axis)

        self.det_init_axes = det_init_axes

        detector = Flat2dDetector(part=dpart, axes=det_init_axes)
        super().__init__(ndim=3, apart=apart, detector=detector,
                         det_init_pos=det_init_pos)

        if self.motion_partition.ndim != 1:
            raise ValueError('`apart` has dimension {}, expected 1'
                             ''.format(self.motion_partition.ndim))

    def __repr__(self):
        """Return ``repr(self)``."""
        arg_fstr = '\n    {!r},\n    {!r}'
        if not np.allclose(self.axis, [0, 0, 1]):
            arg_fstr += ',\n    axis={axis!r}'

        if not np.allclose(self._det_init_pos,
                           perpendicular_vector(self.axis)):
            arg_fstr += ',\n    det_init_pos={det_init_pos!r}'

        default_axes = [np.cross(self.axis, self._det_init_pos), self.axis]
        if not np.allclose(self.detector.axes, default_axes):
            arg_fstr += ',\n    det_init_axes={det_init_axes!r}'

        arg_str = arg_fstr.format(self.motion_partition,
                                  self.det_partition,
                                  axis=self.axis,
                                  det_init_pos=self._det_init_pos,
                                  det_init_axes=self.detector.axes)
        return '{}({})'.format(self.__class__.__name__, arg_str)

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
    geometry : `ParallelGeometry`
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
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
