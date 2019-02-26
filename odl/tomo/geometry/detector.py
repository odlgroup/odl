# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Detectors for tomographic imaging."""

from __future__ import print_function, division, absolute_import
from builtins import object
import numpy as np

from odl.discr import RectPartition
from odl.tomo.util import perpendicular_vector, is_inside_bounds
from odl.util import indent, signature_string, array_str
from odl.util.npy_compat import moveaxis


__all__ = ('Detector',
           'Flat1dDetector', 'Flat2dDetector',
           'CircularDetector')


class Detector(object):

    """Abstract detector class.

    A detector is described by

    * a set of parameters for surface parametrization (including sampling),
    * a function mapping a surface parameter to the location of a detector
      point relative to its reference point,
    * optionally a surface measure function.

    Most implementations implicitly assume that an N-dimensional detector
    is embedded in an (N+1)-dimensional space, but subclasses can override
    this behavior.
    """

    def __init__(self, partition, space_ndim=None, check_bounds=True):
        """Initialize a new instance.

        Parameters
        ----------
        partition : `RectPartition`
            Partition of the detector parameter set (pixelization).
            It determines dimension, parameter range and discretization.
        space_ndim : positive int, optional
            Number of dimensions of the embedding space.
            Default: ``partition.ndim + 1``
        check_bounds : bool, optional
            If ``True``, methods computing vectors check input arguments.
            Checks are vectorized and add only a small overhead.
        """
        if not isinstance(partition, RectPartition):
            raise TypeError('`partition` {!r} is not a RectPartition instance'
                            ''.format(partition))

        if space_ndim is None:
            self.__space_ndim = partition.ndim + 1
        else:
            self.__space_ndim = int(space_ndim)
            if self.space_ndim <= 0:
                raise ValueError('`space_ndim` must be postitive, got {}'
                                 ''.format(space_ndim))

        self.__partition = partition
        self.__check_bounds = bool(check_bounds)

    @property
    def partition(self):
        """Partition of the detector parameter set into subsets."""
        return self.__partition

    @property
    def check_bounds(self):
        """If ``True``, methods computing vectors check input arguments.

        For very large input arrays, these checks can introduce significant
        overhead, but the overhead is kept low by vectorization.
        """
        return self.__check_bounds

    @property
    def ndim(self):
        """Number of dimensions of the parameters (= surface dimension)."""
        return self.partition.ndim

    @property
    def space_ndim(self):
        """Number of dimensions of the embedding space.

        This default (``space_ndim = ndim + 1``) can be overridden by
        subclasses.
        """
        return self.__space_ndim

    @property
    def params(self):
        """Surface parameter set of this detector."""
        return self.partition.set

    @property
    def grid(self):
        """Sampling grid of the parameters."""
        return self.partition.grid

    @property
    def shape(self):
        """Number of subsets (pixels) of the detector per axis."""
        return self.partition.shape

    @property
    def size(self):
        """Total number of pixels."""
        return self.partition.size

    def surface(self, param):
        """Parametrization of the detector reference surface.

        Parameters
        ----------
        param : `array-like` or sequence
            Parameter value(s) at which to evaluate.

        Returns
        -------
        point : `numpy.ndarray`
            Vector(s) pointing from the origin to the detector surface
            point at ``param``.
        """
        raise NotImplementedError('abstract method')

    def surface_deriv(self, param):
        """Partial derivative(s) of the surface parametrization.

        Parameters
        ----------
        param : `array-like` or sequence
            Parameter value(s) at which to evaluate. If ``ndim >= 2``,
            a sequence of length `ndim` must be provided.

        Returns
        -------
        deriv : `numpy.ndarray`
            Array of vectors representing the surface derivative(s) at
            ``param``.
        """
        raise NotImplementedError('abstract method')

    def surface_normal(self, param):
        """Unit vector perpendicular to the detector surface at ``param``.

        The orientation is chosen as follows:

            - In 2D, the system ``(normal, tangent)`` should be
              right-handed.
            - In 3D, the system ``(tangent[0], tangent[1], normal)``
              should be right-handed.

        Here, ``tangent`` is the return value of `surface_deriv` at
        ``param``.

        Parameters
        ----------
        param : `array-like` or sequence
            Parameter value(s) at which to evaluate.  If ``ndim >= 2``,
            a sequence of length `ndim` must be provided.

        Returns
        -------
        normal : `numpy.ndarray`
            Unit vector(s) perpendicular to the detector surface at
            ``param``.
            If ``param`` is a single parameter, an array of shape
            ``(space_ndim,)`` representing a single vector is returned.
            Otherwise the shape of the returned array is

            - ``param.shape + (space_ndim,)`` if `ndim` is 1,
            - ``param.shape[:-1] + (space_ndim,)`` otherwise.
        """
        # Checking is done by `surface_deriv`
        if self.ndim == 1 and self.space_ndim == 2:
            return -perpendicular_vector(self.surface_deriv(param))
        elif self.ndim == 2 and self.space_ndim == 3:
            deriv = self.surface_deriv(param)
            if deriv.ndim > 2:
                # Vectorized, need to reshape (N, 2, 3) to (2, N, 3)
                deriv = moveaxis(deriv, -2, 0)
            normal = np.cross(*deriv, axis=-1)
            normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
            return normal
        else:
            raise NotImplementedError(
                'no default implementation of `surface_normal` available '
                'for `ndim = {}` and `space_ndim = {}`'
                ''.format(self.ndim, self.space_ndim))

    def surface_measure(self, param):
        """Density function of the surface measure.

        This is the default implementation relying on the `surface_deriv`
        method. For a detector with `ndim` equal to 1, the density is given
        by the `Arc length`_, for a surface with `ndim` 2 in a 3D space, it
        is the length of the cross product of the partial derivatives of the
        parametrization, see Wikipedia's `Surface area`_ article.

        Parameters
        ----------
        param : `array-like` or sequence
            Parameter value(s) at which to evaluate.  If ``ndim >= 2``,
            a sequence of length `ndim` must be provided.

        Returns
        -------
        measure : float or `numpy.ndarray`
            The density value(s) at the given parameter(s). If a single
            parameter is provided, a float is returned. Otherwise, an
            array is returned with shape

            - ``param.shape`` if `ndim` is 1,
            - ``broadcast(*param).shape`` otherwise.

        References
        ----------
        .. _Arc length:
            https://en.wikipedia.org/wiki/Curve#Lengths_of_curves
        .. _Surface area:
            https://en.wikipedia.org/wiki/Surface_area
        """
        # Checking is done by `surface_deriv`
        if self.ndim == 1:
            scalar_out = (np.shape(param) == ())
            measure = np.linalg.norm(self.surface_deriv(param), axis=-1)
            if scalar_out:
                measure = float(measure)

            return measure

        elif self.ndim == 2 and self.space_ndim == 3:
            scalar_out = (np.shape(param) == (2,))
            deriv = self.surface_deriv(param)
            if deriv.ndim > 2:
                # Vectorized, need to reshape (N, 2, 3) to (2, N, 3)
                deriv = moveaxis(deriv, -2, 0)
            cross = np.cross(*deriv, axis=-1)
            measure = np.linalg.norm(cross, axis=-1)
            if scalar_out:
                measure = float(measure)

            return measure

        else:
            raise NotImplementedError(
                'no default implementation of `surface_measure` available '
                'for `ndim={}` and `space_ndim={}`'
                ''.format(self.ndim, self.space_ndim))


class Flat1dDetector(Detector):

    """A 1d line detector aligned with a given axis in 2D space."""

    def __init__(self, partition, axis, check_bounds=True):
        """Initialize a new instance.

        Parameters
        ----------
        partition : 1-dim. `RectPartition`
            Partition of the parameter interval, corresponding to the
            line elements.
        axis : `array-like`, shape ``(2,)``
            Fixed axis along which this detector is aligned.
        check_bounds : bool, optional
            If ``True``, methods computing vectors check input arguments.
            Checks are vectorized and add only a small overhead.

        Examples
        --------
        >>> part = odl.uniform_partition(0, 1, 10)
        >>> det = Flat1dDetector(part, axis=[1, 0])
        >>> det.axis
        array([ 1.,  0.])
        >>> np.allclose(det.surface_normal(0), [0, -1])
        True
        """
        super(Flat1dDetector, self).__init__(partition, 2, check_bounds)
        if self.ndim != 1:
            raise ValueError('`partition` must be 1-dimensional, got ndim={}'
                             ''.format(self.ndim))

        if np.linalg.norm(axis) == 0:
            raise ValueError('`axis` cannot be zero')
        self.__axis = np.asarray(axis) / np.linalg.norm(axis)

    @property
    def axis(self):
        """Fixed axis along which this detector is aligned."""
        return self.__axis

    def surface(self, param):
        """Return the detector surface point corresponding to ``param``.

        For parameter value ``p``, the surface point is given by ::

            surf = p * axis

        Parameters
        ----------
        param : float or `array-like`
            Parameter value(s) at which to evaluate.

        Returns
        -------
        point : `numpy.ndarray`
            Vector(s) pointing from the origin to the detector surface
            point at ``param``.
            If ``param`` is a single parameter, the returned array has
            shape ``(2,)``, otherwise ``param.shape + (2,)``.

        Examples
        --------
        The method works with a single parameter, resulting in a single
        vector:

        >>> part = odl.uniform_partition(0, 1, 10)
        >>> det = Flat1dDetector(part, axis=[1, 0])
        >>> det.surface(0)
        array([ 0.,  0.])
        >>> det.surface(1)
        array([ 1.,  0.])

        It is also vectorized, i.e., it can be called with multiple
        parameters at once (or an n-dimensional array of parameters):

        >>> det.surface([0, 1])
        array([[ 0.,  0.],
               [ 1.,  0.]])
        >>> det.surface(np.zeros((4, 5))).shape
        (4, 5, 2)
        """
        squeeze_out = (np.shape(param) == ())
        param = np.array(param, dtype=float, copy=False, ndmin=1)
        if self.check_bounds and not is_inside_bounds(param, self.params):
            raise ValueError('`param` {} not in the valid range '
                             '{}'.format(param, self.params))

        # Create outer product of `params` and `axis`, resulting in shape
        # params.shape + axis.shape
        surf = np.multiply.outer(param, self.axis)
        if squeeze_out:
            surf = surf.squeeze()

        return surf

    def surface_deriv(self, param):
        """Return the surface derivative at ``param``.

        This is a constant function evaluating to `axis` everywhere.

        Parameters
        ----------
        param : float or `array-like`
            Parameter value(s) at which to evaluate.

        Returns
        -------
        deriv : `numpy.ndarray`
            Array representing the derivative vector(s) at ``param``.
            If ``param`` is a single parameter, the returned array has
            shape ``(2,)``, otherwise ``param.shape + (2,)``.

        Examples
        --------
        The method works with a single parameter, resulting in a single
        vector:

        >>> part = odl.uniform_partition(0, 1, 10)
        >>> det = Flat1dDetector(part, axis=[1, 0])
        >>> det.surface_deriv(0)
        array([ 1.,  0.])
        >>> det.surface_deriv(1)
        array([ 1.,  0.])

        It is also vectorized, i.e., it can be called with multiple
        parameters at once (or an n-dimensional array of parameters):

        >>> det.surface_deriv([0, 1])
        array([[ 1.,  0.],
               [ 1.,  0.]])
        >>> det.surface_deriv(np.zeros((4, 5))).shape
        (4, 5, 2)
        """
        squeeze_out = (np.shape(param) == ())
        param = np.array(param, dtype=float, copy=False, ndmin=1)
        if self.check_bounds and not is_inside_bounds(param, self.params):
            raise ValueError('`param` {} not in the valid range '
                             '{}'.format(param, self.params))
        if squeeze_out:
            return self.axis
        else:
            # Produce array of shape `param.shape + (ndim,)` by broadcasting
            bcast_slc = (None,) * param.ndim + (slice(None),)
            return np.broadcast_to(
                self.axis[bcast_slc], param.shape + self.axis.shape)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.partition]
        optargs = [('axis', array_str(self.axis), '')]
        inner_str = signature_string(posargs, optargs, sep=',\n')
        return '{}(\n{}\n)'.format(self.__class__.__name__, indent(inner_str))

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


class Flat2dDetector(Detector):

    """A 2D flat panel detector aligned two given axes in 3D space."""

    def __init__(self, partition, axes, check_bounds=True):
        """Initialize a new instance.

        Parameters
        ----------
        partition : 2-dim. `RectPartition`
            Partition of the parameter rectangle, corresponding to the
            pixels.
        axes : sequence of `array-like`'s
            Fixed pair of of unit vectors with which the detector is aligned.
            The vectors must have shape ``(3,)`` and be linearly
            independent.
        check_bounds : bool, optional
            If ``True``, methods computing vectors check input arguments.
            Checks are vectorized and add only a small overhead.

        Examples
        --------
        >>> part = odl.uniform_partition([0, 0], [1, 1], (10, 10))
        >>> det = Flat2dDetector(part, axes=[(1, 0, 0), (0, 0, 1)])
        >>> det.axes
        array([[ 1.,  0.,  0.],
               [ 0.,  0.,  1.]])
        >>> det.surface_normal([0, 0])
        array([ 0., -1.,  0.])
        """
        super(Flat2dDetector, self).__init__(partition, 3, check_bounds)
        if self.ndim != 2:
            raise ValueError('`partition` must be 2-dimensional, got ndim={}'
                             ''.format(self.ndim))

        axes, axes_in = np.asarray(axes, dtype=float), axes
        if axes.shape != (2, 3):
            raise ValueError('`axes` must be a sequence of 2 3-dimensional '
                             'vectors, got {}'.format(axes_in))
        if np.linalg.norm(np.cross(*axes)) == 0:
            raise ValueError('`axes` {} are linearly dependent'
                             ''.format(axes_in))

        self.__axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)

    @property
    def axes(self):
        """Fixed array of unit vectors with which the detector is aligned."""
        return self.__axes

    def surface(self, param):
        """Return the detector surface point corresponding to ``param``.

        For parameter value ``p``, the surface point is given by ::

            surf = p[0] * axes[0] + p[1] * axes[1]

        Parameters
        ----------
        param : `array-like` or sequence
            Parameter value(s) at which to evaluate. A sequence of
            parameters must have length 2.

        Returns
        -------
        point : `numpy.ndarray`
            Vector(s) pointing from the origin to the detector surface
            point at ``param``.
            If ``param`` is a single parameter, the returned array has
            shape ``(3,)``, otherwise ``broadcast(*param).shape + (3,)``.

        Examples
        --------
        The method works with a single parameter, resulting in a single
        vector:

        >>> part = odl.uniform_partition([0, 0], [1, 1], (10, 10))
        >>> det = Flat2dDetector(part, axes=[(1, 0, 0), (0, 0, 1)])
        >>> det.surface([0, 0])
        array([ 0.,  0.,  0.])
        >>> det.surface([0, 1])
        array([ 0.,  0.,  1.])
        >>> det.surface([1, 1])
        array([ 1.,  0.,  1.])

        It is also vectorized, i.e., it can be called with multiple
        parameters at once (or n-dimensional arrays of parameters):

        >>> # 3 pairs of parameters, resulting in 3 vectors
        >>> det.surface([[0, 0, 1],
        ...              [0, 1, 1]])
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  1.],
               [ 1.,  0.,  1.]])
        >>> # Pairs of parameters in a (4, 5) array each
        >>> param = (np.zeros((4, 5)), np.zeros((4, 5)))
        >>> det.surface(param).shape
        (4, 5, 3)
        >>> # Using broadcasting for "outer product" type result
        >>> param = (np.zeros((4, 1)), np.zeros((1, 5)))
        >>> det.surface(param).shape
        (4, 5, 3)
        """
        squeeze_out = (np.broadcast(*param).shape == ())
        param_in = param
        param = tuple(np.array(p, dtype=float, copy=False, ndmin=1)
                      for p in param)
        if self.check_bounds and not is_inside_bounds(param, self.params):
            raise ValueError('`param` {} not in the valid range '
                             '{}'.format(param_in, self.params))

        # Compute outer product of the i-th spatial component of the
        # parameter and sum up the contributions
        surf = sum(np.multiply.outer(p, ax) for p, ax in zip(param, self.axes))
        if squeeze_out:
            surf = surf.squeeze()

        return surf

    def surface_deriv(self, param):
        """Return the surface derivative at ``param``.

        This is a constant function evaluating to `axes` everywhere.

        Parameters
        ----------
        param : `array-like` or sequence
            Parameter value(s) at which to evaluate. A sequence of
            parameters must have length 2.

        Returns
        -------
        deriv : `numpy.ndarray`
            Array containing the derivative vectors. The first dimension
            enumerates the axes, i.e., has always length 2.
            If ``param`` is a single parameter, the returned array has
            shape ``(2, 3)``, otherwise
            ``broadcast(*param).shape + (2, 3)``.

        Notes
        -----
        To get an array that enumerates the derivative vectors in the first
        dimension, move the second-to-last axis to the first position::

            deriv = surface_deriv(param)
            axes_enumeration = np.moveaxis(deriv, -2, 0)

        Examples
        --------
        The method works with a single parameter, resulting in a 2-tuple
        of vectors:

        >>> part = odl.uniform_partition([0, 0], [1, 1], (10, 10))
        >>> det = Flat2dDetector(part, axes=[(1, 0, 0), (0, 0, 1)])
        >>> det.surface_deriv([0, 0])
        array([[ 1.,  0.,  0.],
               [ 0.,  0.,  1.]])
        >>> det.surface_deriv([1, 1])
        array([[ 1.,  0.,  0.],
               [ 0.,  0.,  1.]])

        It is also vectorized, i.e., it can be called with multiple
        parameters at once (or n-dimensional arrays of parameters):

        >>> # 2 pairs of parameters, resulting in 3 vectors for each axis
        >>> deriv = det.surface_deriv([[0, 1],
        ...                            [0, 1]])
        >>> deriv[0]  # first pair of vectors
        array([[ 1.,  0.,  0.],
               [ 0.,  0.,  1.]])
        >>> deriv[1]  # second pair of vectors
        array([[ 1.,  0.,  0.],
               [ 0.,  0.,  1.]])
        >>> # Pairs of parameters in a (4, 5) array each
        >>> param = (np.zeros((4, 5)), np.zeros((4, 5)))  # pairs of params
        >>> det.surface_deriv(param).shape
        (4, 5, 2, 3)
        >>> # Using broadcasting for "outer product" type result
        >>> param = (np.zeros((4, 1)), np.zeros((1, 5)))  # broadcasting
        >>> det.surface_deriv(param).shape
        (4, 5, 2, 3)
        """
        squeeze_out = (np.broadcast(*param).shape == ())
        param_in = param
        param = tuple(np.array(p, dtype=float, copy=False, ndmin=1)
                      for p in param)
        if self.check_bounds and not is_inside_bounds(param, self.params):
            raise ValueError('`param` {} not in the valid range '
                             '{}'.format(param_in, self.params))

        if squeeze_out:
            return self.axes
        else:
            return np.broadcast_to(
                self.axes, np.broadcast(*param).shape + self.axes.shape)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.partition]
        optargs = [('axes', tuple(array_str(ax) for ax in self.axes), None)]
        inner_str = signature_string(posargs, optargs, sep=',\n')
        return '{}(\n{}\n)'.format(self.__class__.__name__, indent(inner_str))

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


class CircularDetector(Detector):

    """A 1D detector on a circle section in 2D space.

    The circular section that corresponds to the angular partition
    is rotated to be aligned with a given axis and
    shifted to cross the origin. Note, the partition angle increases
    in the clockwise direction, by analogy to flat detectors."""

    def __init__(self, partition, axis, radius, check_bounds=True):
        """Initialize a new instance.

        Parameters
        ----------
        partition : 1-dim. `RectPartition`
            Partition of the parameter interval, corresponding to the
            angular sections along the line.
        axis : `array-like`, shape ``(2,)``
            Fixed axis along which this detector is aligned.
        radius : nonnegative float
            Radius of the circle.
        check_bounds : bool, optional
            If ``True``, methods computing vectors check input arguments.
            Checks are vectorized and add only a small overhead.

        Examples
        --------
        Initialize a detector with circle radius 2 and extending to
        90 degrees on both sides of the origin (a half circle).

        >>> part = odl.uniform_partition(-np.pi / 2, np.pi / 2, 10)
        >>> det = CircularDetector(part, axis=[1, 0], radius=2)
        >>> det.axis
        array([ 1.,  0.])
        >>> det.radius
        2.0
        >>> np.allclose(det.surface_normal(0), [0, -1])
        True
        """
        super(CircularDetector, self).__init__(partition, 2, check_bounds)
        if self.ndim != 1:
            raise ValueError('`partition` must be 1-dimensional, got ndim={}'
                             ''.format(self.ndim))

        if np.linalg.norm(axis) == 0:
            raise ValueError('`axis` cannot be zero')
        self.__axis = np.asarray(axis) / np.linalg.norm(axis)

        self.__radius = float(radius)
        if self.__radius <= 0:
            raise ValueError('`radius` must be positive')

        sin = self.__axis[0]
        cos = -self.__axis[1]
        self.__rotation_matrix = np.array([[cos, -sin], [sin, cos]])
        self.__translation = (- self.__radius
                              * np.matmul(self.__rotation_matrix, (1, 0)))

    @property
    def axis(self):
        """Fixed axis along which this detector is aligned."""
        return self.__axis

    @property
    def radius(self):
        """Curvature radius of the detector."""
        return self.__radius

    @property
    def rotation_matrix(self):
        """Rotation matrix that is used to align the detector
        with a given axis."""
        return self.__rotation_matrix

    @property
    def translation(self):
        """A vector used to shift the detector towards the origin."""
        return self.__translation

    def surface(self, param):
        """Return the detector surface point corresponding to ``param``.

        For a parameter ``phi``, the returned point is given by ::

            surf = R * radius * (cos(phi), -sin(phi)) + t

        where ``R`` is a rotation matrix and ``t`` is a translation vector.
        Note that increase of ``phi`` corresponds to rotation
        in the clockwise direction, by analogy to flat detectors.

        Parameters
        ----------
        param : float or `array-like`
            Parameter value(s) at which to evaluate.

        Returns
        -------
        point : `numpy.ndarray`
            Vector(s) pointing from the origin to the detector surface
            point at ``param``.
            If ``param`` is a single parameter, the returned array has
            shape ``(2,)``, otherwise ``param.shape + (2,)``.

        Examples
        --------
        The method works with a single parameter, resulting in a single
        vector:

        >>> part = odl.uniform_partition(-np.pi / 2, np.pi / 2, 10)
        >>> det = CircularDetector(part, axis=[1, 0], radius=2)
        >>> np.allclose(det.surface(0), [0, 0])
        True

        It is also vectorized, i.e., it can be called with multiple
        parameters at once (or an n-dimensional array of parameters):

        >>> np.round(det.surface([-np.pi / 2, 0, np.pi / 2]), 10)
        array([[-2., -2.],
               [ 0.,  0.],
               [ 2., -2.]])

        >>> det.surface(np.zeros((4, 5))).shape
        (4, 5, 2)
        """
        squeeze_out = (np.shape(param) == ())
        param = np.array(param, dtype=float, copy=False, ndmin=1)
        if self.check_bounds and not is_inside_bounds(param, self.params):
            raise ValueError('`param` {} not in the valid range '
                             '{}'.format(param, self.params))

        surf = np.empty(param.shape + (2,))
        surf[..., 0] = np.cos(param)
        surf[..., 1] = -np.sin(param)
        surf *= self.radius
        surf = np.matmul(surf, np.transpose(self.rotation_matrix))
        surf += self.translation
        if squeeze_out:
            surf = surf.squeeze()

        return surf

    def surface_deriv(self, param):
        """Return the surface derivative at ``param``.

        The derivative at parameter ``phi`` is given by ::

            deriv = R * radius * (-sin(phi), -cos(phi))

        where R is a rotation matrix.

        Parameters
        ----------
        param : float or `array-like`
            Parameter value(s) at which to evaluate.

        Returns
        -------
        deriv : `numpy.ndarray`
            Array representing the derivative vector(s) at ``param``.
            If ``param`` is a single parameter, the returned array has
            shape ``(2,)``, otherwise ``param.shape + (2,)``.

        See Also
        --------
        surface

        Examples
        --------
        The method works with a single parameter, resulting in a single
        vector:

        >>> part = odl.uniform_partition(-np.pi / 2, np.pi / 2, 10)
        >>> det = CircularDetector(part, axis=[1, 0], radius=2)
        >>> det.surface_deriv(0)
        array([ 2.,  0.])

        It is also vectorized, i.e., it can be called with multiple
        parameters at once (or an n-dimensional array of parameters):

        >>> np.round(det.surface_deriv([-np.pi / 2, 0, np.pi / 2]), 10)
        array([[ 0.,  2.],
               [ 2.,  0.],
               [ 0., -2.]])

        >>> det.surface_deriv(np.zeros((4, 5))).shape
        (4, 5, 2)
        """
        squeeze_out = (np.shape(param) == ())
        param = np.array(param, dtype=float, copy=False, ndmin=1)
        if self.check_bounds and not is_inside_bounds(param, self.params):
            raise ValueError('`param` {} not in the valid range '
                             '{}'.format(param, self.params))

        deriv = np.empty(param.shape + (2,))
        deriv[..., 0] = -np.sin(param)
        deriv[..., 1] = -np.cos(param)
        deriv *= self.radius
        deriv = np.matmul(deriv, np.transpose(self.rotation_matrix))

        if squeeze_out:
            deriv = deriv.squeeze()

        return deriv

    def surface_measure(self, param):
        """Return the arc length measure at ``param``.

        This is a constant function evaluating to `radius` everywhere.

        Parameters
        ----------
        param : float or `array-like`
            Parameter value(s) at which to evaluate.

        Returns
        -------
        measure : float or `numpy.ndarray`
            Constant value(s) of the arc length measure at ``param``.
            If ``param`` is a single parameter, a float is returned,
            otherwise an array of shape ``param.shape``.

        See Also
        --------
        surface
        surface_deriv

        Examples
        --------
        The method works with a single parameter, resulting in a float:

        >>> part = odl.uniform_partition(-np.pi / 2, np.pi / 2, 10)
        >>> det = CircularDetector(part, axis=[1, 0], radius=2)
        >>> det.surface_measure(0)
        2.0
        >>> det.surface_measure(np.pi / 2)
        2.0

        It is also vectorized, i.e., it can be called with multiple
        parameters at once (or an n-dimensional array of parameters):

        >>> det.surface_measure([0, np.pi / 2])
        array([ 2.,  2.])
        >>> det.surface_measure(np.zeros((4, 5))).shape
        (4, 5)
        """
        scalar_out = (np.shape(param) == ())
        param = np.array(param, dtype=float, copy=False, ndmin=1)
        if self.check_bounds and not is_inside_bounds(param, self.params):
            raise ValueError('`param` {} not in the valid range '
                             '{}'.format(param, self.params))

        if scalar_out:
            return self.radius
        else:
            return self.radius * np.ones(param.shape)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.partition]
        optargs = [('radius', array_str(self.center), '')]
        inner_str = signature_string(posargs, optargs, sep=',\n')
        return '{}(\n{}\n)'.format(self.__class__.__name__, indent(inner_str))

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
