# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Detectors for tomographic imaging."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from builtins import object

import numpy as np

from odl.discr import RectPartition
from odl.tomo.util.utility import perpendicular_vector
from odl.util import indent_rows, signature_string


__all__ = ('Detector',
           'Flat1dDetector', 'Flat2dDetector',
           'CircleSectionDetector')


class Detector(object):

    """Abstract detector class.

    A detector is described by

    * a set of parameters for surface parametrization (including sampling),
    * a function mapping a surface parameter to the location of a detector
      point relative to its reference point,
    * optionally a surface measure function.
    """

    def __init__(self, partition, check_bounds=True):
        """Initialize a new instance.

        Parameters
        ----------
        partition : `RectPartition`
           Partition of the detector parameter set (pixelization).
           It determines dimension, parameter range and discretization.
        check_bounds : bool, optional
            If ``True``, methods perform sanity checks on provided input
            parameters.
        """
        if not isinstance(partition, RectPartition):
            raise TypeError('`partition` {!r} is not a RectPartition instance'
                            ''.format(partition))

        self.__partition = partition
        self.__check_bounds = bool(check_bounds)

    @property
    def partition(self):
        """Partition of the detector parameter set into subsets."""
        return self.__partition

    @property
    def check_bounds(self):
        """Whether to check if method parameters are in the valid range."""
        return self.__check_bounds

    @property
    def ndim(self):
        """Number of dimensions of this detector (0, 1 or 2)."""
        return self.partition.ndim

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
        param : `array-like`
            Parameter value(s) at which to evaluate. An array should
            stack parameters along axis 0.

        Returns
        -------
        point : `numpy.ndarray`, shape (ndim,) or (num_params, ndim)
            Vector(s) pointing from the origin to the detector surface
            point at ``param``.
            If ``param`` is a single parameter, a single vector is
            returned, otherwise a stack of vectors along axis 0.
        """
        raise NotImplementedError('abstract method')

    def surface_deriv(self, param):
        """Partial derivative(s) of the surface parametrization.

        Parameters
        ----------
        param : `array-like`
            Parameter value(s) at which to evaluate. An array should
            stack parameters along axis 0.

        Returns
        -------
        deriv : `numpy.ndarray` or tuple
            If `ndim` is 1, an array of shape ``(ndim,)`` is returned if
            ``param`` is a single parameter, and an array of shape
            ``(num_params, ndim)`` for multiple parameters.

            For higher `ndim`, a tuple of length `ndim` is returned,
            where each entry is as described for the 1D case. The
            i-th entry corresponds to the i-th partial derivative.
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
        param : float or `array-like`
            Parameter value(s) at which to evaluate. An array should stack
            parameters along axis 0.

        Returns
        -------
        normal : `numpy.ndarray`, shape (2,) or (num_params, 2)
            Unit vector(s) perpendicular to the detector surface at
            ``param``.
            If ``param`` is a single parameter, a single vector is
            returned, otherwise a stack of vectors along axis 0.
        """
        if self.ndim == 1:
            return -perpendicular_vector(self.surface_deriv(param))
        elif self.ndim == 2:
            normal = np.cross(*self.surface_deriv(param), axis=-1)
            normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
            return normal
        else:
            raise NotImplementedError('normal not defined for ndim >= 3')

    def surface_measure(self, param):
        """Density function of the surface measure.

        This is the default implementation relying on the `surface_deriv`
        method. For ``ndim == 1``, the density is given by the `Arc
        length`_, for ``ndim == 2``, it is the length of the cross product
        of the partial derivatives of the parametrization, see Wikipedia's
        `Surface area`_ article.

        Parameters
        ----------
        param : `array-like`
            Parameter value(s) at which to evaluate. An array should
            stack parameters along axis 0.

        Returns
        -------
        measure : float or `numpy.ndarray`
            The density value(s) at the given parameter(s).

        .. _Arc length:
            https://en.wikipedia.org/wiki/Curve#Lengths_of_curves
        .. _Surface area:
            https://en.wikipedia.org/wiki/Surface_area
        """
        if self.ndim == 1:
            scalar_out = np.isscalar(param)
            measure = np.linalg.norm(self.surface_deriv(param), axis=-1)
            if scalar_out:
                measure = float(measure)

            return measure

        elif self.ndim == 2:
            scalar_out = (np.shape(param) == (2,))
            deriv = self.surface_deriv(param)
            cross = np.cross(*deriv, axis=-1)
            measure = np.linalg.norm(cross, axis=-1)
            if scalar_out:
                measure = float(measure)

            return measure

        else:
            raise NotImplementedError('not implemented for ndim >= 3')


class Flat1dDetector(Detector):

    """A 1d line detector aligned with a given axis."""

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
            If ``True``, methods perform sanity checks on provided input
            parameters.

        Examples
        --------
        >>> part = odl.uniform_partition(0, 1, 10)
        >>> det = Flat1dDetector(part, axis=[1, 0])
        >>> det.axis
        array([ 1.,  0.])
        >>> np.allclose(det.surface_normal(0), [0, -1])
        True
        """
        super(Flat1dDetector, self).__init__(partition, check_bounds)
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
        point : `numpy.ndarray`, shape (2,) or (num_params, 2)
            Vector(s) pointing from the origin to the detector surface
            point at ``param``.
            If ``param`` is a single parameter, a single vector is
            returned, otherwise a stack of vectors along axis 0.

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
        parameters at once:

        >>> det.surface([0, 1])
        array([[ 0.,  0.],
               [ 1.,  0.]])
        """
        squeeze_out = np.isscalar(param)
        param = np.array(param, dtype=float, copy=False, ndmin=1)
        if self.check_bounds and not self.params.contains_all(param):
            raise ValueError('`param` {} not in the valid range '
                             '{}'.format(param, self.params))
        surf = param[:, None] * self.axis[None, :]
        if squeeze_out:
            surf = surf.squeeze()

        return surf

    def surface_deriv(self, param=None):
        """Return the surface derivative at ``param``.

        This is a constant function evaluating to `axis` everywhere.

        Parameters
        ----------
        param : float or `array-like`
            Parameter value(s) at which to evaluate.

        Returns
        -------
        deriv : `numpy.ndarray`, shape (2,) or (num_params, 2)
            Vector(s) representing the detector surface derivative at
            ``param``.
            If ``param`` is a single parameter, a single vector is
            returned, otherwise a stack of vectors along axis 0.

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
        parameters at once:

        >>> det.surface_deriv([0, 1])
        array([[ 1.,  0.],
               [ 1.,  0.]])
        """
        squeeze_out = np.isscalar(param)
        param = np.array(param, dtype=float, copy=False, ndmin=1)
        if self.check_bounds and not self.params.contains_all(param):
            raise ValueError('`param` {} not in the valid range '
                             '{}'.format(param, self.params))
        if squeeze_out:
            return self.axis
        else:
            return np.vstack([self.axis] * len(param))

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.partition]
        optargs = [('axis', self.axis.tolist(), None)]
        inner_str = signature_string(posargs, optargs, sep=',\n')
        return '{}(\n{}\n)'.format(self.__class__.__name__,
                                   indent_rows(inner_str))

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


class Flat2dDetector(Detector):

    """A 2d flat panel detector aligned two given axes."""

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
            If ``True``, methods perform sanity checks on provided input
            parameters.

        Examples
        --------
        >>> part = odl.uniform_partition([0, 0], [1, 1], (10, 10))
        >>> det = Flat2dDetector(part, axes=[(1, 0, 0), (0, 0, 1)])
        >>> det.axes
        (array([ 1.,  0.,  0.]), array([ 0.,  0.,  1.]))
        >>> det.surface_normal([0, 0])
        array([ 0., -1.,  0.])
        """
        super(Flat2dDetector, self).__init__(partition, check_bounds)
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

        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        self.__axes = tuple(axes)

    @property
    def axes(self):
        """Fixed 2-tuple of unit vectors with which the detector is aligned."""
        return self.__axes

    def surface(self, param):
        """Return the detector surface point corresponding to ``param``.

        For parameter value ``p``, the surface point is given by ::

            surf = p[0] * axes[0] + p[1] * axes[1]

        Parameters
        ----------
        param : `array-like`
            Parameter value(s) at which to evaluate. A 2D array should stack
            parameters along axis 0.

        Returns
        -------
        point : `numpy.ndarray`, shape (3,) or (num_params, 3)
            Vector(s) pointing from the origin to the detector surface
            point at ``param``.
            If ``param`` is a single parameter, a single vector is
            returned, otherwise a stack of vectors along axis 0.

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
        parameters at once:

        >>> det.surface([[0, 0], [0, 1], [1, 1]])
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  1.],
               [ 1.,  0.,  1.]])
        """
        squeeze_out = (np.shape(param) == (2,))
        # Need to transpose (IntervalProd.contains_all expects first axis
        # to be spatial component, second axis parameter enumeration)
        param, param_in = (np.array(param, dtype=float, copy=False, ndmin=2).T,
                           param)
        if self.check_bounds and not self.params.contains_all(param):
            raise ValueError('`param` {} not in the valid range '
                             '{}'.format(param_in, self.params))

        surf = sum(ax[None, :] * p[:, None]
                   for p, ax in zip(param, self.axes))
        if squeeze_out:
            surf = surf.squeeze()

        return surf

    def surface_deriv(self, param):
        """Return the surface derivative at ``param``.

        This is a constant function evaluating to `axes` everywhere.

        Parameters
        ----------
        param : `array-like`
            Parameter value(s) at which to evaluate. A 2D array should
            stack parameters along axis 0.

        Returns
        -------
        deriv : 2-tuple of `numpy.ndarray`
            The i-th entry is an array representing the surface derivative
            vector(s) at ``param`` with respect to the i-th coordinate.
            If ``param`` is a single parameter, each entry is an array
            of shape ``(3,)``, otherwise each entry is a stack of vectors
            along axis 0, i.e., an array of shape ``(num_params, 3)``.

        Examples
        --------
        The method works with a single parameter, resulting in a 2-tuple
        of vectors:

        >>> part = odl.uniform_partition([0, 0], [1, 1], (10, 10))
        >>> det = Flat2dDetector(part, axes=[(1, 0, 0), (0, 0, 1)])
        >>> det.surface_deriv([0, 0])
        (array([ 1.,  0.,  0.]), array([ 0.,  0.,  1.]))
        >>> det.surface_deriv([1, 1])
        (array([ 1.,  0.,  0.]), array([ 0.,  0.,  1.]))

        It is also vectorized, i.e., it can be called with multiple
        parameters at once:

        >>> deriv = det.surface_deriv([[0, 0], [1, 1]])
        >>> deriv[0]
        array([[ 1.,  0.,  0.],
               [ 1.,  0.,  0.]])
        >>> deriv[1]
        array([[ 0.,  0.,  1.],
               [ 0.,  0.,  1.]])
        """
        squeeze_out = (np.shape(param) == (2,))
        # Need to transpose (IntervalProd expects first axis to be spatial
        # component, not parameter enumeration)
        param, param_in = (np.array(param, dtype=float, copy=False, ndmin=2).T,
                           param)
        if self.check_bounds and not self.params.contains_all(param):
            raise ValueError('`param` {} not in the valid range '
                             '{}'.format(param_in, self.params))

        if squeeze_out:
            return self.axes
        else:
            return tuple(np.vstack([ax] * param.shape[1]) for ax in self.axes)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.partition]
        optargs = [('axes', tuple(ax.tolist() for ax in self.axes), None)]
        inner_str = signature_string(posargs, optargs, sep=',\n')
        return '{}(\n{}\n)'.format(self.__class__.__name__,
                                   indent_rows(inner_str))

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


class CircleSectionDetector(Detector):

    """A 1d detector given by a circle section that crosses the origin.

    The parametrization is chosen such that parameter (=angle) 0
    corresponds to the origin. Negative angles correspond to points
    "left" of the line from circle center to origin, positive angles
    to points on the "right" of that line.
    """

    def __init__(self, partition, center, check_bounds=True):
        """Initialize a new instance.

        Parameters
        ----------
        partition : 1-dim. `RectPartition`
            Partition of the parameter interval, corresponding to the
            angular sections along the line.
        center : `array-like`, shape ``(2,)``
            Center point of the circle, cannot be zero. Larger distance
            to the origin results in less curvature.
        check_bounds : bool, optional
            If ``True``, methods perform sanity checks on provided input
            parameters.

        Examples
        --------
        Initialize a detector with circle radius 2 and extending to
        90 degrees on both sides of the origin (a half circle).

        >>> part = odl.uniform_partition(-np.pi / 2, np.pi / 2, 10)
        >>> det = CircleSectionDetector(part, center=[0, -2])
        >>> det.radius
        2.0
        >>> det.center_dir
        array([ 0., -1.])
        >>> det.tangent_at_0
        array([ 1.,  0.])
        """
        super(CircleSectionDetector, self).__init__(partition, check_bounds)
        if self.ndim != 1:
            raise ValueError('`partition` must be 1-dimensional, got ndim={}'
                             ''.format(self.ndim))

        self.__center = np.asarray(center, dtype=float)
        if self.center.shape != (2,):
            raise ValueError('`center` must have shape (2,), got {}'
                             ''.format(self.center.shape))
        if np.linalg.norm(self.center) == 0:
            raise ValueError('`center` cannot be zero')

    @property
    def center(self):
        """Center point of the circle."""
        return self.__center

    @property
    def radius(self):
        """Curvature radius the detector."""
        return np.linalg.norm(self.center)

    @property
    def center_dir(self):
        """Unit vector from the origin to the center of the circle."""
        return self.center / self.radius

    @property
    def tangent_at_0(self):
        """Unit vector tangent to the circle at ``param=0``.

        The direction is chosen such that the circle is traversed clockwise
        for growing angle parameter.
        """
        return perpendicular_vector(self.center_dir)

    def surface(self, param):
        """Return the detector surface point corresponding to ``param``.

        The surface point lies on a circle around `center` through the
        origin. More precisely, for a parameter ``phi``, the returned
        point is given by ::

            surf = radius * ((1 - cos(phi)) * center_dir +
                             sin(phi) * tangent_at_0)

        In particular, ``phi=0`` yields the origin ``(0, 0)``.

        Parameters
        ----------
        param : float or `array-like`
            Parameter value(s) at which to evaluate.

        Returns
        -------
        point : `numpy.ndarray`, shape (2,) or (num_params, 2)
            Vector(s) pointing from the origin to the detector surface
            point at ``param``.
            If ``param`` is a single parameter, a single vector is
            returned, otherwise a stack of vectors along axis 0.

        Examples
        --------
        The method works with a single parameter, resulting in a single
        vector:

        >>> part = odl.uniform_partition(-np.pi / 2, np.pi / 2, 10)
        >>> det = CircleSectionDetector(part, center=[0, -2])
        >>> det.surface(0)
        array([ 0.,  0.])
        >>> det.surface(-np.pi / 2)
        array([-2., -2.])
        >>> det.surface(np.pi / 2)
        array([ 2., -2.])

        It is also vectorized, i.e., it can be called with multiple
        parameters at once:

        >>> det.surface([-np.pi / 2, 0, np.pi / 2])
        array([[-2., -2.],
               [ 0.,  0.],
               [ 2., -2.]])
        """
        squeeze_out = np.isscalar(param)
        param = np.array(param, dtype=float, copy=False, ndmin=1)
        if self.check_bounds and not self.params.contains_all(param):
            raise ValueError('`param` {} not in the valid range '
                             '{}'.format(param, self.params))
        surf = ((1 - np.cos(param))[:, None] * self.center_dir[None, :] +
                np.sin(param)[:, None] * self.tangent_at_0[None, :])
        surf *= self.radius
        if squeeze_out:
            surf = surf.squeeze()

        return surf

    def surface_deriv(self, param):
        """Return the surface derivative at ``param``.

        The derivative at parameter ``phi`` is given by ::

            deriv = radius * (sin(phi) * center_dir + cos(phi) * tangent_at_0)

        Parameters
        ----------
        param : float or `array-like`
            Parameter value(s) at which to evaluate.

        Returns
        -------
        deriv : `numpy.ndarray`, shape (2,) or (num_params, 2)
            Vector(s) representing the detector surface derivative at
            ``param``.
            If ``param`` is a single parameter, a single vector is
            returned, otherwise a stack of vectors along axis 0.

        See Also
        --------
        surface

        Examples
        --------
        The method works with a single parameter, resulting in a single
        vector:

        >>> part = odl.uniform_partition(-np.pi / 2, np.pi / 2, 10)
        >>> det = CircleSectionDetector(part, center=[0, -2])
        >>> det.surface_deriv(0)
        array([ 2.,  0.])
        >>> np.allclose(det.surface_deriv(-np.pi / 2), [0, 2])
        True
        >>> np.allclose(det.surface_deriv(np.pi / 2), [0, -2])
        True

        It is also vectorized, i.e., it can be called with multiple
        parameters at once:

        >>> deriv = det.surface_deriv([-np.pi / 2, 0, np.pi / 2])
        >>> np.allclose(deriv, [[0, 2],
        ...                     [2, 0],
        ...                     [0, -2]])
        True
        """
        squeeze_out = np.isscalar(param)
        param = np.array(param, dtype=float, copy=False, ndmin=1)
        if self.check_bounds and not self.params.contains_all(param):
            raise ValueError('`param` {} not in the valid range '
                             '{}'.format(param, self.params))
        deriv = (np.sin(param)[:, None] * self.center_dir[None, :] +
                 np.cos(param)[:, None] * self.tangent_at_0[None, :])
        deriv *= self.radius
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
        deriv : float or `numpy.ndarray`
            Constant value(s) of the arc length measure at ``param``.
            If ``param`` is a single parameter, a float is returned,
            otherwise a vector of the same length as ``param``.

        See Also
        --------
        surface
        surface_deriv

        Examples
        --------
        The method works with a single parameter, resulting in a float:

        >>> part = odl.uniform_partition(-np.pi / 2, np.pi / 2, 10)
        >>> det = CircleSectionDetector(part, center=[0, -2])
        >>> det.surface_measure(0)
        2.0
        >>> det.surface_measure(np.pi / 2)
        2.0

        It is also vectorized, i.e., it can be called with multiple
        parameters at once:

        >>> det.surface_measure([0, np.pi / 2])
        array([ 2.,  2.])
        """
        scalar_out = np.isscalar(param)
        param = np.array(param, dtype=float, copy=False, ndmin=1)
        if self.check_bounds and not self.params.contains_all(param):
            raise ValueError('`param` {} not in the valid range '
                             '{}'.format(param, self.params))

        if scalar_out:
            return self.radius
        else:
            return self.radius * np.ones(param.shape[0])

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.partition]
        optargs = [('center', self.center.tolist(), None)]
        inner_str = signature_string(posargs, optargs, sep=',\n')
        return '{}(\n{}\n)'.format(self.__class__.__name__,
                                   indent_rows(inner_str))

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
