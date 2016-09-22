# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Detectors for tomographic imaging."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import object, super

from abc import ABCMeta, abstractmethod
import numpy as np

from odl.discr import RectPartition
from odl.tomo.util.utility import perpendicular_vector
from odl.util.utility import with_metaclass


__all__ = ('Detector', 'FlatDetector', 'Flat1dDetector', 'Flat2dDetector',
           'CircleSectionDetector')


class Detector(with_metaclass(ABCMeta, object)):

    """Abstract detector class.

    A detector is described by

    * a set of parameters for surface parametrization (including sampling),
    * a function mapping a surface parameter to the location of a detector
      point relative to its reference point,
    * optionally a surface measure function.
    """

    def __init__(self, part):
        """Initialize a new instance.

        Parameters
        ----------
        part : `RectPartition`
           Partition of the detector parameter set (pixelization).
           It determines dimension, parameter range and discretization.
        """
        if not isinstance(part, RectPartition):
            raise TypeError('`part` {!r} is not a RectPartition instance'
                            ''.format(part))

        self._part = part

    @abstractmethod
    def surface(self, param):
        """Parametrization of the detector reference surface.

        Parameters
        ----------
        param : `params` element
            Parameter value where to evaluate the function

        Returns
        -------
        point :
            Spatial location of the detector point corresponding to
            ``param``
        """

    @property
    def partition(self):
        """Partition of the detector parameter set into subsets."""
        return self._part

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

    def surface_deriv(self, param):
        """Partial derivative(s) of the surface parametrization.

        Parameters
        ----------
        param : `params` element
            The parameter value where to evaluate the function

        Returns
        -------
        deriv :
            Vector (``ndim=1``) or sequence of vectors corresponding
            to the partial derivatives at ``param``
        """
        raise NotImplementedError

    def surface_measure(self, param):
        """Density function of the surface measure.

        This is the default implementation relying on the `surface_deriv`
        method. For ``ndim == 1``, the density is given by the `Arc
        length`_, for ``ndim == 2``, it is the length of the cross product
        of the partial derivatives of the parametrization, see Wikipedia's
        `Surface area`_ article.

        Parameters
        ----------
        param : `params` element
            The parameter value where to evaluate the function

        Returns
        -------
        measure : float
            The density value at the given parameter

        .. _Arc length:
            https://en.wikipedia.org/wiki/Curve#Lengths_of_curves
        .. _Surface area:
            https://en.wikipedia.org/wiki/Surface_area
        """
        if param not in self.params:
            raise ValueError('`param` {} not in the valid range {}'
                             ''.format(param, self.params))
        if self.ndim == 1:
            return float(np.linalg.norm(self.surface_deriv(param)))
        elif self.ndim == 2:
            return float(np.linalg.norm(np.cross(*self.surface_deriv(param))))
        else:
            raise NotImplementedError


class FlatDetector(Detector):

    """Abstract class for flat detectors in 2 and 3 dimensions."""

    def surface_measure(self, param=None):
        """Constant density function of the surface measure.

        Parameters
        ----------
        param : `params` element, optional
            Parameter value where to evaluate the function

        Returns
        -------
        measure : float
            Constant density 1.0
        """
        if param not in self.params:
            raise ValueError('`param` {} not in the valid range '
                             '{}'.format(param, self.params))
        # TODO: apart from being constant, there is no big simplification
        # in this method compared to parent. Consider removing FlatDetector
        # altogether.
        return super().surface_measure(self.params.min_pt)


class Flat1dDetector(FlatDetector):

    """A 1d line detector aligned with ``axis``."""

    def __init__(self, part, axis):
        """Initialize a new instance.

        Parameters
        ----------
        part : 1-dim. `RectPartition`
            Partition of the parameter interval, corresponding to the
            line elements
        axis : `array-like`, shape ``(2,)``
            Principal axis of the detector
        """
        super().__init__(part)
        if self.ndim != 1:
            raise ValueError('expected partition to have 1 dimension, '
                             'got {}'.format(self.ndim))

        if np.linalg.norm(axis) <= 1e-10:
            raise ValueError('`axis` vector {} too close to zero'
                             ''.format(axis))
        self._axis = np.asarray(axis) / np.linalg.norm(axis)
        self._normal = perpendicular_vector(self.axis)

    @property
    def axis(self):
        """Normalized principal axis of the detector."""
        return self._axis

    @property
    def normal(self):
        """Unit vector perpendicular to the detector.

        Its orientation is chosen such that the system ``axis, normal``
        is right-handed.
        """
        return self._normal

    def surface(self, param):
        """Parametrization of the (1d) detector reference surface.

        The reference line segment is chosen to be aligned with the
        second coordinate axis, such that the parameter value 0 results
        in the reference point (0, 0).

        Parameters
        ----------
        param : `params` element
            The parameter value where to evaluate the function

        Returns
        -------
        point : `numpy.ndarray`, shape (2,)
            The point on the detector surface corresponding to the
            given parameters
        """
        param = float(param)
        if param not in self.params:
            raise ValueError('`param` {} not in the valid range '
                             '{}'.format(param, self.params))
        return self.axis * param

    def surface_deriv(self, param=None):
        """Derivative of the surface parametrization.

        Parameters
        ----------
        param : `params` element, optional
            The parameter value where to evaluate the function

        Returns
        -------
        derivative : `numpy.ndarray`, shape (2,)
            The constant derivative
        """
        if param is not None and param not in self.params:
            raise ValueError('`param` {} not in the valid range '
                             '{}'.format(param, self.params))
        return self.axis

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_fstr = '\n    {!r},\n    {!r}'
        inner_str = inner_fstr.format(self.partition, self.axis)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """Return ``str(self)``."""
        # TODO: prettify
        inner_fstr = '\n    {},\n    {}'
        inner_str = inner_fstr.format(self.partition, self.axis)
        return '{}({})'.format(self.__class__.__name__, inner_str)


class Flat2dDetector(FlatDetector):

    """A 2d flat panel detector aligned with ``axes``."""

    def __init__(self, part, axes):
        """Initialize a new instance.

        Parameters
        ----------
        part : 1-dim. `RectPartition`
            Partition of the parameter interval, corresponding to the
            pixels
        axes : 2-tuple of `array-like`'s (shape ``(3,)``)
            Principal axes of the detector, e.g.
            ``[(0, 1, 0), (0, 0, 1)]``
        """
        super().__init__(part)
        if self.ndim != 2:
            raise ValueError('expected partition to have 2 dimensions, '
                             'got {}'.format(self.ndim))

        for i, a in enumerate(axes):
            if np.linalg.norm(a) <= 1e-10:
                raise ValueError('axis vector {} {} too close to zero'
                                 ''.format(i, axes[i]))
            if np.shape(a) != (3,):
                raise ValueError('axis vector {} has shape {}. expected (3,)'
                                 ''.format(i, np.shape(a)))

        self._axes = tuple(np.asarray(a) / np.linalg.norm(a) for a in axes)
        self._normal = np.cross(self.axes[0], self.axes[1])

        if np.linalg.norm(self.normal) <= 1e-4:
            raise ValueError('`axes` are almost parallel (norm of normal = '
                             '{})'.format(np.linalg.norm(self.normal)))

    @property
    def axes(self):
        """Normalized principal axes of this detector as a 2-tuple."""
        return self._axes

    @property
    def normal(self):
        """Unit vector perpendicular to this detector.

        The orientation is chosen such that the triple
        ``axes[0], axes[1], normal`` form a right-hand system.
        """
        return self._normal

    def surface(self, param):
        """Parametrization of the 2d detector reference surface.

        The reference plane segment is chosen to be aligned with the
        second and third coordinate axes, in this order, such that
        the parameter value (0, 0) results in the reference (0, 0, 0).

        Parameters
        ----------
        param : `params` element
            The parameter value where to evaluate the function

        Returns
        -------
        point : `numpy.ndarray`, shape (3,)
            The point on the detector surface corresponding to the
            given parameters
        """
        if param not in self.params:
            raise ValueError('`param` {} not in the valid range '
                             '{}'.format(param, self.params))

        return sum(float(p) * ax for p, ax in zip(param, self.axes))

    def surface_deriv(self, param=None):
        """Derivative of the surface parametrization.

        Parameters
        ----------
        param : `params` element, optional
            The parameter value where to evaluate the function

        Returns
        -------
        derivatives : 2-tuple of `numpy.ndarray`'s (shape ``(3,)``)
            The constant partial derivatives given by the detector axes
        """
        if param is not None and param not in self.params:
            raise ValueError('`param` {} not in the valid range '
                             '{}'.format(param, self.params))
        return self.axes

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_fstr = '\n    {!r},\n    {!r}'
        inner_str = inner_fstr.format(self.partition, self.axes)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """Return ``str(self)``."""
        # TODO: prettify
        inner_fstr = '\n    {},\n    {}'
        inner_str = inner_fstr.format(self.partition, self.axes)
        return '{}({})'.format(self.__class__.__name__, inner_str)


class CircleSectionDetector(Detector):

    """A 1d detector given by a section of a circle.

    The reference circular section is part of a circle with radius ``r``,
    which is shifted by the vector ``(-r, 0)`` such that the parameter
    value 0 results in the detector reference point ``(0, 0)``.
    """

    def __init__(self, part, circ_rad):
        """Initialize a new instance.

        Parameters
        ----------
        part : 1-dim. `RectPartition`
            Partition of the parameter interval, corresponding to the
            angle sections along the line
        circ_rad : positive float
            Radius of the circle along which the detector is curved
        """
        super().__init__(part)
        if self.ndim != 1:
            raise ValueError('expected `part` to have 1 dimension, '
                             'got {}'.format(self.ndim))

        self._circ_rad, circ_rad_in = float(circ_rad), circ_rad
        if self.circ_rad <= 0:
            raise ValueError('`circ_rad` {} is not positive'
                             ''.format(circ_rad_in))

    @property
    def circ_rad(self):
        """Circle radius of this detector."""
        return self._circ_rad

    def surface(self, param):
        """Parametrization of the detector reference surface.

        Parameters
        ----------
        param : `params` element
            The parameter value where to evaluate the function
        """
        if param in self.params or self.params.contains_all(param):
            return (self.circ_rad *
                    np.array([np.cos(param) - 1, np.sin(param)]).T)
        else:
            raise ValueError('`param` value(s) {} not in the valid range '
                             '{}'.format(param, self.params))

    def surface_deriv(self, param):
        """Partial derivative(s) of the surface parametrization.

        Parameters
        ----------
        param : `params` element
            The parameter value where to evaluate the function
        """
        if param in self.params or self.params.contains_all(param):
            return self.circ_rad * np.array([-np.sin(param), np.cos(param)]).T
        else:
            raise ValueError('`param` value(s) {} not in the valid range '
                             '{}'.format(param, self.params))

    def surface_measure(self, param):
        """Constant density function of the surface measure.

        Parameters
        ----------
        param : `params` element
            The parameter value where to evaluate the function

        Returns
        -------
        measure : float
            The constant density ``r``, equal to the length of the
            tangent to the detector circle at any point
        """
        if param in self.params:
            return self.circ_rad
        elif self.params.contains_all(param):
            return self.circ_rad * np.ones_like(param, dtype=float)
        else:
            raise ValueError('`param` value(s) {} not in the valid range '
                             '{}'.format(param, self.params))

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_fstr = '\n    {!r},\n    {}'
        inner_str = inner_fstr.format(self.partition, self.circ_rad)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """Return ``str(self)``."""
        # TODO: prettify
        inner_fstr = '\n    {},\n    {}'
        inner_str = inner_fstr.format(self.partition, self.circ_rad)
        return '{}({})'.format(self.__class__.__name__, inner_str)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
