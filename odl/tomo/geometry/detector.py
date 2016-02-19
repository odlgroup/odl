# Copyright 2014, 2015 The ODL development group
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

"""Detectors."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from abc import ABCMeta, abstractmethod
from future import standard_library
standard_library.install_aliases()
from builtins import object, super

# External
import numpy as np

# Internal
from odl.util.utility import with_metaclass
from odl.discr.partition import RectPartition

__all__ = ('Detector', 'FlatDetector', 'Flat1dDetector', 'Flat2dDetector',
           'CircleSectionDetector')


class Detector(with_metaclass(ABCMeta, object)):

    """Abstract detector class.

    A detector is described by

    * a dimension parameter,
    * a set of parameters for surface parametrization,
    * a function mapping motion and surface parameters to the location
      of a detector point relative to the reference point,
    * optionally a surface measure function defined on the surface
      parametrization parameters only
    * optionally a sampling grid for the parameters
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
            raise TypeError('partition {!r} is not a RectPartition instance.'
                            ''.format(part))

        self._part = part

    @abstractmethod
    def surface(self, param):
        """The parametrization of the detector reference surface.

        Parameters
        ----------
        param : element of `params`
            The parameter value where to evaluate the function
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
        """The sampling grid for the parameters."""
        return self.partition.grid

    @property
    def shape(self):
        """The shape of the detector grid."""
        return self.partition.shape

    @property
    def size(self):
        """The total number of pixels (sampling points)."""
        return self.partition.size

    def surface_deriv(self, param):
        """The partial derivative(s) of the surface parametrization.

        Parameters
        ----------
        param : element of `params`
            The parameter value where to evaluate the function
        """
        raise NotImplementedError

    def surface_measure(self, param):
        """The density function of the surface measure.

        This is the default implementation relying on the `surface_deriv`
        method. For ``ndim == 1``, the density is given by the `Arc
        length`_, for ``ndim == 2``, it is the length of the cross product
        of the partial derivatives of the parametrization, see Wikipedia's
        `Surface area`_ article.

        Parameters
        ----------
        param : element of `params`
            The parameter value where to evaluate the function

        Returns
        -------
        measure : `float`
            The density value at the given parameter

        .. _Arc length:
            https://en.wikipedia.org/wiki/Curve#Lengths_of_curves
        .. _Surface area:
            https://en.wikipedia.org/wiki/Surface_area
        """
        if param not in self.params:
            raise ValueError('parameter value {} not in the valid range {}.'
                             ''.format(param, self.params))
        if self.ndim == 1:
            return float(np.linalg.norm(self.surface_deriv(param)))
        elif self.ndim == 2:
            return float(np.linalg.norm(np.cross(*self.surface_deriv(param))))
        else:
            raise NotImplementedError


class FlatDetector(Detector):

    """Abstract class for flat detectors in 2 and 3 dimensions."""

    def surface_measure(self, param):
        """The constant density function of the surface measure.

        Parameters
        ----------
        param : element of `params`
            The parameter value where to evaluate the function

        Returns
        -------
        measure : `float`
            The constant density 1.0
        """
        if param not in self.params:
            raise ValueError('parameter value {} not in the valid range '
                             '{}.'.format(param, self.params))
        return 1.0

    def __repr__(self):
        """Return ``repr(self)``."""
        # TODO: adapt to partitions
        inner_fstr = '{!r},\n grid={grid!r}'
        inner_str = inner_fstr.format(self.params, grid=self.param_grid)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """Return ``str(self)``."""
        # TODO: prettify
        # TODO: adapt to partitions
        inner_fstr = '{},\n grid={grid}'
        inner_str = inner_fstr.format(self.params, grid=self.param_grid)
        return '{}({})'.format(self.__class__.__name__, inner_str)


class Flat1dDetector(FlatDetector):

    """A 1d line detector aligned with the ``detector_axis``."""

    def __init__(self, part, detector_axis):
        """Initialize a new instance.

        Parameters
        ----------
        part : 1-dim. `RectPartition`
            Partition of the parameter interval, corresponding to the
            line elements
        detector_axis : 2-element `array-like`
            Unit direction along the detector parameter of the detector.
        """
        super().__init__(part)
        if self.ndim != 1:
            raise ValueError('expected partition to be 1-dimensional ,'
                             'got {}.'.format(self.ndim))

        self._detector_axis = np.asarray(detector_axis)
        if np.linalg.norm(self.detector_axis) <= 1e-10:
            raise ValueError('detector axis vector {} too close to zero.'
                             ''.format(detector_axis))

    @property
    def detector_axis(self):
        """The direction of the principal axis of the detector."""
        return self._detector_axis

    def surface(self, param):
        """The parametrization of the (1d) detector reference surface.

        The reference line segment is chosen to be aligned with the
        second coordinate axis, such that the parameter value 0 results
        in the reference point (0, 0).

        Parameters
        ----------
        param : element of `params`
            The parameter value where to evaluate the function

        Returns
        -------
        point : `numpy.ndarray`, shape (2,)
            The point on the detector surface corresponding to the
            given parameters
        """
        if param not in self.params:
            raise ValueError('parameter value {} not in the valid range '
                             '{}.'.format(param, self.params))
        return self.detector_axis * float(param)

    def surface_deriv(self, param=None):
        """The derivative of the surface parametrization.

        Parameters
        ----------
        param : element of `params`, optional
            The parameter value where to evaluate the function

        Returns
        -------
        derivative : `numpy.ndarray`, shape (2,)
            The constant derivative
        """
        if param is not None and param not in self.params:
            raise ValueError('parameter value {} not in the valid range '
                             '{}.'.format(param, self.params))
        return self.detector_axis


class Flat2dDetector(FlatDetector):

    """A 2d flat panel detector aligned with the ``detector_axes``."""

    def __init__(self, part, detector_axes):
        """Initialize a new instance.

        Parameters
        ----------
        part : 1-dim. `RectPartition`
            Partition of the parameter interval, corresponding to the
            pixels
        detector_axes : `sequence` of two 3-element `array-like`
            The directions of the axes of the detector
            Example: [(0, 1, 0), (0, 0, 1)]
        """
        super().__init__(part)
        if self.ndim != 2:
            raise ValueError('expected partition to be 2-dimensional ,'
                             'got {}.'.format(self.ndim))

        self._detector_axes = (np.asarray(detector_axes[0]),
                               np.asarray(detector_axes[1]))

        if np.linalg.norm(self.detector_axes[0]) <= 1e-10:
            raise ValueError('first axis vector {} too close to zero.'
                             ''.format(detector_axes[0]))
        if np.linalg.norm(self.detector_axes[1]) <= 1e-10:
            raise ValueError('second axis vector {} too close to zero.'
                             ''.format(detector_axes[1]))

    @property
    def detector_axes(self):
        """The directions of the principal axes of the detector."""
        return self._detector_axes

    def surface(self, param):
        """The parametrization of the (2d) detector reference surface.

        The reference plane segment is chosen to be aligned with the
        second and third coordinate axes, in this order, such that
        the parameter value (0, 0) results in the reference (0, 0, 0).

        Parameters
        ----------
        param : element of `params`
            The parameter value where to evaluate the function

        Returns
        -------
        point : `numpy.ndarray`, shape (3,)
            The point on the detector surface corresponding to the
            given parameters
        """
        if param not in self.params:
            raise ValueError('parameter value {} not in the valid range '
                             '{}.'.format(param, self.params))

        return (self.detector_axes[0] * float(param[0]) +
                self.detector_axes[1] * float(param[1]))

    def surface_deriv(self, param=None):
        """The derivative of the surface parametrization.

        Parameters
        ----------
        param : element of `params`, optional
            The parameter value where to evaluate the function

        Returns
        -------
        derivatives : 2-tuple of ndarray with shape (3,)
            The constant partial derivatives, where each axis "points" in
            space.
        """
        if param is not None and param not in self.params:
            raise ValueError('parameter value {} not in the valid range '
                             '{}.'.format(param, self.params))
        return self.detector_axes


class CircleSectionDetector(Detector):

    """A 1d detector lying on a section of a circle.

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
        circ_rad : positive `float`
            Radius of the circle on which the detector is situated
        """
        super().__init__(part)
        if self.ndim != 1:
            raise ValueError('expected partition to be 1-dimensional ,'
                             'got {}.'.format(self.ndim))

        self._circ_rad = float(circ_rad)
        if self.circ_rad <= 0:
            raise ValueError('circle radius {} is not positive.'
                             ''.format(circ_rad))

    @property
    def circ_rad(self):
        """Circle radius of this detector."""
        return self._circ_rad

    def surface(self, param):
        """The parametrization of the detector reference surface.

        Parameters
        ----------
        param : element of `params`
            The parameter value where to evaluate the function
        """
        if (param in self.params or
            (param.contains_all(np.ndarray) and
             all(par in self.params for par in param))):
            return (self.circ_rad *
                    np.array([np.cos(param) - 1, np.sin(param)]).T)
        else:
            raise ValueError('parameter value(s) {} not in the valid range '
                             '{}.'.format(param, self.params))

    def surface_deriv(self, param):
        """The partial derivative(s) of the surface parametrization.

        Parameters
        ----------
        param : element of `params`
            The parameter value where to evaluate the function
        """
        if (param in self.params or (param.contains_all(np.ndarray) and all(
                par in self.params for par in param))):
            return self.circ_rad * np.array([-np.sin(param), np.cos(param)]).T
        else:
            raise ValueError('parameter value(s) {} not in the valid range '
                             '{}.'.format(param, self.params))

    def surface_measure(self, param):
        """The constant density function of the surface measure.

        Parameters
        ----------
        param : element of `params`
            The parameter value where to evaluate the function

        Returns
        -------
        measure : `float`
            The constant density ``r``, equal to the length of the
            tangent to the detector circle at any point
        """
        if param in self.params:
            return self.circ_rad
        elif (isinstance(param, np.ndarray) and
              all(par in self.params for par in param)):
            return self.circ_rad * np.ones_like(param, dtype=float)
        else:
            raise ValueError('parameter value(s) {} not in the valid range '
                             '{}.'.format(param, self.params))

    def __repr__(self):
        """Returns ``repr(self)``."""
        # TODO: adapt to partitions
        inner_fstr = '{!r}, {},\n grid={grid!r}'
        inner_str = inner_fstr.format(self.params, self.circ_rad,
                                      grid=self.param_grid)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """d.__str__() <==> str(d)."""
        # TODO: adapt to partitions
        # TODO: prettify
        inner_fstr = '{}, {},\n grid={grid}'
        inner_str = inner_fstr.format(self.params, self.circ_rad,
                                      grid=self.param_grid)
        return '{}({})'.format(self.__class__.__name__, inner_str)
