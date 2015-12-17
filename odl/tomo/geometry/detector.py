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
from abc import ABCMeta, abstractmethod, abstractproperty
from future import standard_library
standard_library.install_aliases()
from future.utils import with_metaclass
from builtins import object, super

# External
import numpy as np

# Internal
from odl.set.domain import IntervalProd
from odl.discr.grid import TensorGrid


__all__ = ('Detector', 'LineDetector', 'Flat2dDetector',
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

    def __init__(self, params, grid=None):
        """Initialize a new instance.

        Parameters
        ----------
        params : `IntervalProd`
            The parameter set defining the detector area
        grid : `TensorGrid`, optional
            A sampling grid for the parameter set, in which it must be
            contained
        """
        if not isinstance(params, IntervalProd):
            raise TypeError('parameter set {} is not a an interval product.'
                            ''.format(params))

        if grid is not None:
            if not isinstance(grid, TensorGrid):
                raise TypeError('grid {} is not a `TensorGrid` instance.'
                                ''.format(grid))
            if not params.contains_set(grid):
                raise ValueError('grid {} not contained in parameter set {}.'
                                 ''.format(grid, params))

        self._params = params
        self._param_grid = grid

    @abstractproperty
    def ndim(self):
        """The number of dimensions of the detector (0, 1 or 2)."""

    @abstractmethod
    def surface(self, param):
        """The parametrization of the detector reference surface.

        Parameters
        ----------
        param : element of `params`
            The parameter value where to evaluate the function
        """

    @property
    def params(self):
        """Surface parameter set of this geometry."""
        return self._params

    @property
    def param_grid(self):
        """The sampling grid for the parameters."""
        return self._param_grid

    @property
    def has_sampling(self):
        """Return `True` if a sampling grid is given, else `False`."""
        return self.param_grid is not None

    @property
    def npixels(self):
        """The number of pixels (sampling points)."""
        if not self.has_sampling:
            raise ValueError('no sampling defined for {}.'.format(self))
        return self.param_grid.shape

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


class FlatDetector(with_metaclass(ABCMeta, Detector)):

    """Abstract class for flat detectors in 2 and 3 dimensions."""

    def surface_measure(self, param):
        """The constant density function of the surface measure.

        Parameters
        ----------
        param : element of `params`
            The parameter value where to evaluate the function

        Returns
        -------
        meas : `float`
            The constant density 1.0
        """
        if param not in self.params:
            raise ValueError('parameter value {} not in the valid range '
                             '{}.'.format(param, self.params))
        return 1.0

    def __repr__(self):
        """d.__repr__() <==> repr(d)."""
        inner_fstr = '{!r}'
        if self.has_sampling:
            inner_fstr += ',\n grid={grid!r}'
        inner_str = inner_fstr.format(self.params, grid=self.param_grid)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """d.__str__() <==> str(d)."""
        # TODO: prettify
        inner_fstr = '{}'
        if self.has_sampling:
            inner_fstr += ',\n grid={grid}'
        inner_str = inner_fstr.format(self.params, grid=self.param_grid)
        return '{}({})'.format(self.__class__.__name__, inner_str)


# TODO: rename to Flat1dDetector to be consitent with Flat2dDetector? or
# Flat2dDetectorto to AreaDetector, or FlatLineDetector and FlatAreaDetctor
class LineDetector(FlatDetector):

    """A 1d line detector aligned with the y-axis."""

    def __init__(self, params, grid=None):
        """Initialize a new instance.

        Parameters
        ----------
        params : `Interval` or 1-dim. `IntervalProd`
            The range of the parameters defining the detector area.
        grid : 1-dim. `TensorGrid`, optional
            A sampling grid for the parameter interval, in which it must
            be contained
        """
        super().__init__(params, grid)

        if params.ndim != 1:
            raise ValueError('parameters {} are not 1-dimensional.'
                             ''.format(params))

    @property
    def ndim(self):
        """The number of dimensions of the detector."""
        return 1

    @property
    def npixels(self):
        """The number of pixels (sampling points)."""
        if not self.has_sampling:
            raise ValueError('no sampling defined.')
        return self.param_grid.shape[0]

    def surface(self, param):
        """The parametrization of the (1d) detector reference surface.

        The reference line segment is chosen to be aligned with the
        second coordinate axis, such that the parameter value 0 results
        in the reference point :math:`(0, 0)`.

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
        return np.array([0, float(param)])

    def surface_deriv(self, param):
        """The derivative of the surface parametrization.
        Parameters
        ----------
        param : element of `params`
            The parameter value where to evaluate the function

        Returns
        -------
        deriv : `numpy.ndarray`, shape (2,)
            The constant derivative (0, 1)
        """
        if param not in self.params:
            raise ValueError('parameter value {} not in the valid range '
                             '{}.'.format(param, self.params))
        return np.array([0., 1.])


class Flat2dDetector(FlatDetector):

    """A 2d flat panel detector aligned with the y-z axes."""

    def __init__(self, params, grid=None):
        """Initialize a new instance.

        Parameters
        ----------
        params : `Rectangle` or 2-dim. `IntervalProd`
            The range of the parameters defining the detector area.
        grid : 2-dim. `TensorGrid`, optional
            A sampling grid for the parameter rectangle, in which it
            must be contained
        """
        super().__init__(params, grid)

        if params.ndim != 2:
            raise ValueError('parameters {} are not 2-dimensional.'
                             ''.format(params))

    @property
    def ndim(self):
        """The number of dimensions of the detector."""
        return 2

    def surface(self, param):
        """The parametrization of the (2d) detector reference surface.

        The reference plane segment is chosen to be aligned with the
        second and third coordinate axes, in this order, such that
        the parameter value :math:`(0, 0)` results in the reference
        point :math:`(0, 0, 0)`.

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
        return np.array([0, float(param[0]), float(param[1])])

    def surface_deriv(self, param):
        """The derivative of the surface parametrization.

        Parameters
        ----------
        param : element of `params`
            The parameter value where to evaluate the function

        Returns
        -------
        deriv : 2-tuple of ndarray with shape (3,)
            The constant partial derivatives (0, 1, 0), (0, 0, 1)
        """
        if param not in self.params:
            raise ValueError('parameter value {} not in the valid range '
                             '{}.'.format(param, self.params))
        return np.array([0., 1., 0.]), np.array([0., 0., 1.])


class CircleSectionDetector(Detector):

    """A 1d detector lying on a section of a circle.

    The reference cirular section is part of a circle with radius :math:`r`,
    which is shifted by the vector :math:`(-r, 0)`, such that the parameter
    value 0 results in the detector reference point :math:`(0, 0)`.

    """

    def __init__(self, params, circ_rad, grid=None):
        """Initialize a new instance.

        Parameters
        ----------
        params : `Interval` or 1-dim. `IntervalProd`
            The range of the parameters defining the detector area.
        circ_rad : positive `float`
            Radius of the circle on which the detector is situated
        grid : 1-dim. `TensorGrid`, optional
            A sampling grid for the parameter interval, in which it must
            be contained. Default: `None`
        """
        super().__init__(params, grid)

        if params.ndim != 1:
            raise ValueError('parameters {} are not 1-dimensional.'
                             ''.format(params))

        self._circ_rad = float(circ_rad)
        if self._circ_rad <= 0:
            raise ValueError('circle radius {} is not positive.'
                             ''.format(circ_rad))

    @property
    def circ_rad(self):
        """Circle radius of this detector."""
        return self._circ_rad

    @property
    def ndim(self):
        """The number of dimensions of the detector."""
        return 1

    @property
    def npixels(self):
        """The number of pixels (sampling points)."""
        if not self.has_sampling:
            raise ValueError('no sampling defined for {}.'.format(self))
        return self.param_grid.shape[0]

    def surface(self, param):
        """The parametrization of the detector reference surface."""
        if (param in self.params or
                (isinstance(param, np.ndarray) and
                 all(par in self.params for par in param))):
            return (self.circ_rad *
                    np.array([np.cos(param) - 1, np.sin(param)]).T)
        else:
            raise ValueError('parameter value(s) {} not in the valid range '
                             '{}.'.format(param, self.params))

    def surface_deriv(self, param):
        """The partial derivative(s) of the surface parametrization."""
        if (param in self.params or
                (isinstance(param, np.ndarray) and
                 all(par in self.params for par in param))):
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
        meas : `float`
            The constant density :math:`r`, equal to the length of the
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
        """d.__repr__() <==> repr(d)."""
        inner_fstr = '{!r}, {}'
        if self.has_sampling:
            inner_fstr += ',\n grid={grid!r}'
        inner_str = inner_fstr.format(self.params, self.circ_rad,
                                      grid=self.param_grid)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """d.__str__() <==> str(d)."""
        # TODO: prettify
        inner_fstr = '{}, {}'
        if self.has_sampling:
            inner_fstr += ',\n grid={grid}'
        inner_str = inner_fstr.format(self.params, self.circ_rad,
                                      grid=self.param_grid)
        return '{}({})'.format(self.__class__.__name__, inner_str)
