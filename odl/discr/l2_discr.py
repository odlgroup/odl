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

"""Discretizations of default spaces."""

# pylint: disable=abstract-method

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import super, str
import numpy as np

# External

# ODL
from odl.discr.discretization import Discretization
from odl.discr.discretization import dspace_type
from odl.discr.discr_mappings import GridCollocation, NearestInterpolation
from odl.space.default import L2
from odl.sets.domain import IntervalProd


__all__ = ('DiscreteL2', 'l2_uniform_discretization')

_supported_interp = ('nearest',)


class DiscreteL2(Discretization):

    """Discretization of an :math:`L^2` space."""

    def __init__(self, l2space, grid, dspace, interp='nearest', **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        l2space : ``L2``
            The continuous space to be discretized
        dspace : ``FnBase``, same `field` as `l2space`
            The space of elements used for data storage
        grid : ``TensorGrid``
            The sampling grid for the discretization. Must be contained
            in `l2space.domain`.
        interp : string, optional
            The interpolation type to be used for discretization.

            'nearest' : use nearest-neighbor interpolation (default)

            'linear' : use linear interpolation (not implemented)
        kwargs : {'order'}
            'order' : {'C', 'F'}, optional  (Default: 'C')
                Ordering of the values in the flat data arrays. 'C'
                means the first grid axis varies fastest, the last most
                slowly, 'F' vice versa.
        """
        if not isinstance(l2space, L2):
            raise TypeError('{} is not an `L2` type space.'.format(l2space))

        if not isinstance(l2space.domain, IntervalProd):
            raise TypeError('l2space.domain {} is not an `IntervalProd`.'.format(l2space.domain))

        interp = str(interp)
        if interp not in _supported_interp:
            raise TypeError('{} is not among the supported interpolation'
                            'types {}.'.format(interp, _supported_interp))

        self._order = kwargs.pop('order', 'C')
        restriction = GridCollocation(l2space, grid, dspace, order=self.order)
        if interp == 'nearest':
            extension = NearestInterpolation(l2space, grid, dspace,
                                             order=self.order)
        else:
            raise NotImplementedError

        super().__init__(l2space, dspace, restriction, extension)
        self._interp = interp

    def element(self, inp=None):
        """Create an element from `inp` or from scratch.

        Parameters
        ----------
        inp : `object`, optional
            The input data to create an element from. Must be
            recognizable by the `element()` method of either `dspace`
            or `uspace`.

        Returns
        -------
        element : ``DiscreteL2.Vector``
            The discretized element, calculated as
            `dspace.element(inp)` or
            `restriction(uspace.element(inp))`, tried in this order.
        """
        if inp is None:
            return self.Vector(self, self.dspace.element())
        elif inp in self.dspace:
            return self.Vector(self, inp)
        elif inp in self.uspace:
            return self.Vector(
                self, self.restriction(self.uspace.element(inp)))
        else:  # Sequence-type input
            arr = np.asarray(inp, dtype=self.dtype, order=self.order)
            if arr.shape != self.grid.shape:
                raise ValueError('inp.shape {} does not match space.grid.shape {}'.format(arr.shape, self.grid.shape))
            arr = arr.flatten(order=self.order)
            return self.Vector(self, self.dspace.element(arr))

    @property
    def grid(self):
        """Sampling grid of the discretization mappings."""
        return self.restriction.grid
    
    @property
    def order(self):
        """Axis ordering for array flattening."""
        return self._order

    @property
    def interp(self):
        """Interpolation type of this discretization."""
        return self._interp

    def points(self):
        return self.grid.points(order=self.order)

    def __repr__(self):
        """l2.__repr__() <==> repr(l2)."""
        # Check if the factory repr can be used
        if (self.uspace.domain.uniform_sampling(
                self.grid.shape, as_midp=True) == self.grid):
            if dspace_type(self.uspace, 'numpy') == self.dspace_type:
                impl = 'numpy'
            elif dspace_type(self.uspace, 'cuda') == self.dspace_type:
                impl = 'cuda'
            else:  # This should never happen
                raise RuntimeError('unable to determine data space impl.')
            arg_fstr = '{!r}, {!r}'
            if self.interp != 'nearest':
                arg_fstr += ', interp={interp!r}'
            if impl != 'numpy':
                arg_fstr += ', impl={impl!r}'
            if self.order != 'C':
                arg_fstr += ', order={order!r}'
            return 'l2_uniform_discretization({})'.format(arg_fstr.format(
                self.uspace, self.grid.shape, interp=self.interp,
                impl=impl, order=self.order))
        else:
            arg_fstr = '''
    {!r},
    {!r},
    {!r}'''
            if self.interp != 'nearest':
                arg_fstr += ', interp={interp!r}'
            if self.order != 'C':
                arg_fstr += ', order={order!r}'

            return 'DiscreteL2({})'.format(arg_fstr.format(
                self.uspace, self.grid, self.dspace, interp=self.interp,
                order=self.order))

    def __str__(self):
        """l2.__str__() <==> str(l2)."""
        return self.__repr__()

    class Vector(Discretization.Vector):

        """Representation of a ``DiscreteL2`` element."""

        def asarray(self, out=None):
            """Extract the data of this array as a numpy array.

            Parameters
            ----------
            out : `ndarray`, Optional (default: `None`)
                Array in which the result should be written in-place.
                Has to be contiguous and of the correct dtype and
                shape.
            """
            if out is None:
                return super().asarray().reshape(self.space.grid.shape,
                                                 order=self.space.order)
            else:
                if out.shape not in (self.space.grid.shape,
                                     (self.space.grid.ntotal,)):
                    raise ValueError('output array has shape {}, expected '
                                     '{} or ({},).'
                                     ''.format(out.shape,
                                               self.space.grid.shape,
                                               self.space.grid.ntotal))
                out_r = out.reshape(self.space.grid.shape,
                                    order=self.space.order)
                if out_r.flags.c_contiguous:
                    out_order = 'C'
                elif out_r.flags.f_contiguous:
                    out_order = 'F'
                else:
                    raise ValueError('output array not contiguous.')

                if out_order != self.space.order:
                    raise ValueError('output array has ordering {!r}, '
                                     'expected {!r}.'
                                     .format(self.space.order, out_order))

                super().asarray(out=out.reshape(-1, order=self.space.order))
                return out


def l2_uniform_discretization(l2space, nsamples, interp='nearest',
                              impl='numpy', **kwargs):
    """Discretize an L2 space by uniform sampling.

    Parameters
    ----------
    l2space : ``L2``
        Continuous L2 type space. Its domain must be an
        ``IntervalProd`` instance.
    nsamples : int or tuple of int
        Number of samples per axis. For dimension >= 2, a tuple is
        required.
    interp : string, optional
            Interpolation type to be used for discretization.

            'nearest' : use nearest-neighbor interpolation (default)

            'linear' : use linear interpolation (not implemented)
    impl : {'numpy', 'cuda'}
        Implementation of the data storage arrays
    kwargs : {'order', 'dtype'}
            'order' : 'C' or 'F', optional  (Default: 'C')
                Axis ordering in the data storage
            'dtype' : type, optional  (Default: depends on `impl`)
                Data type for the discretized space

                Default for 'numpy': 'float64' / 'complex128'
                Default for 'cuda': 'float32' / TODO

    Returns
    -------
    l2discr : ``DiscreteL2``
        The uniformly discretized L2 space
    """
    if not isinstance(l2space, L2):
        raise TypeError('space {!r} is not an L2 instance.'.format(l2space))

    if not isinstance(l2space.domain, IntervalProd):
        raise TypeError('domain {!r} of the L2 space is not an `IntervalProd` '
                        'instance.'.format(l2space.domain))

    ds_type = dspace_type(l2space, impl)
    dtype = kwargs.pop('dtype', None)

    grid = l2space.domain.uniform_sampling(nsamples, as_midp=True)
    if dtype is not None:
        dspace = ds_type(grid.ntotal, dtype=dtype)
    else:
        dspace = ds_type(grid.ntotal)

    order = kwargs.pop('order', 'C')

    return DiscreteL2(l2space, grid, dspace, interp=interp, order=order)
