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

""":math:`L^p` type discretizations of function spaces."""

# TODO: write some introduction doc

# pylint: disable=abstract-method

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super, str

# External
import numpy as np

# ODL
from odl.discr.discretization import (Discretization, DiscretizationVector, 
                                      dspace_type)
from odl.discr.discr_mappings import GridCollocation, NearestInterpolation
from odl.discr.grid import uniform_sampling
from odl.set.domain import IntervalProd
from odl.space.ntuples import Fn
from odl.space.fspace import FunctionSpace
from odl.space import CUDA_AVAILABLE
if CUDA_AVAILABLE:
    from odl.space.cu_ntuples import CudaFn
else:
    CudaFn = type(None)

__all__ = ('DiscreteLp', 'uniform_discr')

_SUPPORTED_INTERP = ('nearest',)


# TODO: other types of discrete spaces

class DiscreteLp(Discretization):

    """Discretization of a Lebesgue :math:`L^p` space."""

    def __init__(self, fspace, grid, dspace, exponent=2.0, interp='nearest',
                 **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        fspace : `FunctionSpace`
            The continuous space to be discretized
        grid : `TensorGrid`
            The sampling grid for the discretization. Must be contained
            in ``fspace.domain``.
        dspace : `FnBase`
            Space of elements used for data storage. It must have the
            same `FnBase.field` as
            ``fspace``
        exponent : positive `float`, optional
            The parameter :math:`p` in :math:`L^p`. If the exponent is
            not equal to the default 2.0, the space has no inner
            product.
        interp : `str`, optional
            The interpolation type to be used for discretization.

            'nearest' : use nearest-neighbor interpolation (default)

            'linear' : use linear interpolation (not implemented)
        kwargs : {'order'}
            'order' : {'C', 'F'}, optional  (Default: 'C')
                Ordering of the values in the flat data arrays. 'C'
                means the first grid axis varies fastest, the last most
                slowly, 'F' vice versa.
        """
        if not isinstance(fspace, FunctionSpace):
            raise TypeError('{} is not a `FunctionSpace` instance.'
                            ''.format(fspace))
        if not isinstance(fspace.domain, IntervalProd):
            raise TypeError('Function space domain {} is not an `IntervalProd`'
                            ' instance.'.format(fspace.domain))

        self._interp = str(interp).lower()
        if self.interp not in _SUPPORTED_INTERP:
            raise TypeError('{} is not among the supported interpolation'
                            'types {}.'.format(interp, _SUPPORTED_INTERP))

        self._order = str(kwargs.pop('order', 'C')).upper()
        restriction = GridCollocation(fspace, grid, dspace, order=self.order)
        if self.interp == 'nearest':
            extension = NearestInterpolation(fspace, grid, dspace,
                                             order=self.order)
        else:
            raise NotImplementedError

        super().__init__(fspace, dspace, restriction, extension)

        self._exponent = float(exponent)
        if (hasattr(self.dspace, 'exponent') and
                self._exponent != dspace.exponent):
            raise ValueError('exponent {} not equal to data space exponent '
                             '{}.'.format(self._exponent, dspace.exponent))

    @property
    def exponent(self):
        """The exponent :math:`p` in :math:`L^p`."""
        return self._exponent

    def element(self, inp=None):
        """Create an element from ``inp`` or from scratch.

        Parameters
        ----------
        inp : `object`, optional
            The input data to create an element from. Must be
            recognizable by the `LinearSpace.element` method
            of either `RawDiscretization.dspace` or
            `RawDiscretization.uspace`.

        Returns
        -------
        element : `DiscreteLpVector`
            The discretized element, calculated as
            ``dspace.element(inp)`` or
            ``restriction(uspace.element(inp))``, tried in this order.
        """
        if inp is None:
            return self.element_type(self, self.dspace.element())
        elif inp in self.dspace:
            return self.element_type(self, inp)
        elif inp in self.uspace:
            return self.element_type(
                self, self.restriction(self.uspace.element(inp)))
        else:  # Sequence-type input
            arr = np.asarray(inp, dtype=self.dtype, order=self.order)
            if arr.ndim > 1 and arr.shape != self.grid.shape:
                raise ValueError('input shape {} does not match grid shape {}'
                                 ''.format(arr.shape, self.grid.shape))
            arr = arr.flatten(order=self.order)
            return self.element_type(self, self.dspace.element(arr))

    @property
    def grid(self):
        """Sampling grid of the discretization mappings."""
        return self.restriction.grid

    def points(self):
        """All points in the sampling grid."""
        return self.grid.points(order=self.order)

    @property
    def order(self):
        """Axis ordering for array flattening."""
        return self._order

    @property
    def interp(self):
        """Interpolation type of this discretization."""
        return self._interp

    def __repr__(self):
        """``lp.__repr__() <==> repr(lp).``"""
        # Check if the factory repr can be used
        if (uniform_sampling(self.uspace.domain, self.grid.shape,
                             as_midp=True) == self.grid):
            if isinstance(self.dspace, Fn):
                impl = 'numpy'
            elif isinstance(self.dspace, CudaFn):
                impl = 'cuda'
            else:  # This should never happen
                raise RuntimeError('unable to determine data space impl.')
            arg_fstr = '{!r}, {!r}'
            if self.exponent != 2.0:
                arg_fstr += ', exponent={ex}'
            if self.interp != 'nearest':
                arg_fstr += ', interp={interp!r}'
            if impl != 'numpy':
                arg_fstr += ', impl={impl!r}'
            if self.order != 'C':
                arg_fstr += ', order={order!r}'

            arg_str = arg_fstr.format(
                self.uspace, self.grid.shape, interp=self.interp,
                impl=impl, order=self.order)
            return 'uniform_discr({})'.format(arg_str)
        else:
            arg_fstr = '''
    {!r},
    {!r},
    {!r}
    '''
            if self.exponent != 2.0:
                arg_fstr += ', exponent={ex}'
            if self.interp != 'nearest':
                arg_fstr += ', interp={interp!r}'
            if self.order != 'C':
                arg_fstr += ', order={order!r}'

            arg_str = arg_fstr.format(
                self.uspace, self.grid, self.dspace, interp=self.interp,
                order=self.order, ex=self.exponent)
            return '{}({})'.format(self.__class__.__name__, arg_str)

    def __str__(self):
        """``lp.__str__() <==> str(lp)``."""
        return self.__repr__()

    @property
    def element_type(self):
        """ `DiscreteLpVector` """
        return DiscreteLpVector

class DiscreteLpVector(DiscretizationVector):

    """Representation of a `DiscreteLp` element."""

    def asarray(self, out=None):
        """Extract the data of this array as a numpy array.

        Parameters
        ----------
        out : `numpy.ndarray`, optional
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

            super().asarray(out=out.ravel(order=self.space.order))
            return out

    @property
    def ndim(self):
        """Number of dimensions."""
        return self.space.grid.ndim

    @property
    def shape(self):
        # override shape
        return self.space.grid.shape

    def __setitem__(self, indices, values):
        """Set values of this vector.

        Parameters
        ----------
        indices : `int` or `slice`
            The position(s) that should be set
        values : {scalar, array-like, `NtuplesVector`}
            The value(s) that are to be assigned.

            If ``indices`` is an `int`, ``values`` must be a single
            value.

            If ``indices`` is a `slice`, ``values`` must be
            broadcastable to the size of the slice (same size,
            shape ``(1,)`` or single value).
            For ``indices==slice(None, None, None)``, i.e. in the call
            ``vec[:] = values``, a multi-dimensional array of correct
            shape is allowed as ``values``.
        """
        if values in self.space:
            self.ntuple.__setitem__(indices, values.ntuple)
        else:
            if indices == slice(None, None, None):
                values = np.atleast_1d(values)
                if (values.ndim > 1 and
                        values.shape != self.space.grid.shape):
                    raise ValueError('shape {} of value array {} not equal'
                                        ' to sampling grid shape {}.'
                                        ''.format(values.shape, values,
                                                self.space.grid.shape))
                values = values.ravel(order=self.space.order)

            super().__setitem__(indices, values)

    def show(self, method='', title='', **kwargs):
        """Create a figure displaying the function in 1d or 2d.

        Parameters
        ----------
        method : `str`, optional
            1d methods:

            'plot' : graph plot

            2d methods:

            'imshow' : image plot with coloring according to value,
            including a colorbar.

            'scatter' : cloud of scattered 3d points
            (3rd axis <-> value)

            'wireframe', 'plot_wireframe' : surface plot

        title : `str`, optional
            Set the title of the figure
        kwargs : {'figsize', 'saveto', ...}
            Extra keyword arguments passed on to display method
            See the Matplotlib functions for documentation of extra
            options.

        See Also
        --------
        matplotlib.pyplot.plot : Show graph plot

        matplotlib.pyplot.imshow : Show data as image

        matplotlib.pyplot.scatter : Show scattered 3d points
        """

        from odl.util.graphics import show_discrete_function
        show_discrete_function(self, method=method, title=title,
                                **kwargs)


def uniform_discr(fspace, nsamples, exponent=2.0, interp='nearest',
                  impl='numpy', **kwargs):
    """Discretize an Lp function space by uniform sampling.

    Parameters
    ----------
    fspace : `FunctionSpace`
        Continuous function space. Its domain must be an
        `IntervalProd` instance.
    nsamples : `int` or `tuple` of `int`
        Number of samples per axis. For dimension >= 2, a tuple is
        required.
    exponent : positive `float`, optional
        The parameter :math:`p` in :math:`L^p`. If the exponent is not
        equal to the default 2.0, the space has no inner product.
    interp : `str`, optional
            Interpolation type to be used for discretization.

            'nearest' : use nearest-neighbor interpolation (default)

            'linear' : use linear interpolation (not implemented)
    impl : {'numpy', 'cuda'}
        Implementation of the data storage arrays
    kwargs : {'order', 'dtype', 'weighting'}
            'order' : {'C', 'F'}  (Default: 'C')
                Axis ordering in the data storage
            'dtype' : dtype
                Data type for the discretized space

                Default for 'numpy': 'float64' / 'complex128'
                Default for 'cuda': 'float32' / TODO
            'weighting' : {'simple', 'consistent'}
                Weighting of the discretized space functions.

                'simple': weight is a constant (cell volume)

                'consistent': weight is a matrix depending on the
                interpolation type

    Returns
    -------
    discr : `DiscreteLp`
        The uniformly discretized function space
    """
    if not isinstance(fspace, FunctionSpace):
        raise TypeError('space {!r} is not a `FunctionSpace` instance.'
                        ''.format(fspace))
    if not isinstance(fspace.domain, IntervalProd):
        raise TypeError('domain {!r} of the function space is not an '
                        '`IntervalProd` instance.'.format(fspace.domain))

    if impl == 'cuda' and not CUDA_AVAILABLE:
        raise ValueError('CUDA not available.')

    dtype = kwargs.pop('dtype', None)
    ds_type = dspace_type(fspace, impl, dtype)

    grid = uniform_sampling(fspace.domain, nsamples, as_midp=True)

    weighting = kwargs.pop('weighting', 'simple')
    weighting_ = weighting.lower()
    if weighting_ not in ('simple', 'consistent'):
        raise ValueError('weighting {!r} not understood.'.format(weighting))

    if weighting_ == 'simple':
        weight = np.prod(grid.stride)
    else:  # weighting_ == 'consistent'
        # TODO: implement
        raise NotImplemented

    if dtype is not None:
        dspace = ds_type(grid.ntotal, dtype=dtype, weight=weight,
                         exponent=exponent)
    else:
        dspace = ds_type(grid.ntotal, weight=weight, exponent=exponent)

    order = kwargs.pop('order', 'C')

    return DiscreteLp(fspace, grid, dspace, exponent=exponent, interp=interp,
                      order=order)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
