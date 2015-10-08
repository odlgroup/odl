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

"""Discretizations of L2 spaces."""

# pylint: disable=abstract-method

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import super, str

# External
import numpy as np

# ODL
from odl.discr.discretization import Discretization, dspace_type
from odl.discr.discr_mappings import GridCollocation, NearestInterpolation
from odl.discr.grid import uniform_sampling, RegularGrid
from odl.set.domain import IntervalProd
from odl.space.ntuples import ConstWeightedInnerProduct, Fn
from odl.space.default import L2
from odl.util.utility import is_complex_dtype
from odl.space import CUDA_AVAILABLE
if CUDA_AVAILABLE:
    from odl.space.cu_ntuples import CudaConstWeightedInnerProduct, CudaFn
else:
    CudaConstWeightedInnerProduct = None
    CudaFn = type(None)

__all__ = ('DiscreteL2', 'l2_uniform_discretization')

_SUPPORTED_INTERP = ('nearest',)


class DiscreteL2(Discretization):

    """Discretization of an :math:`L^2` space."""

    def __init__(self, l2space, grid, dspace, interp='nearest', **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        l2space : `L2`
            The continuous space to be discretized
        dspace : `FnBase`, same `field` as `l2space`
            The space of elements used for data storage
        grid : `TensorGrid`
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
            raise TypeError('L2 space domain {} is not an `IntervalProd` '
                            'instance.'.format(l2space.domain))

        interp = str(interp).lower()
        if interp not in _SUPPORTED_INTERP:
            raise TypeError('{} is not among the supported interpolation'
                            'types {}.'.format(interp, _SUPPORTED_INTERP))

        self._order = str(kwargs.pop('order', 'C')).upper()
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
        element : `DiscreteL2.Vector`
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
            if arr.ndim > 1 and arr.shape != self.grid.shape:
                raise ValueError('input shape {} does not match grid shape {}'
                                 ''.format(arr.shape, self.grid.shape))
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
        if (uniform_sampling(self.uspace.domain, self.grid.shape,
                             as_midp=True) == self.grid):
            if isinstance(self.dspace, Fn):
                impl = 'numpy'
            elif isinstance(self.dspace, CudaFn):
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

        """Representation of a `DiscreteL2` element."""

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

                super().asarray(out=out.ravel(order=self.space.order))
                return out

        @property
        def ndim(self):
            """Number of dimensions, always 1."""
            return self.space.grid.ndim

        @property
        def shape(self):
            # override shape
            return self.space.grid.shape

        def __setitem__(self, indices, values):
            """Set values of this vector.

            Parameters
            ----------
            indices : int or slice
                The position(s) that should be set
            values : {scalar, array-like, `Ntuples.Vector`}
                The value(s) that are to be assigned.

                If `indices` is an `int`, `value` must be single value.

                If `indices` is a `slice`, `value` must be
                broadcastable to the size of the slice (same size,
                shape (1,) or single value).
                For `indices=slice(None, None, None)`, i.e. in the call
                `vec[:] = values`, a multi-dimensional array of correct
                shape is allowed as `values`.
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

        def show(self, method='', figsize=None, saveto='', **kwargs):
            """Create a plot of the function in 1d or 2d.

            Parameters
            ----------
            method : string, optional
                1d methods:

                'plot' : graph plot

                2d methods:

                'imshow' : image plot with coloring according to value,
                including a colorbar.

                'scatter' : cloud of scattered 3d points
                (3rd axis <-> value)

                'wireframe', 'plot_wireframe' : surface plot

            kwargs : extra keyword arguments passed on to display method
                See the Matplotlib functions for documentation of extra
                options.

            See Also
            --------
            matplotlib.pyplot.plot : Show graph plot

            matplotlib.pyplot.imshow : Show data as image

            matplotlib.pyplot.scatter : Show scattered 3d points
            """
            from matplotlib import pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            args_re = []
            args_im = []
            dsp_kwargs = {}
            sub_kwargs = {}
            arrange_subplots = (121, 122)  # horzontal arrangement

            values = self.asarray()

            if self.ndim == 1:  # TODO: maybe a plotter class would be better
                if not method:
                    method = 'plot'

                if method == 'plot':
                    args_re += [self.space.grid.coord_vecs[0], values.real]
                    args_im += [self.space.grid.coord_vecs[0], values.imag]
                else:
                    raise ValueError('display method {!r} not supported.'
                                     ''.format(method))

            elif self.ndim == 2:
                if not method:
                    method = 'imshow'

                if method == 'imshow':
                    from matplotlib.cm import gray
                    args_re = [values.real.T]
                    args_im = [values.imag.T]
                    extent = [self.space.grid.min()[0],
                              self.space.grid.max()[0],
                              self.space.grid.min()[1],
                              self.space.grid.max()[1]]

                    # TODO: Make interpolation smart
                    dsp_kwargs.update({'interpolation': 'none', 'cmap': gray,
                                       'extent': extent,
                                       'aspect': 'auto'})
                elif method == 'scatter':
                    pts = self.space.grid.points()
                    args_re = [pts[:, 0], pts[:, 1], values.ravel().real]
                    args_re = [pts[:, 0], pts[:, 1], values.ravel().imag]
                    sub_kwargs.update({'projection': '3d'})
                elif method in ('wireframe', 'plot_wireframe'):
                    method = 'plot_wireframe'
                    xm, ym = self.space.grid.meshgrid()
                    args_re = [xm, ym, values.real.T]
                    args_im = [xm, ym, values.imag.T]
                    sub_kwargs.update({'projection': '3d'})
                else:
                    raise ValueError('display method {!r} not supported.'
                                     ''.format(method))

            else:
                raise NotImplemented('no method for {}d display implemented.'
                                     ''.format(self.space.ndim))

            # Additional keyword args are passed on to the display method
            dsp_kwargs.update(**kwargs)

            fig = plt.figure(figsize=figsize)
            if is_complex_dtype(self.space.dspace.dtype):
                sub_re = plt.subplot(arrange_subplots[0], **sub_kwargs)
                sub_re.set_title('Real part')
                sub_re.set_xlabel('x')
                sub_re.set_ylabel('y')
                display_re = getattr(sub_re, method)
                csub_re = display_re(*args_re, **dsp_kwargs)

                if method == 'imshow':
                    minval_re = np.min(values.real)
                    maxval_re = np.max(values.real)
                    ticks_re = [minval_re, (maxval_re + minval_re) / 2.,
                                maxval_re]
                    plt.colorbar(csub_re, orientation='horizontal',
                                 ticks=ticks_re, format='%.4g')

                sub_im = plt.subplot(arrange_subplots[1], **sub_kwargs)
                sub_im.set_title('Imaginary part')
                sub_im.set_xlabel('x')
                sub_im.set_ylabel('y')
                display_im = getattr(sub_im, method)
                csub_im = display_im(*args_im, **dsp_kwargs)

                if method == 'imshow':
                    minval_im = np.min(values.imag)
                    maxval_im = np.max(values.imag)
                    ticks_im = [minval_im, (maxval_im + minval_im) / 2.,
                                maxval_im]
                    plt.colorbar(csub_im, orientation='horizontal',
                                 ticks=ticks_im, format='%.4g')

            else:
                sub = plt.subplot(111, **sub_kwargs)
                sub.set_xlabel('x')
                sub.set_ylabel('y')
                try:
                    # For 3d plots
                    sub.set_zlabel('z')
                except AttributeError:
                    pass
                display = getattr(sub, method)
                csub = display(*args_re, **dsp_kwargs)

                if method == 'imshow':
                    minval = np.min(values)
                    maxval = np.max(values)
                    ticks = [minval, (maxval + minval) / 2., maxval]
                    plt.colorbar(csub, ticks=ticks, format='%.4g')

            plt.show()
            if saveto:
                fig.savefig(saveto)


def l2_uniform_discretization(l2space, nsamples, interp='nearest',
                              impl='numpy', **kwargs):
    """Discretize an L2 space by uniform sampling.

    Parameters
    ----------
    l2space : `L2`
        Continuous :math:`L^2` type space. Its domain must be an
        `IntervalProd` instance.
    nsamples : int or tuple of int
        Number of samples per axis. For dimension >= 2, a tuple is
        required.
    interp : string, optional
            Interpolation type to be used for discretization.

            'nearest' : use nearest-neighbor interpolation (default)

            'linear' : use linear interpolation (not implemented)
    impl : {'numpy', 'cuda'}
        Implementation of the data storage arrays
    kwargs : {'order', 'dtype', 'weighting'}
            'order' : {'C', 'F'}  (Default: 'C')
                Axis ordering in the data storage
            'dtype' : type
                Data type for the discretized space

                Default for 'numpy': 'float64' / 'complex128'
                Default for 'cuda': 'float32' / TODO
            'weighting' : {'simple', 'consistent'}
                Weighting of the discretized inner product.

                'simple': weight is a constant (cell volume)

                'consistent': weight is a matrix depending on the
                interpolation type

    Returns
    -------
    l2discr : `DiscreteL2`
        The uniformly discretized L2 space
    """
    if not isinstance(l2space, L2):
        raise TypeError('space {!r} is not an L2 instance.'.format(l2space))

    if not isinstance(l2space.domain, IntervalProd):
        raise TypeError('domain {!r} of the L2 space is not an `IntervalProd` '
                        'instance.'.format(l2space.domain))

    if impl == 'cuda' and not CUDA_AVAILABLE:
        raise ValueError('CUDA not available.')

    ds_type = dspace_type(l2space, impl)
    dtype = kwargs.pop('dtype', None)

    grid = uniform_sampling(l2space.domain, nsamples, as_midp=True)

    weighting = kwargs.pop('weighting', 'simple')
    if weighting not in ('simple', 'consistent'):
        raise ValueError('weighting {!r} not understood.'.format(weighting))

    if weighting == 'simple':
        weighting_const = np.prod(grid.stride)
        if impl == 'numpy':
            inner = ConstWeightedInnerProduct(weighting_const)
        else:
            inner = CudaConstWeightedInnerProduct(weighting_const)
    else:  # weighting == 'consistent'
        # TODO: implement
        raise NotImplementedError

    if dtype is not None:
        # FIXME: CUDA spaces do not yet support custom inner product
        if impl == 'cuda':  # ignore inner until fix
            dspace = ds_type(grid.ntotal, dtype=dtype)
        else:
            dspace = ds_type(grid.ntotal, dtype=dtype, inner=inner)
    else:
        # FIXME: CUDA spaces do not yet support custom inner product
        if impl == 'cuda':  # ignore inner until fix
            dspace = ds_type(grid.ntotal)
        else:
            dspace = ds_type(grid.ntotal, inner=inner)

    order = kwargs.pop('order', 'C')

    return DiscreteL2(l2space, grid, dspace, interp=interp, order=order)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
