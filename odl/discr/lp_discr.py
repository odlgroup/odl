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

""":math:`L^p` type discretizations of function spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
from future.utils import raise_from
standard_library.install_aliases()
from builtins import super, str

# External
import numpy as np
from numbers import Integral

# ODL
from odl.discr.discretization import (
    Discretization, DiscretizationVector, dspace_type)
from odl.discr.discr_mappings import (
    PointCollocation, NearestInterpolation, LinearInterpolation)
from odl.discr.partition import RectPartition, uniform_partition_fromintv
from odl.set.sets import RealNumbers, ComplexNumbers
from odl.set.domain import IntervalProd
from odl.space.ntuples import Fn
from odl.space.cu_ntuples import CudaFn, CUDA_AVAILABLE
from odl.space.fspace import FunctionSpace
from odl.space.ntuples import Ntuples
from odl.util.numerics import apply_on_boundary
from odl.util.ufuncs import DiscreteLpUFuncs
from odl.util.utility import (
    is_real_dtype, is_complex_floating_dtype, dtype_repr, default_dtype,
    equiv_views)

__all__ = ('DiscreteLp', 'DiscreteLpVector',
           'uniform_discr_frompartition', 'uniform_discr_fromspace',
           'uniform_discr', 'sequence_space')

_SUPPORTED_INTERP = ('nearest', 'linear')


class DiscreteLp(Discretization):

    """Discretization of a Lebesgue :math:`L^p` space."""

    def __init__(self, fspace, partition, dspace, exponent=2.0,
                 interp='nearest', **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        fspace : `FunctionSpace`
            The continuous space to be discretized
        partition : `RectPartition`
            Partition of (a subset of) ``fspace.domain`` based on a
            `TensorGrid`
        dspace : `FnBase`
            Space of elements used for data storage. It must have the
            same `FnBase.field` as ``fspace``
        exponent : positive `float`, optional
            The parameter :math:`p` in :math:`L^p`. If the exponent is
            not equal to the default 2.0, the space has no inner
            product.
        interp : `str`, optional
            The interpolation type to be used for discretization.

            'nearest' : use nearest-neighbor interpolation (default)

            'linear' : use linear interpolation

        order : {'C', 'F'}, optional
            Ordering of the axes in the data storage. 'C' means the
            first axis varies slowest, the last axis fastest;
            vice versa for 'F'.
            Default: 'C'
        """
        if not isinstance(fspace, FunctionSpace):
            raise TypeError('{!r} is not a FunctionSpace instance.'
                            ''.format(fspace))
        if not isinstance(fspace.domain, IntervalProd):
            raise TypeError('Function space domain {!r} is not an '
                            'IntervalProd instance.'.format(fspace.domain))
        if not isinstance(partition, RectPartition):
            raise TypeError('Partition {!r} is not a RectPartition '
                            'instance.'.format(partition))
        if not fspace.domain.contains_set(partition.set):
            raise ValueError('Partition {} is not a subset of the function '
                             'domain {}'.format(partition, fspace.domain))

        self._interp = str(interp).lower()
        if self.interp not in _SUPPORTED_INTERP:
            raise ValueError("'{}' is not among the supported interpolation "
                             "types {}.".format(interp, _SUPPORTED_INTERP))

        order = str(kwargs.pop('order', 'C'))
        if str(order).upper() not in ('C', 'F'):
            raise ValueError('order {!r} not recognized.'.format(order))
        else:
            self._order = str(order).upper()

        self._partition = partition
        restriction = PointCollocation(fspace, self.partition, dspace,
                                       order=self.order)
        if self.interp == 'nearest':
            extension = NearestInterpolation(fspace, self.partition, dspace,
                                             order=self.order)
        elif self.interp == 'linear':
            extension = LinearInterpolation(fspace, self.partition, dspace,
                                            order=self.order)
        else:
            # Should not happen
            raise RuntimeError

        Discretization.__init__(self, fspace, dspace, restriction, extension)
        self._exponent = float(exponent)
        if (hasattr(self.dspace, 'exponent') and
                self.exponent != dspace.exponent):
            raise ValueError('exponent {} not equal to data space exponent '
                             '{}.'.format(self.exponent, dspace.exponent))

    @property
    def order(self):
        """Axis ordering for array flattening."""
        return self._order

    @property
    def partition(self):
        """The `RectPartition` of the domain."""
        return self._partition

    @property
    def grid(self):
        """Sampling grid of the discretization mappings."""
        return self.partition.grid

    @property
    def shape(self):
        """Shape of the underlying partition."""
        return self.partition.shape

    @property
    def ndim(self):
        """Number of dimensions."""
        return self.partition.ndim

    @property
    def size(self):
        """Total number of underlying partition cells."""
        return self.partition.size

    @property
    def cell_sides(self):
        """Side lengths of a cell in an underlying *uniform* partition."""
        return self.partition.cell_sides

    @property
    def cell_volume(self):
        """Cell volume of an underlying regular partition."""
        return self.partition.cell_volume

    @property
    def meshgrid(self):
        """All sampling points in the partition as a sparse meshgrid."""
        return self.partition.meshgrid

    def points(self):
        """All sampling points in the partition."""
        return self.partition.points()

    @property
    def exponent(self):
        """The exponent ``p`` in ``L^p``."""
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
        try:
            inp_elem = self.uspace.element(inp)
            return self.element_type(self, self.restriction(inp_elem))
        except TypeError:
            pass

        # Sequence-type input
        try:
            arr = np.asarray(inp, dtype=self.dtype, order=self.order)
            if arr.ndim > 1 and arr.shape != self.shape:
                arr = np.squeeze(arr)  # Squeeze could solve the problem
                if arr.shape != self.shape:
                    raise ValueError(
                        'input shape {} does not match grid shape {}.'
                        ''.format(arr.shape, self.shape))
            arr = arr.ravel(order=self.order)
            return self.element_type(self, self.dspace.element(arr))
        except TypeError as err:
            if str(err.args[0]).startswith('output contains points outside'):
                raise err
            else:
                raise_from(TypeError('unable to create an element of {} from '
                                     '{!r}.'.format(self, inp)), err)

    @property
    def interp(self):
        """Interpolation type of this discretization."""
        return self._interp

    # Overrides for space functions depending on partition
    #
    # The inherited methods by default use a weighting by a constant
    # (the grid cell size). In dimensions where the partitioned set contains
    # only a fraction of the outermost cells (e.g. if the outermost grid
    # points lie at the boundary), the corresponding contribuitons to
    # discretized integrals need to be scaled by that fraction.
    def _inner(self, x, y):
        """Return ``self.inner(x, y)``."""
        bdry_fracs = self.partition.boundary_cell_fractions
        if np.allclose(bdry_fracs, 1.0) or self.exponent == float('inf'):
            # no boundary weighting
            return super()._inner(x, y)
        else:
            # TODO: implement without copying x
            func_list = _scaling_func_list(bdry_fracs)

            x_arr = apply_on_boundary(x, func=func_list, only_once=False)
            return super()._inner(self.element(x_arr), y)

    def _norm(self, x):
        """Return ``self.norm(x)``."""
        bdry_fracs = self.partition.boundary_cell_fractions
        if np.allclose(bdry_fracs, 1.0) or self.exponent == float('inf'):
            # no boundary weighting
            return super()._norm(x)
        else:
            # TODO: implement without copying x
            func_list = _scaling_func_list(bdry_fracs, exponent=self.exponent)

            x_arr = apply_on_boundary(x, func=func_list, only_once=False)
            return super()._norm(self.element(x_arr))

    def _dist(self, x, y):
        """Return ``self.dist(x, y)``."""
        bdry_fracs = self.partition.boundary_cell_fractions
        if np.allclose(bdry_fracs, 1.0) or self.exponent == float('inf'):
            # no boundary weighting
            return super()._dist(x, y)
        else:
            # TODO: implement without copying x
            func_list = _scaling_func_list(bdry_fracs, exponent=self.exponent)

            arrs = [apply_on_boundary(vec, func=func_list, only_once=False)
                    for vec in (x, y)]

            return super()._dist(self.element(arrs[0]), self.element(arrs[1]))

    def __repr__(self):
        """Return ``repr(self).``"""
        # Check if the factory repr can be used
        if (uniform_partition_fromintv(
                self.uspace.domain, self.shape,
                nodes_on_bdry=False) == self.partition):
            use_uniform = True
            nodes_on_bdry = False
        elif (uniform_partition_fromintv(
                self.uspace.domain, self.shape,
                nodes_on_bdry=True) == self.partition):
            use_uniform = True
            nodes_on_bdry = True
        else:
            use_uniform = False

        if use_uniform:
            if isinstance(self.dspace, Fn):
                impl = 'numpy'
                default_dtype = np.float64
            elif isinstance(self.dspace, CudaFn):
                impl = 'cuda'
                default_dtype = np.float32
            else:  # This should never happen
                raise RuntimeError('unable to determine data space impl.')
            arg_fstr = '{}, {}, {}'
            if self.exponent != 2.0:
                arg_fstr += ', exponent={exponent}'
            if self.dtype != default_dtype:
                arg_fstr += ', dtype={dtype}'
            if self.interp != 'nearest':
                arg_fstr += ', interp={interp!r}'
            if impl != 'numpy':
                arg_fstr += ', impl={impl!r}'
            if self.order != 'C':
                arg_fstr += ', order={order!r}'
            if nodes_on_bdry:
                arg_fstr += ', nodes_on_bdry={nodes_on_bdry!r}'

            if self.ndim == 1:
                min_str = '{!r}'.format(self.uspace.domain.min()[0])
                max_str = '{!r}'.format(self.uspace.domain.max()[0])
                shape_str = '{!r}'.format(self.shape[0])
            else:
                min_str = '{!r}'.format(list(self.uspace.domain.min()))
                max_str = '{!r}'.format(list(self.uspace.domain.max()))
                shape_str = '{!r}'.format(list(self.shape))

            arg_str = arg_fstr.format(
                min_str, max_str, shape_str,
                exponent=self.exponent,
                dtype=dtype_repr(self.dtype),
                interp=self.interp,
                impl=impl,
                order=self.order,
                nodes_on_bdry=nodes_on_bdry)
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
                self.uspace, self.partition, self.dspace, interp=self.interp,
                ex=self.exponent, order=self.order)
            return '{}({})'.format(self.__class__.__name__, arg_str)

    def __str__(self):
        """Return ``str(self)``."""
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
            return super().asarray().reshape(self.shape,
                                             order=self.space.order)
        else:
            if out.shape not in (self.space.shape, (self.space.size,)):
                raise ValueError('output array has shape {}, expected '
                                 '{} or ({},).'
                                 ''.format(out.shape, self.space.shape,
                                           self.space.size))
            out_r = out.reshape(self.space.shape,
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
                                 ''.format(self.space.order, out_order))

            super().asarray(out=out.ravel(order=self.space.order))
            return out

    @property
    def ndim(self):
        """Number of dimensions."""
        return self.space.ndim

    @property
    def shape(self):
        """Multi-dimensional shape of this discrete function."""
        # override shape
        return self.space.shape

    @property
    def cell_sides(self):
        """Side lengths of a cell in an underlying *uniform* partition."""
        return self.space.cell_sides

    @property
    def cell_volume(self):
        """Cell volume of an underlying regular grid."""
        return self.space.cell_volume

    @property
    def order(self):
        """Axis ordering for array flattening."""
        return self.space.order

    def __setitem__(self, indices, values):
        """Set values of this vector.

        Parameters
        ----------
        indices : `int` or `slice`
            The position(s) that should be set
        values : scalar, `array-like` or `NtuplesVector`
            The value(s) that are to be assigned.
            If ``indices`` is an `int`, ``values`` must be a single
            value.
            If ``indices`` is a `slice`, ``values`` must be
            broadcastable to the size of the slice (same size,
            shape ``(1,)`` or single value).
            For ``indices=slice(None)``, i.e. in the call
            ``vec[:] = values``, a multi-dimensional array of correct
            shape is allowed as ``values``.
        """
        if values in self.space:
            # For RawDiscretizationVector of the same type, use ntuple directly
            return self.ntuple.__setitem__(indices, values.ntuple)
        elif (isinstance(self.space.dspace, Ntuples) and
              equiv_views(self.asarray(), values) and
              indices in (slice(None), Ellipsis)):
            # Optimization for self[:] = self.asarray() if the implementation
            # is Numpy. Otherwise, asarray() would create a copy.
            return self
        else:
            # Other sequence types are piped through a Numpy array
            if indices == slice(None):
                values = np.atleast_1d(values)
                if (values.ndim > 1 and
                        values.shape != self.space.shape):
                    raise ValueError('shape {} of value array {} not equal'
                                     ' to sampling grid shape {}.'
                                     ''.format(values.shape, values,
                                               self.space.shape))
                values = values.ravel(order=self.space.order)

            return super().__setitem__(indices, values)

    @property
    def ufunc(self):
        """`DiscreteLpUFuncs`, access to numpy style ufuncs.

        Examples
        --------
        >>> X = uniform_discr(0, 1, 2)
        >>> x = X.element([1, -2])
        >>> x.ufunc.absolute()
        uniform_discr(0.0, 1.0, 2).element([1.0, 2.0])

        These functions can also be used with broadcasting

        >>> x.ufunc.add(3)
        uniform_discr(0.0, 1.0, 2).element([4.0, 1.0])

        and non-space elements

        >>> x.ufunc.subtract([3, 3])
        uniform_discr(0.0, 1.0, 2).element([-2.0, -5.0])

        There is also support for various reductions (sum, prod, min, max)

        >>> x.ufunc.sum()
        -1.0

        Also supports out parameter

        >>> y = X.element([3, 4])
        >>> out = X.element()
        >>> result = x.ufunc.add(y, out=out)
        >>> result
        uniform_discr(0.0, 1.0, 2).element([4.0, 2.0])
        >>> result is out
        True

        Notes
        -----
        These are optimized to use the underlying ntuple space and incur no
        overhead unless these do.
        """
        return DiscreteLpUFuncs(self)

    def show(self, method='', title=None, indices=None, show=False, fig=None,
             **kwargs):
        """Display the function graphically.

        Parameters
        ----------
        method : `str`, optional
            1d methods:

            'plot' : graph plot

            'scatter' : scattered 2d points
            (2nd axis <-> value)

            2d methods:

            'imshow' : image plot with coloring according to value,
            including a colorbar.

            'scatter' : cloud of scattered 3d points
            (3rd axis <-> value)

        indices : index expression, optional
            Display a slice of the array instead of the full array. The
            index expression is most easily created with the `numpy.s_`
            constructor, i.e. supply ``np.s_[:, 1, :]`` to display the
            first slice along the second axis.
            For data with 3 or more dimensions, the 2d slice in the first
            two axes at the "middle" along the remaining axes is shown
            (semantically ``[:, :, shape[2:] // 2]``).

        title : `str`, optional
            Set the title of the figure

        show : `bool`, optional
            If the plot should be showed now or deferred until later.

        fig : `matplotlib.figure.Figure`
            The figure to show in. Expected to be of same "style", as
            the figure given by this function. The most common use case
            is that ``fig`` is the return value from an earlier call to
            this function.

        kwargs : {'figsize', 'saveto', 'clim', ...}
            Extra keyword arguments passed on to display method
            See the Matplotlib functions for documentation of extra
            options.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The resulting figure. It is also shown to the user.

        See Also
        --------
        odl.util.graphics.show_discrete_data : Underlying implementation
        """

        from odl.util.graphics import show_discrete_data

        # Default to showing x-y slice "in the middle"
        if indices is None and self.ndim >= 3:
            indices = [np.s_[:]] * 2
            indices += [n // 2 for n in self.space.shape[2:]]

        if isinstance(indices, (Integral, slice)):
            indices = [indices]
        elif indices is None or indices == Ellipsis:
            indices = [np.s_[:]] * self.ndim
        else:
            indices = list(indices)

        if Ellipsis in indices:
            # Replace Ellipsis with the correct number of [:] expressions
            pos = indices.index(Ellipsis)
            indices = (indices[:pos] +
                       [np.s_[:]] * (self.ndim - len(indices) + 1) +
                       indices[pos + 1:])

        if len(indices) < self.ndim:
            raise ValueError('too few axes ({} < {}).'.format(len(indices),
                                                              self.ndim))
        if len(indices) > self.ndim:
            raise ValueError('too many axes ({} > {}).'.format(len(indices),
                                                               self.ndim))

        if self.ndim <= 3:
            axis_labels = ['x', 'y', 'z']
        else:
            axis_labels = ['x{}'.format(axis) for axis in range(self.ndim)]
        squeezed_axes = [axis for axis in range(self.ndim)
                         if not isinstance(indices[axis], Integral)]
        axis_labels = [axis_labels[axis] for axis in squeezed_axes]

        # Squeeze grid and values according to the index expression
        grid = self.space.grid[indices].squeeze()
        values = self.asarray()[indices].squeeze()

        return show_discrete_data(values, grid, method=method, title=title,
                                  show=show, fig=fig, axis_labels=axis_labels,
                                  **kwargs)


def uniform_discr_frompartition(partition, exponent=2.0, interp='nearest',
                                impl='numpy', **kwargs):
    """Discretize an Lp function space given a uniform partition.

    Parameters
    ----------
    partition : `RectPartition`
        Regular (uniform) partition to be used for discretization
    exponent : positive `float`, optional
        The parameter ``p`` in ``L^p``. If the exponent is not
        equal to the default 2.0, the space has no inner product.
    interp : `str`, optional
        Interpolation type to be used for discretization.

            'nearest' : use nearest-neighbor interpolation (default)

            'linear' : use linear interpolation (not implemented)

    impl : {'numpy', 'cuda'}
        Implementation of the data storage arrays

    Other Parameters
    ----------------
    order : {'C', 'F'}
        Axis ordering in the data storage. Default: 'C'
    dtype : dtype
        Data type for the discretized space

            Default for 'numpy': 'float64' / 'complex128'

            Default for 'cuda': 'float32' / (not implemented)

    weighting : {'const', 'none'}
        Weighting of the discretized space functions.

            'const' : weight is a constant, the cell volume (default)

            'none' : no weighting

    Returns
    -------
    discr : `DiscreteLp`
        The uniformly discretized function space

    Examples
    --------
    >>> from odl import uniform_partition
    >>> part = uniform_partition(0, 1, 10)
    >>> uniform_discr_frompartition(part)
    uniform_discr(0.0, 1.0, 10)

    See also
    --------
    uniform_discr : implicit uniform Lp discretization
    uniform_discr_fromspace : uniform Lp discretization from an existing
        function space
    uniform_partition : partition of the function domain
    """
    if not isinstance(partition, RectPartition):
        raise TypeError('partition {!r} is not a RectPartition instance.'
                        ''.format(partition))
    if not partition.is_regular:
        raise ValueError('partition is not regular.')

    impl_ = str(impl).lower()
    if impl_ == 'numpy':
        dtype = np.dtype(kwargs.pop('dtype', 'float64'))
    elif impl_ == 'cuda':
        if not CUDA_AVAILABLE:
            raise ValueError('CUDA not available.')
        dtype = np.dtype(kwargs.pop('dtype', 'float32'))

    if is_real_dtype(dtype):
        field = RealNumbers()
    elif is_complex_floating_dtype(dtype):
        field = ComplexNumbers()
    else:
        raise ValueError('cannot use non-scalar data type {}.'.format(dtype))

    fspace = FunctionSpace(partition.set, field=field)
    ds_type = dspace_type(fspace, impl, dtype)

    order = kwargs.pop('order', 'C')

    weighting = kwargs.pop('weighting', 'const')
    weighting_ = str(weighting).lower()
    if weighting_ == 'none' or float(exponent) == float('inf'):
        weight = None
    elif weighting_ == 'const':
        weight = partition.cell_volume
    else:
        raise ValueError("weighting '{}' not understood.".format(weighting))

    if dtype is not None:
        dspace = ds_type(partition.size, dtype=dtype, weight=weight,
                         exponent=exponent)
    else:
        dspace = ds_type(partition.size, weight=weight, exponent=exponent)

    return DiscreteLp(fspace, partition, dspace, exponent, interp, order=order)


def uniform_discr_fromspace(fspace, nsamples, exponent=2.0, interp='nearest',
                            impl='numpy', **kwargs):
    """Discretize an Lp function space by uniform partition.

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

    nodes_on_bdry : `bool` or boolean `array-like`
        If `True`, place the outermost grid points at the boundary. For
        `False`, they are shifted by half a cell size to the 'inner'.
        If an array-like is given, it must have shape ``(ndim, 2)``,
        where ``ndim`` is the number of dimensions. It defines per axis
        whether the leftmost (first column) and rightmost (second column)
        nodes node lie on the boundary.
        Default: `False`
    order : {'C', 'F'}
        Axis ordering in the data storage. Default: 'C'
    dtype : dtype
        Data type for the discretized space

            Default for 'numpy': 'float64' / 'complex128'

            Default for 'cuda': 'float32' / (not implemented)

    Returns
    -------
    discr : `DiscreteLp`
        The uniformly discretized function space

    Examples
    --------
    >>> from odl import Interval, FunctionSpace
    >>> intv = Interval(0, 1)
    >>> space = FunctionSpace(intv)
    >>> uniform_discr_fromspace(space, 10)
    uniform_discr(0.0, 1.0, 10)

    See also
    --------
    uniform_discr : implicit uniform Lp discretization
    uniform_discr_frompartition : uniform Lp discretization using a given
        uniform partition of a function domain
    odl.discr.partition.uniform_partition :
        partition of the function domain
    """
    if not isinstance(fspace, FunctionSpace):
        raise TypeError('space {!r} is not a `FunctionSpace` instance.'
                        ''.format(fspace))
    if not isinstance(fspace.domain, IntervalProd):
        raise TypeError('domain {!r} of the function space is not an '
                        '`IntervalProd` instance.'.format(fspace.domain))

    field = fspace.field
    dtype = kwargs.pop('dtype', None)
    if dtype is None:
        dtype = default_dtype(str(impl).lower(), field)
    else:
        dtype = np.dtype(dtype)

    if field == RealNumbers() and not is_real_dtype(dtype):
        raise ValueError('cannot discretize real space {} with '
                         'non-real data type {}.'
                         ''.format(fspace, dtype))
    elif field == ComplexNumbers() and not is_complex_floating_dtype(dtype):
        raise ValueError('cannot discretize complex space {} with '
                         'non-complex-floating data type {}.'
                         ''.format(fspace, dtype))

    nodes_on_bdry = kwargs.pop('nodes_on_bdry', False)
    partition = uniform_partition_fromintv(fspace.domain, nsamples,
                                           nodes_on_bdry)

    return uniform_discr_frompartition(partition, exponent, interp, impl,
                                       dtype=dtype, **kwargs)


def uniform_discr(min_corner, max_corner, nsamples,
                  exponent=2.0, interp='nearest', impl='numpy', **kwargs):
    """Discretize an Lp function space by uniform sampling.

    Parameters
    ----------
    min_corner : `float` or `tuple` of `float`
        Minimum corner of the result.
    nsamples : `float` or `tuple` of `float`
        Minimum corner of the result.
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
    nodes_on_bdry : `bool` or `sequence`, optional
        If a sequence is provided, it determines per axis whether to
        place the last grid point on the boundary (True) or shift it
        by half a cell size into the interior (False). In each axis,
        an entry may consist in a single `bool` or a 2-tuple of
        `bool`. In the latter case, the first tuple entry decides for
        the left, the second for the right boundary. The length of the
        sequence must be ``array.ndim``.

        A single boolean is interpreted as a global choice for all
        boundaries.
        Default: `False`

    dtype : dtype
        Data type for the discretized space

            Default for 'numpy': 'float64'

            Default for 'cuda': 'float32'

    order : {'C', 'F'}, optional
        Ordering of the axes in the data storage. 'C' means the
        first axis varies slowest, the last axis fastest;
        vice versa for 'F'.
        Default: 'C'
    weighting : {'const', 'none'}
        Weighting of the discretized space functions.

            'simple': weight is a constant (cell volume)

            'consistent': weight is a matrix depending on the
            interpolation type

    Returns
    -------
    discr : `DiscreteLp`
        The uniformly discretized function space

    Examples
    --------
    Create real space:

    >>> uniform_discr([0, 0], [1, 1], [10, 10])
    uniform_discr([0.0, 0.0], [1.0, 1.0], [10, 10])

    Can create complex space by giving a dtype

    >>> uniform_discr([0, 0], [1, 1], [10, 10], dtype='complex')
    uniform_discr([0.0, 0.0], [1.0, 1.0], [10, 10], dtype='complex')

    See also
    --------
    uniform_discr_frompartition : uniform Lp discretization using a given
        uniform partition of a function domain
    uniform_discr_fromspace : uniform discretization from an existing
        function space
    """
    # Select field by dtype
    dtype = kwargs.get('dtype', None)
    if dtype is None or is_real_dtype(dtype):
        field = RealNumbers()
    else:
        field = ComplexNumbers()

    fspace = FunctionSpace(IntervalProd(min_corner, max_corner), field)

    return uniform_discr_fromspace(fspace, nsamples, exponent, interp, impl,
                                   **kwargs)


def sequence_space(shape, exponent=2.0, impl='numpy', **kwargs):
    """Return an object mimicing the sequence space ``l^p(R^d)``.

    The returned object is a `DiscreteLp` without restriction and
    extension operators. It uses a grid with stride 1 and no
    weighting.

    Parameters
    ----------
    shape : `sequence` of `int`
        Multi-dimensional size of the elements in this space
    exponent : positive `float`, optional
        The parameter ``p`` in ```L^p``. If the exponent is
        not equal to the default 2.0, the space has no inner
        product.
    impl : {'numpy', 'cuda'}

    Other Parameters
    ----------------
    dtype : dtype
        Data type for the discretized space

            Default for 'numpy': 'float64'

            Default for 'cuda': 'float32'

    order : {'C', 'F'}, optional
        Ordering of the axes in the data storage. 'C' means the
        first axis varies slowest, the last axis fastest;
        vice versa for 'F'.
        Default: 'C'

    Returns
    -------
    seqspc : `DiscreteLp`
        The sequence-space-like discrete Lp

    Examples
    --------
    >>> seq_spc = sequence_space((3, 3))
    >>> seq_spc.one().norm() == 3.0
    True
    >>> seq_spc = sequence_space((3, 3), exponent=1)
    >>> seq_spc.one().norm() == 9.0
    True
    """
    kwargs.pop('weighting', None)
    kwargs.pop('nodes_on_bdry', None)
    shape = np.atleast_1d(shape)
    return uniform_discr([0] * len(shape), shape, shape, impl=impl,
                         exponent=exponent, nodes_on_bdry=False,
                         weighting='none', **kwargs)


def _scaling_func_list(bdry_fracs, exponent=1.0):
    """Return a list of lists of scaling functions for the boundary."""
    def scaling(factor):
        def scaling_func(x):
            return x * factor
        return scaling_func

    func_list = []
    for frac_l, frac_r in bdry_fracs:
        func_list_entry = []
        if np.isclose(frac_l, 1.0):
            func_list_entry.append(None)
        else:
            func_list_entry.append(scaling(frac_l ** (1 / exponent)))

        if np.isclose(frac_r, 1.0):
            func_list_entry.append(None)
        else:
            func_list_entry.append(scaling(frac_r ** (1 / exponent)))

        func_list.append(func_list_entry)
    return func_list


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
