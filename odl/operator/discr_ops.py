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

"""Operators defined on `DiscreteLp`."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

# ODL imports
from odl.operator.operator import Operator
from odl.set.pspace import ProductSpace
from odl.discr.lp_discr import DiscreteLp
from odl.space.ntuples import Rn
from odl.space.cu_ntuples import CudaRn


__all__ = ('DiscretePartDeriv', 'DiscreteGradient', 'DiscreteDivergence')


def finite_diff(f, out=None, axis=0, dx=1.0, edge_order=None,
                zero_padding=False):
    """Calculate the partial derivative of `f` along a given `axis` using
    discrete differences.

    The partial derivative is computed using second-order accurate central
    differences in the interior and either first- or second-order accurate
    one-sides (forward or backward) differences at the boundaries.

    Assuming (implicit) zero padding central differences are used on the
    interior and on endpoints. Otherwise one-sided differences are used. If
    `zero_padding` is False first-order accuracy can be triggered on
    endpoints with parameter `edge_order`.

    The returned array has the same shape as the input array `f`.

    Parameters
    ----------
    f : array-like
         An N-dimensional array
    out : array-like
         An N-dimensional array
    axis : `int`, optional
        The axis along which the partial derivative is evaluated. Default: 0
    dx : `float`, optional
        Scalars specifying the sample distances in each dimension `axis`.
        Default distance: 1.0
    edge_order : {1, 2}, optional
        First-order accurate differences (1) can be used at the boundaries
        if no zero padding is used. Default edge order: 2
    zero_padding : `bool`, optional
        Implicit zero padding. Assumes values outside the domain of `f` to be
        zero. Default: False

    Returns
    -------
    out : `numpy.ndarray`
        Returns an N-dimensional array of the same shape as `f`

    Examples
    --------
    >>> f = np.arange(10, dtype=float)
    >>> f
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
    >>> finite_diff(f)
    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
    >>> finite_diff(f, axis=0, dx=1.0, edge_order=2, zero_padding=False)
    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
    >>> finite_diff(f, dx=0.5)
    array([ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.])
    >>> finite_diff(f, zero_padding=True)
    array([ 0.5,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. , -4. ])
    >>> finite_diff(f, zero_padding=False, edge_order=1)
    array([ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.])
    >>> out = finite_diff(f)
    >>> out is finite_diff(f)
    False
    >>> out is finite_diff(f, out)
    True
    """
    # TODO: implement forward/backward differences
    # TODO: implement other boundary conditions

    if zero_padding is True and edge_order == 1:
        raise ValueError("zero padding uses second-order accurate "
                         "differences at boundaries. First-order accurate "
                         "edges can only be used without zero padding.")
    f_data = np.asarray(f)
    ndim = f_data.ndim

    if not 0 <= axis < ndim:
        raise IndexError("axis parameter ({0}) exceeds the number of "
                         "dimensions ({1}).".format(axis, ndim))

    if f_data.shape[axis] < 2:
        raise ValueError("shape of array too small to calculate a numerical "
                         "gradient, at least two elements are required.")
    if out is None:
        out = np.empty_like(f_data)
    else:
        if not out.shape == f.shape:
            raise TypeError("shape of `out` array ({0}) does not match the "
                            "shape of input array `f` ({1}).".format(out.shape,
                                                                    f.shape))

    if edge_order not in {1, 2, None}:
        raise ValueError("edge order ({0}) not valid".format(edge_order))

    if edge_order is None:
        edge_order = 2

    if dx == 0:
        raise ValueError("step length is zero.")
    else:
        dx = float(dx)

    # create slice objects --- initially all are [:, :, ..., :]
    # current slice
    slice_out = [slice(None)] * ndim
    # slices used to calculate finite differences
    slice_node1 = [slice(None)] * ndim
    slice_node2 = [slice(None)] * ndim
    slice_node3 = [slice(None)] * ndim

    # Numerical differentiation: 2nd order interior
    slice_out[axis] = slice(1, -1)
    slice_node1[axis] = slice(2, None)
    slice_node2[axis] = slice(None, -2)
    # 1D equivalent: out[1:-1] = (f[2:] - f[:-2])/2.0
    np.subtract(f_data[slice_node1], f_data[slice_node2], out[slice_out])
    out[slice_out] /= 2.0

    # central differences
    if zero_padding:
        # Assume zeros for indices outside the domain of `f`

        slice_out[axis] = 0
        slice_node1[axis] = 1
        # 1D equivalent: out[0] = (f[1] - 0)/2.0
        out[slice_out] = f_data[slice_node1] / 2.0

        slice_out[axis] = -1
        slice_node2[axis] = -2
        # 1D equivalent: out[-1] = (0 - f[-2])/2.0
        out[slice_out] = - f_data[slice_node2] / 2.0

    # one-sided differences
    else:

        # Numerical differentiation: 1st order edges
        if f_data.shape[axis] == 2 or edge_order == 1:

            slice_out[axis] = 0
            slice_node1[axis] = 1
            slice_node2[axis] = 0
            # 1D equivalent: out[0] = (f[1] - f[0])
            np.subtract(f_data[slice_node1], f_data[slice_node2],
                        out[slice_out])

            slice_out[axis] = -1
            slice_node1[axis] = -1
            slice_node2[axis] = -2
            # 1D equivalent: out[-1] = (f[-1] - f[-2])
            np.subtract(f_data[slice_node1], f_data[slice_node2],
                        out[slice_out])

        # Numerical differentiation: 2nd order edges
        else:

            slice_out[axis] = 0
            slice_node1[axis] = 0
            slice_node2[axis] = 1
            slice_node3[axis] = 2
            # 1D equivalent: out[0] = -(3*f[0] - 4*f[1] + f[2]) / 2.0
            out[slice_out] = -(3.0 * f_data[slice_node1] - 4.0 * f_data[
                slice_node2] + f_data[slice_node3]) / 2.0

            slice_out[axis] = -1
            slice_node1[axis] = -1
            slice_node2[axis] = -2
            slice_node3[axis] = -3
            # 1D equivalent: out[-1] = (3*f[-1] - 4*f[-2] + f[-3]) / 2.0
            out[slice_out] = (3.0 * f_data[slice_node1] - 4.0 * f_data[
                slice_node2] + f_data[slice_node3]) / 2.0

    # divide by step size
    out /= dx

    return out


class DiscretePartDeriv(Operator):
    """Calculate the discrete partial derivative along a given axis.

        Calls helper function `finite_diff` to calculate finite difference.
        Preserves the shape of the underlying grid.
    """
    # TODO: implement adjoint

    def __init__(self, space, axis=0, dx=1.0, edge_order=None,
                 zero_padding=False):
        """Initialize an operator instance.

        Parameters
        ----------
        space : `DiscreteLp`
            The space of elements which the operator is acting on
        axis : `int`, optional
            The axis along which the partial derivative is evaluated.
            Default: 0
        dx : `float`, optional
            Scalars specifying the sampling distances in dimension `axis`.
            Default distance: 1.0
        edge_order : {1, 2}, optional
            First-order accurate differences can be used at the boundaries
            if no zero padding is used. Default edge order: 2
        zero_padding : `bool`, optional
            Implicit zero padding. Assumes values outside the domain of `f`
            to be zero. Default: False
        """

        if not isinstance(space, DiscreteLp):
            raise TypeError('space {!r} is not a `DiscreteLp` '
                            'instance.'.format(space))

        super().__init__(domain=space, range=space, linear=True)
        self.axis = axis
        self.dx = dx
        self.edge_order = edge_order
        self.zero_padding = zero_padding

    def _apply(self, x, out):
        """Apply gradient operator to ``x`` and store result in ``out``.

        Parameters
        ----------
        x : ``domain`` element
            Input vector to which the operator is applied to
        out : ``range`` element
            Output vector to which the result is written

        Examples
        --------
        >>> from odl import uniform_discr, FunctionSpace, Rectangle
        >>> data = np.arange(5, dtype=float) * np.arange(1, 3).reshape(2, 1)
        >>> space = FunctionSpace(Rectangle([0, 0], [2, 1]))
        >>> disc = uniform_discr(space, data.shape)
        >>> par_div = DiscretePartDeriv(disc)
        >>> f = par_div.domain.element(data)
        >>> par_div_f = par_div(f)
        >>> print(par_div_f)
        [[0.0, 1.0, 2.0, 3.0, 4.0],
         [0.0, 1.0, 2.0, 3.0, 4.0]]
        """

        if isinstance(self.domain.dspace, Rn):
            finite_diff(x.asarray(), out.asarray(), axis=self.axis,
                        dx=self.dx, edge_order=self.edge_order,
                        zero_padding=self.zero_padding)
        elif isinstance(self.domain.dspace, CudaRn):
            pass

    @property
    def adjoint(self):
        """Return the adjoint."""
        raise NotImplementedError('adjoint not implemented')


class DiscreteGradient(Operator):
    """Gradient operator applicable to `DiscreteLp` spaces with number of
    dimensions ``n``.

    Calls helper function `finite_diff` to calculate each component of the
    resulting product space vector. For the adjoint of the `Gradient`
    operator to match the negative `DiscreteDivergence` operator
    `zero_padding` is assumed.
    """

    def __init__(self, space):
        """Initialize a `DiscreteGradient` operator instance.

        Zero padding is assumed for the negative `DiscreteDivergence`
        operator to match the adjoint of the `DiscreteGradient` operator.

        Parameters
        ----------
        space : `DiscreteLp`
            The space of elements which the operator is acting on
        """

        if not isinstance(space, DiscreteLp):
            raise TypeError('space {!r} is not a `DiscreteLp` '
                            'instance.'.format(space))

        super().__init__(domain=space,
                         range=ProductSpace(space, space.grid.ndim),
                         linear=True)

    def _apply(self, x, out):
        """Apply the gradient operator to ``x`` and store result in ``out``.

        Parameters
        ----------
        x : ``domain`` element
            Input vector to which the `DiscreteGradient` operator is applied to
        out : ``range`` element
            Output vector to which the result is written

        Examples
        --------
        >>> from odl import uniform_discr, FunctionSpace, Rectangle
        >>> data = np.arange(5, dtype=float) * np.arange(1, 3).reshape(2,1)
        >>> data
        array([[ 0.,  1.,  2.,  3.,  4.],
               [ 0.,  2.,  4.,  6.,  8.]])
        >>> space = FunctionSpace(Rectangle([0,0], [2,1]))
        >>> disc = uniform_discr(space, data.shape)
        >>> f = disc.element(data)
        >>> grad = DiscreteGradient(disc)
        >>> grad_f = grad(f)
        >>> print(grad_f[0])
        [[0.0, 1.0, 2.0, 3.0, 4.0],
         [-0.0, -0.5, -1.0, -1.5, -2.0]]
        >>> print(grad_f[0])
        [[0.0, 1.0, 2.0, 3.0, 4.0],
         [-0.0, -0.5, -1.0, -1.5, -2.0]]
        >>> g = grad.range.element((data, data ** 2))
        >>> adj = grad.adjoint
        >>> adj_g = adj(g)
        >>> print(adj_g)
        [[-2.5, -11.0, -22.0, -33.0, 18.5],
         [-10.0, -39.5, -79.0, -118.5, 92.0]]
        >>> g.inner(grad_f) - f.inner(adj_g)
        0.0
        """
        x_data = np.asarray(x)
        ndim = self.domain.grid.ndim
        dx = self.domain.grid.stride

        for axis in range(ndim):
            finite_diff(x_data, out=out[axis].asarray(), axis=axis,
                        dx=dx[axis], edge_order=2, zero_padding=True)

    @property
    def adjoint(self):
        """Return the adjoint operator given by the negative of the
        `DiscreteDivergence` operator assuming implicit zero padding.

        Note that the `space` argument of the `DiscreteDivergence` operator
        is not the range but the domain of the `DiscreteGradient` operator.
        """
        return - DiscreteDivergence(self.domain)


class DiscreteDivergence(Operator):
    """Divergence operator applicable to `DiscreteLp` spaces with number of
    dimensions ``n``.

    Calls helper function `finite_diff` for each component of the input
    product space vector. For the adjoint of the `DiscreteDivergence`
    operator to match the negative `DiscreteGradient` operator implicit zero
    padding is assumed.
    """

    def __init__(self, space):
        """Initialize a `DiscreteDivergence` operator instance.

        Zero padding is assumed for the negative `DiscreteGradient`
        operator to match the adjoint of the `DiscreteDivergence` operator.

        Parameters
        ----------
        space : `DiscreteLp`
            The space of elements which the operator is acting on
        """
        if not isinstance(space, DiscreteLp):
            raise TypeError('space {!r} is not a `DiscreteLp` '
                            'instance.'.format(space))

        self.space = space
        super().__init__(domain=ProductSpace(space, space.grid.ndim),
                         range=space, linear=True)

    def _apply(self, x, out):
        """Apply the divergence operator to ``x`` and store result in ``out``.

        Parameters
        ----------
        x : ``domain`` element
            `ProductSpaceVector` to which the divergence operator is applied to
        out : ``range`` element
            Output vector to which the result is written

        Examples
        --------
        >>> from odl import Rectangle, uniform_discr, FunctionSpace
        >>> data = np.arange(5, dtype=float) * np.arange(1, 3).reshape(2,1)
        >>> space = FunctionSpace(Rectangle([0,0], [2,1]))
        >>> disc = uniform_discr(space, data.shape)
        >>> div = DiscreteDivergence(disc)
        >>> f = div.domain.element([data, data ** 2])
        >>> div_f = div(f)
        >>> print(div_f)
        [[2.5, 11.0, 22.0, 33.0, -18.5],
         [10.0, 39.5, 79.0, 118.5, -92.0]]
        >>> g = div.range.element(data ** 3)
        >>> adj = div.adjoint
        >>> adj_g = adj(g)
        >>> g.inner(div_f)
        -3248.4
        >>> f.inner(adj_g)
        -3248.4
        """

        ndim = self.range.grid.ndim
        dx = self.range.grid.stride

        # TODO: can the following code be optimized?
        tmp = np.zeros_like(out.asarray())
        for axis in range(ndim):
            tmp += finite_diff(x[axis], axis=axis, dx=dx[axis],
                               edge_order=2, zero_padding=True)
        out[:] = tmp

    @property
    def adjoint(self):
        """Return the adjoint operator given by the negative of the
        `DiscreteGradient` operator assuming implicit zero padding.
        """
        return - DiscreteGradient(self.range)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
