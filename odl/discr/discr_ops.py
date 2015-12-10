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


__all__ = ('DiscretePartDeriv', 'DiscreteGradient', 'DiscreteDivergence')


def finite_diff(f, out=None, axis=0, dx=1.0, edge_order=2,
                zero_padding=False):
    """Calculate the partial derivative of ``f`` along a given ``axis``.

    The partial derivative is computed using second-order accurate central
    differences in the interior and either first- or second-order accurate
    one-sides (forward or backward) differences at the boundaries.

    Assuming (implicit) zero padding central differences are used on the
    interior and on endpoints. Otherwise one-sided differences are used. If
    ``zero_padding`` is `False` first-order accuracy can be triggered on
    endpoints with parameter ``edge_order``.

    The returned array has the same shape as the input array ``f``.

    Parameters
    ----------
    f : array-like
         An N-dimensional array
    out : `numpy.ndarray`, optional
         An N-dimensional array to which the output is written.
    axis : `int`, optional
        The axis along which the partial derivative is evaluated. Default: 0
    dx : `float`, optional
        Scalar specifying the sample distances in each dimension ``axis``.
        Default distance: 1.0
    edge_order : {1, 2}, optional
        First-order accurate differences (1) can be used at the boundaries
        if no zero padding is used. Default edge order: 2
    zero_padding : `bool`, optional
        Implicit zero padding. Assumes values outside the domain of ``f`` to be
        zero. Default: `False`

    Returns
    -------
    out : `numpy.ndarray`
        N-dimensional array of the same shape as ``f``, identical with
        input ``out`` if provided.

    Examples
    --------
    >>> f = np.array([ 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    >>> finite_diff(f)
    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
    >>> finite_diff(f, axis=0, dx=1.0, edge_order=2, zero_padding=False)
    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
    >>> finite_diff(f, dx=0.5)
    array([ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.])
    >>> finite_diff(f, zero_padding=True)
    array([ 0.5,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. , -4. ])
    >>> finite_diff(f, zero_padding=False, edge_order=1)
    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
    >>> out = finite_diff(f)
    >>> out is finite_diff(f)
    False
    >>> out is finite_diff(f, out)
    True
    """
    # TODO: implement forward/backward differences
    # TODO: implement alternative boundary conditions

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
        raise ValueError("shape ({0}) of array too small to calculate a "
                         "numerical gradient, at least two elements are "
                         "required.".format(f_data.shape))

    if out is None:
        out = np.empty_like(f_data)
    else:
        if out.shape != f.shape:
            raise ValueError(
                "shape of `out` array ({0}) does not match the shape of "
                "input array `f` ({1}).".format(out.shape, f.shape))

    if edge_order not in [1, 2]:
        raise ValueError("edge order ({0}) not valid".format(edge_order))

    if dx <= 0:
        raise ValueError("step length ({0}) not positive.".format(dx))
    else:
        dx = float(dx)

    # create slice objects: initially all are [:, :, ..., :]
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
            out[slice_out] = f_data[slice_node1] - f_data[slice_node2]

            slice_out[axis] = -1
            slice_node1[axis] = -1
            slice_node2[axis] = -2
            # 1D equivalent: out[-1] = (f[-1] - f[-2])
            out[slice_out] = f_data[slice_node1] - f_data[slice_node2]

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

    def __init__(self, space, axis=0, dx=1.0, edge_order=2,
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
            Scalars specifying the sampling distances in dimension ``axis``.
            Default distance: 1.0
        edge_order : {1, 2}, optional
            First-order accurate differences can be used at the boundaries
            if no zero padding is used. Default edge order: 2
        zero_padding : `bool`, optional
            Implicit zero padding. Assumes values outside the domain of ``f``
            to be zero. Default: `False`
        """

        if not isinstance(space, DiscreteLp):
            raise TypeError('space {!r} is not a `DiscreteLp` '
                            'instance.'.format(space))

        super().__init__(domain=space, range=space, linear=True)
        self.axis = axis
        self.dx = dx
        self.edge_order = edge_order
        self.zero_padding = zero_padding

    def _call(self, x, out=None):
        """Apply gradient operator to ``x`` and store result in ``out``.

        Parameters
        ----------
        x : ``domain`` element
            Input vector to which the operator is applied to
        out : ``range`` element, optional
            Output vector to which the result is written

        Returns
        -------
        out : ``range`` element, optional
            Result of the evaluation, identical with input ``out`` if
            provided.

        Examples
        --------
        >>> from odl import uniform_discr
        >>> data = np.array([[ 0.,  1.,  2.,  3.,  4.],
        ...                  [ 0.,  2.,  4.,  6.,  8.]])
        >>> discr = uniform_discr([0, 0], [2, 1], data.shape)
        >>> par_div = DiscretePartDeriv(discr)
        >>> f = par_div.domain.element(data)
        >>> par_div_f = par_div(f)
        >>> print(par_div_f)
        [[0.0, 1.0, 2.0, 3.0, 4.0],
         [0.0, 1.0, 2.0, 3.0, 4.0]]
        """
        if out is None:
            out = self.range.element()

        # TODO: this pipes CUDA arrays through NumPy. Write native operator.
        out_arr = out.asarray()

        finite_diff(x.asarray(), out=out_arr, axis=self.axis, dx=self.dx,
                    edge_order=self.edge_order,
                    zero_padding=self.zero_padding)

        # self assignment: no overhead in the case asarray is a view
        out[:] = out_arr
        return out

    @property
    def adjoint(self):
        """Return the adjoint operator."""
        raise NotImplementedError('adjoint not implemented')


class DiscreteGradient(Operator):
    """Spatial gradient operator for `DiscreteLp` spaces.

    Calls helper function `finite_diff` to calculate each component of the
    resulting product space vector. For the adjoint of the
    `DiscreteGradient` operator to match the negative `DiscreteDivergence`
    operator ``zero_padding`` is assumed.
    """

    def __init__(self, space):
        """Initialize a `DiscreteGradient` operator instance.

        Zero padding is assumed for the adjoint of the `DiscreteGradient`
        operator to match  negative `DiscreteDivergence` operator.

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

    def _call(self, x, out=None):
        """Calculate the spatial gradient of ``x``.

        Parameters
        ----------
        x : ``domain`` element
            Input vector to which the `DiscreteGradient` operator is
            applied
        out : ``range`` element, optional
            Output vector to which the result is written

        Returns
        -------
        out : ``range`` element, optional
            Result of the evaluation, identical with input ``out`` if
            provided.

        Examples
        --------
        >>> from odl import uniform_discr
        >>> data = np.array([[ 0., 1., 2., 3., 4.],
        ...                  [ 0., 2., 4., 6., 8.]])
        >>> discr = uniform_discr([0,0], [2,5], data.shape)
        >>> f = discr.element(data)
        >>> grad = DiscreteGradient(discr)
        >>> grad_f = grad(f)
        >>> print(grad_f[0])
        [[0.0, 1.0, 2.0, 3.0, 4.0],
         [-0.0, -0.5, -1.0, -1.5, -2.0]]
        >>> print(grad_f[1])
        [[0.5, 1.0, 1.0, 1.0, -1.5],
         [1.0, 2.0, 2.0, 2.0, -3.0]]
        >>> g = grad.range.element((data, data ** 2))
        >>> adj_g = grad.adjoint(g)
        >>> print(adj_g)
        [[-0.5, -3.0, -6.0, -9.0, 0.5],
         [-2.0, -7.5, -15.0, -22.5, 20.0]]
        >>> g.inner(grad_f) - f.inner(adj_g)
        0.0
        """
        if out is None:
            out = self.range.element()

        x_data = x.asarray()
        ndim = self.domain.grid.ndim
        dx = self.domain.grid.stride

        for axis in range(ndim):
            out_arr = out[axis].asarray()

            finite_diff(x_data, out=out_arr, axis=axis,
                        dx=dx[axis], edge_order=2, zero_padding=True)

            out[axis][:] = out_arr

        return out

    @property
    def adjoint(self):
        """Return the adjoint operator.

        Assuming implicit zero padding, the adjoint operator is given by the
        negative of the `DiscreteDivergence` operator

        Note that the ``space`` argument of the `DiscreteDivergence`
        operator is not the range but the domain of the `DiscreteGradient`
        operator.
        """
        return - DiscreteDivergence(self.domain)


class DiscreteDivergence(Operator):
    """Divergence operator for `DiscreteLp` spaces.

    Calls helper function `finite_diff` for each component of the input
    product space vector. For the adjoint of the `DiscreteDivergence`
    operator to match the negative `DiscreteGradient` operator implicit zero
    padding is assumed.
    """

    def __init__(self, space):
        """Initialize a `DiscreteDivergence` operator instance.

        Zero padding is assumed for the adjoint of the `DiscreteDivergence`
        operator to match negative `DiscreteGradient` operator.

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

    def _call(self, x, out=None):
        """Calculate the divergence of ``x``.

        Parameters
        ----------
        x : ``domain`` element
            `ProductSpaceVector` to which the divergence operator is applied to
        out : ``range`` element, optional
            Output vector to which the result is written

        Returns
        -------
        out : ``range`` element, optional
            Result of the evaluation, identical with input ``out`` if
            provided.

        Examples
        --------
        >>> from odl import Rectangle, uniform_discr
        >>> data = np.array([[0., 1., 2., 3., 4.],
        ...                  [1., 2., 3., 4., 5.],
        ...                  [2., 3., 4., 5., 6.]])
        >>> discr = uniform_discr([0, 0], [3, 5], data.shape)
        >>> div = DiscreteDivergence(discr)
        >>> f = div.domain.element([data, data])
        >>> div_f = div(f)
        >>> print(div_f)
        [[1.0, 2.0, 2.5, 3.0, 1.0],
         [2.0, 2.0, 2.0, 2.0, -1.0],
         [1.0, 0.0, -0.5, -1.0, -5.0]]
        >>> g = div.range.element(data ** 2)
        >>> adj_g = div.adjoint(g)
        >>> g.inner(div_f)
        -119.0
        >>> f.inner(adj_g)
        -119.0
        """
        if out is None:
            out = self.range.element()

        ndim = self.range.grid.ndim
        dx = self.range.grid.stride

        arr = out.asarray()
        tmp = np.empty(out.shape, out.dtype, order=out.space.order)
        for axis in range(ndim):
            finite_diff(x[axis], out=tmp, axis=axis, dx=dx[axis],
                        edge_order=2, zero_padding=True)
            if axis == 0:
                arr[:] = tmp
            else:
                arr += tmp

        # self assignment: no overhead in the case asarray is a view
        out[:] = arr
        return out

    @property
    def adjoint(self):
        """Return the adjoint operator.

        Assuming implicit zero padding the adjoint operator is given by the
        negative of the `DiscreteGradient` operator.
        """
        return - DiscreteGradient(self.range)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
