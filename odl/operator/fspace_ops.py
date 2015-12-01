# Copyright 2014, 2015 Jonas Adler
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


__all__ = ('DiscreteGradient', 'DiscreteDivergence')


def discrete_part_deriv(f, axis=0, dx=1.0, edge_order=2,
                        zero_padding=False):
    """Calculate the partial derivative of `f` along a given `axis` using
    first-order discrete differences.

    The partial derivative is computed using second order accurate central
    differences in the interior and either first differences or second order
    accurate one-sides (forward or backwards) differences at the boundaries.

    Assuming (implicit) zero padding central differences are used on the
    interior and on endpoints. Otherwise one-sided differences differences
    are used. Then first-order accuracy can be triggered on endpoints with
    parameter `edge_order`.

    The returned array has the same shape as the input array `f`.

    Parameters
    ----------
    f : `numpy.ndarray`
         An N-dimensional array
    axis : `int`, optional
        The axis along which the partial derivative is evaluated. Default: 0
    dx : `float`, optional
        Scalars specifying the sample distances in each dimension `axis`.
        Default distance: 1.
    edge_order : {1, 2}, optional
        Partial derivative is calculated using Nth order accurate
        differences at the boundaries. Default edge order: 2
    zero_padding : `bool`, optional
        Implicit zero padding. Assumes values outside the domain of f to be
        zero. Default: False

    Returns
    -------
    out : `numpy.ndarray`
        Returns an N-dimensional array of the same shape as `f`

    Examples
    --------
    >>> f = np.arange(10, dtype=float)
    >>> discrete_part_deriv(f)
    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
    >>> discrete_part_deriv(f, dx=0.5)
    array([ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.])
    >>> discrete_part_deriv(f, dx=1.0, zero_padding=True)
    array([ 0.5,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. , -4. ])
    >>> df1 = discrete_part_deriv(np.sin(f/10*np.pi), edge_order=1)
    >>> df2 = discrete_part_deriv(np.sin(f/10*np.pi), edge_order=2)
    >>> np.array_equal(df1[1:-1], df2[1:-1])
    True
    >>> df1[0] == df2[0]
    False
    >>> df1[-1] == df2[-1]
    False
    >>> n = 5
    >>> f = np.arange(n, dtype=float)
    >>> f = f * f.reshape((n,1))
    >>> discrete_part_deriv(f, 0)
    array([[-0.,  1.,  2.,  3.,  4.],
           [ 0.,  1.,  2.,  3.,  4.],
           [ 0.,  1.,  2.,  3.,  4.],
           [ 0.,  1.,  2.,  3.,  4.],
           [ 0.,  1.,  2.,  3.,  4.]])
    >>> discrete_part_deriv(f, 1)
    array([[-0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 2.,  2.,  2.,  2.,  2.],
           [ 3.,  3.,  3.,  3.,  3.],
           [ 4.,  4.,  4.,  4.,  4.]])
    >>> try:
    ...    discrete_part_deriv(f, 2)
    ... except IndexError as e:
    ...    print(e)
    Axis paramater (2) exceeds number of dimensions (2).
    """

    f_data = np.asanyarray(f)
    ndim = f_data.ndim

    if not 0 <= axis < ndim:
        raise IndexError("Axis paramater ({0}) exceeds number of dimensions "
                         "({1}).".format(axis, ndim))

    if f_data.shape[axis] < 2:
            raise ValueError("Shape of array too small to calculate a "
                             "numerical gradient, at least two elements are "
                             "required.")

    # create slice objects --- initially all are [:, :, ..., :]
    # current slice
    slice_out = [slice(None)] * ndim
    # slices used to calculate finite differences
    slice_node1 = [slice(None)] * ndim
    slice_node2 = [slice(None)] * ndim
    slice_node3 = [slice(None)] * ndim

    out = np.empty_like(f_data)

    # Numerical differentiation: 2nd order interior
    slice_out[axis] = slice(1, -1)
    slice_node1[axis] = slice(2, None)
    slice_node2[axis] = slice(None, -2)
    # 1D equivalent -- out[1:-1] = (f[2:] - f[:-2])/2.0
    out[slice_out] = (f_data[slice_node1] - f_data[slice_node2]) / 2.0

    # central differences
    if zero_padding:
        # Assume zeros for indices outside the domain of `f`

        # 1D equivalent -- out[0] = (f[1] - 0)/2.0
        slice_out[axis] = 0
        slice_node1[axis] = 1
        out[slice_out] = f_data[slice_node1] / 2.0

        # 1D equivalent -- out[-1] = (0 - f[-2])/2.0
        slice_out[axis] = -1
        slice_node2[axis] = -2
        out[slice_out] = - f_data[slice_node2] / 2.0

    # one-side differences
    else:
        # Numerical differentiation: 1st order edges
        if f_data.shape[axis] == 2 or edge_order == 1:

            # 1D equivalent -- out[0] = (f[1] - f[0])
            slice_out[axis] = 0
            slice_node1[axis] = 1
            slice_node2[axis] = 0
            out[slice_out] = (f_data[slice_node1] - f_data[slice_node2])

            # 1D equivalent -- out[-1] = (f[-1] - f[-2])
            slice_out[axis] = -1
            slice_node1[axis] = -1
            slice_node2[axis] = -2
            out[slice_out] = (f_data[slice_node1] - f_data[slice_node2])

        # Numerical differentiation: 2nd order edges
        else:

            # 1D equivalent -- out[0] = -(3*f[0] - 4*f[1] + f[2]) / 2.0
            slice_out[axis] = 0
            slice_node1[axis] = 0
            slice_node2[axis] = 1
            slice_node3[axis] = 2
            out[slice_out] = -(3.0 * f_data[slice_node1] - 4.0 * f_data[
                slice_node2] + f_data[slice_node3]) / 2.0

            # 1D equivalent -- out[-1] = (3*f[-1] - 4*f[-2] + f[-3]) / 2.0
            slice_out[axis] = -1
            slice_node1[axis] = -1
            slice_node2[axis] = -2
            slice_node3[axis] = -3
            out[slice_out] = (3.0 * f_data[slice_node1] - 4.0 * f_data[
                slice_node2] + f_data[slice_node3]) / 2.0

    # divide by step size
    out /= dx

    return out


class DiscreteGradient(Operator):
    """Gradient operator for any number of dimensions `n`.

    Calls helper function `discrete_part_deriv` to calculate each component
    of the resulting product space vector. For the adjoint of the `Gradient`
    operator to match the negative `DiscreteDivergence` operator requires
    `zero_padding`.
    """

    def __init__(self, space, voxel_size=(1,), edge_order=2,
                 zero_padding=True):
        """Initialize a `DiscreteGradient` operator instance.

        Zero padding is required for the negative `DiscreteDivergence`
        operator to match the adjoint of the `DiscreteGradient` operator.

        Parameters
        ----------
        space : `FunctionSpace`
            The space of elements which the operator is acting on
        voxel_size : `n`-tuple of `floats`
            `n`-tuple of scalars specifying the sample distances in each
            dimension. Default distance is 1 in each direction
        edge_order : {1, 2}, optional
            Partial derivative is calculated using Nth order accurate
            differences at the boundaries. Default edge order: 2
        zero_padding : `bool`, optional
            Implicit zero padding. Assumes values outside of `space` to be
            zero. Default: True
        """
        self.voxel_size = voxel_size
        self.edge_order = edge_order
        self.zero_padding = zero_padding
        super().__init__(domain=space,
                         range=ProductSpace(space, len(voxel_size)),
                         linear=True)

    def _apply(self, rhs, out):
        """Apply gradient operator to `rhs` and store result in `out`.

        Parameters
        ----------
        rhs : ``domain`` element
            Input vector to which the `DiscreteGradient` operator is applied to
        out : ``range`` element
            Output vector to which the result is written

        Examples
        --------
        >>> from odl import uniform_discr, FunctionSpace, IntervalProd
        >>> def ndvolume(vol_size, ndim, dtype=None):
        ...     s = [1]
        ...     vol = np.arange(vol_size, dtype=dtype)
        ...     for _ in range(ndim - 1):
        ...         s.insert(0, vol_size)
        ...         vol = vol * vol.reshape(s)
        ...     return vol
        >>> ndim = 3
        >>> vsize = 100
        >>> disc = uniform_discr(FunctionSpace(IntervalProd(
        ... [0.]*ndim, [vsize]*ndim)), [vsize]*ndim)
        >>> f = disc.element(ndvolume(vsize, ndim, np.int32))
        >>> A = DiscreteGradient(disc, (1.,)*ndim, zero_padding=True)
        >>> Af = A(f)
        >>> g = A.range.one()
        >>> Adg = A.adjoint(g)
        >>> g.inner(Af) - f.inner(Adg)
        0.0
        >>> B = DiscreteDivergence(disc, (1.,)*ndim, zero_padding=True)
        >>> Bg = B(g)
        >>> Bdf = B.adjoint(f)
        >>> f.inner(Bg) - g.inner(Bdf)
        0.0
        >>> ndim = 3
        >>> vsize = 9
        >>> disc = uniform_discr(FunctionSpace(IntervalProd(
        ... [0.]*ndim, [vsize]*ndim)), [vsize]*ndim)
        >>> f = disc.element(ndvolume(vsize, ndim, np.int32))
        >>> A = DiscreteGradient(disc, (1.,)*ndim, zero_padding=True)
        >>> Af = A(f)
        >>> g = A.range.one()
        >>> Adg = A.adjoint(g)
        >>> g.inner(Af) - f.inner(Adg)
        0.0
        >>> B = DiscreteDivergence(disc, (1.,)*ndim, zero_padding=True)
        >>> Bg = B(g)
        >>> Bdf = B.adjoint(f)
        >>> f.inner(Bg) - g.inner(Bdf)
        0.0

        """
        rhs_data = np.asanyarray(rhs)
        ndim = rhs_data.ndim

        dx = self.voxel_size
        if np.size(dx) == 1:
            dx = [dx for _ in range(ndim)]

        for axis in range(ndim):
            out[axis][:] = discrete_part_deriv(rhs_data, axis, dx[axis],
                                               self.edge_order,
                                               self.zero_padding)

    @property
    def adjoint(self):
        """Assuming zero padding, this returns the adjoint operator given by
        the negative of the `DiscreteDivergence` operator.

        Note that the first argument (`space`) of the `DiscreteDivergence`
        operator is not the range but the  domain of the `DiscreteGradient`
        operator.
        """
        return -DiscreteDivergence(self.domain, voxel_size=self.voxel_size,
                                   edge_order=self.edge_order,
                                   zero_padding=self.zero_padding)


class DiscreteDivergence(Operator):
    """Divergence operator for any number of dimensions.

    Calls helper function `discrete_part_deriv` for each component of the
    input product space vector vector. For the adjoint of the
    `DiscreteDivergence` operator to match the negative `DiscreteGradient`
    operator requires `zero_padding`.
    """

    def __init__(self, space, voxel_size=(1,), edge_order=2,
                 zero_padding=True):
        """Initialize a `DiscreteDivergence` operator instance.

        Zero padding is required for the negative `DiscreteGradient`
        operator to match the adjoint of the `DiscreteDivergence` operator.

        Parameters
        ----------
        space : `FunctionSpace`
            The space of elements which the operator is acting on
        voxel_size : n-tuple of `floats`
            n-tuple of scalars specifying the sample distances in each
            dimension. Default distance is 1 in each direction
        edge_order : {1, 2}, optional
            Partial derivative is calculated using Nth order accurate
            differences at the boundaries. Default edge order: 2
        zero_padding : `bool`, optional
            Implicit zero padding. Assumes values outside of `space` to be
            zero. Default: True
        """

        self.space = space
        self.voxel_size = voxel_size
        self.edge_order = edge_order
        self.zero_padding = zero_padding
        super().__init__(domain=ProductSpace(space, len(voxel_size)),
                         range=space, linear=True)

    def _apply(self, rhs, out):
        """Apply `DiscreteDivergence` operator to `rhs` and store result in
        `out`.

        Parameters
        ----------
        rhs : ``domain`` element
            `ProductSpaceVector` to which the `DiscreteDivergence` operator is
            applied to
        out : ``range`` element
            Output vector to which the result is written

        Examples
        --------
        >>> from odl import uniform_discr, FunctionSpace, IntervalProd
        >>> def ndvolume(vol_size, ndim, dtype=None):
        ...     s = [1]
        ...     vol = np.arange(vol_size, dtype=dtype)
        ...     for _ in range(ndim - 1):
        ...         s.insert(0, vol_size)
        ...         vol = vol * vol.reshape(s)
        ...     return vol
        >>> ndim = 4
        >>> vsize = 6
        >>> disc = uniform_discr(FunctionSpace(IntervalProd(
        ... [0.]*ndim, [vsize]*ndim)), [vsize]*ndim)
        >>> f = disc.element(ndvolume(vsize, ndim, np.int32))
        >>> B = DiscreteDivergence(disc, (1.,)*ndim, zero_padding=True)
        >>> g = B.domain.one()
        >>> Bg = B(g)
        >>> Bdf = B.adjoint(f)
        >>> f.inner(Bg) - g.inner(Bdf)
        0.0
        """

        tmp = np.zeros_like(rhs[0].asarray())
        for axis in range(tmp.ndim):
            tmp += discrete_part_deriv(rhs[axis].asarray(), axis=axis,
                                       dx=self.voxel_size[axis],
                                       edge_order=self.edge_order,
                                       zero_padding=self.zero_padding)
        out[:] = tmp

    @property
    def adjoint(self):
        """Assuming zero padding, this returns the adjoint operator given by
        the negative of the `DiscreteGradient` operator.
        """
        return -DiscreteGradient(self.range, voxel_size=self.voxel_size,
                                 edge_order=self.edge_order,
                                 zero_padding=self.zero_padding)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
