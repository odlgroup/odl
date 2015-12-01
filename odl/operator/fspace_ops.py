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

"""Default operators defined on any `FunctionSpace`."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

# ODL imports
from odl.operator.operator import Operator
from odl.set.pspace import ProductSpace


__all__ = ('Gradient', 'Divergence')


def partial_derivative(f, axis=0, dx=1.0, edge_order=2,
                       zero_padding=False):
    """Calculate the partial derivative of 'f' along direction of 'axis'.

    The number of voxels of `f` is preserved. Assuming (implicit) zero padding
    central differences are used on the interior and on endpoints. Otherwise
    one-sided differences differences are used. In the latter case
    first-order accuracy can be triggered on endpoints with parameter
    'edge_order'. Zero padding is required for the `Divergence` operator to
    match the adjoint of the 'Gradient' operator.

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
    >>> partial_derivative(f)
    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
    >>> partial_derivative(f, dx=0.5)
    array([ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.])
    >>> partial_derivative(f, dx=1.0, zero_padding=True)
    array([ 0.5,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. , -4. ])
    >>> df1 = partial_derivative(np.sin(f/10*np.pi), edge_order=1)
    >>> df2 = partial_derivative(np.sin(f/10*np.pi), edge_order=2)
    >>> np.array_equal(df1[1:-1], df2[1:-1])
    True
    >>> df1[0] == df2[0]
    False
    >>> df1[-1] == df2[-1]
    False
    >>> n = 5
    >>> f = np.arange(n, dtype=float)
    >>> f = f * f.reshape((n,1))
    >>> partial_derivative(f, 0)
    array([[-0.,  1.,  2.,  3.,  4.],
           [ 0.,  1.,  2.,  3.,  4.],
           [ 0.,  1.,  2.,  3.,  4.],
           [ 0.,  1.,  2.,  3.,  4.],
           [ 0.,  1.,  2.,  3.,  4.]])
    >>> partial_derivative(f, 1)
    array([[-0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 2.,  2.,  2.,  2.,  2.],
           [ 3.,  3.,  3.,  3.,  3.],
           [ 4.,  4.,  4.,  4.,  4.]])
    >>> try:
    ...    partial_derivative(f, 2)
    ... except IndexError, e:
    ...    print(e)
    Axis paramater (2) exceeds number of dimensions (2).
    """

    f_data = np.asanyarray(f)
    ndim = f_data.ndim

    # create slice objects --- initially all are [:, :, ..., :]
    # noinspection PyTypeChecker
    slice1 = [slice(None)] * ndim
    # noinspection PyTypeChecker
    slice2 = [slice(None)] * ndim
    # noinspection PyTypeChecker
    slice3 = [slice(None)] * ndim
    # noinspection PyTypeChecker
    slice4 = [slice(None)] * ndim

    try:
        if f_data.shape[axis] < 2:
            raise ValueError("Shape of array too small to calculate a "
                             "numerical gradient, at least two elements are "
                             "required.")
    except IndexError:
        raise IndexError("Axis paramater ({0}) exceeds number of dimensions "
                         "({1}).".format(axis, ndim))

    out = np.empty_like(f_data)

    # Numerical differentiation: 2nd order interior
    slice1[axis] = slice(1, -1)
    slice2[axis] = slice(2, None)
    # noinspection PyTypeChecker
    slice3[axis] = slice(None, -2)
    # 1D equivalent -- out[1:-1] = (y[2:] - y[:-2])/2.0
    out[slice1] = (f_data[slice2] - f_data[slice3]) / 2.0

    # central differences
    if zero_padding:
        # Assume zeros for indices outside the volume

        # 1D equivalent -- out[0] = (y[1] - 0)/2.0
        slice1[axis] = 0
        slice2[axis] = 1
        out[slice1] = f_data[slice2] / 2.0

        # 1D equivalent -- out[-1] = (0 - y[-2])/2.0
        slice1[axis] = -1
        slice3[axis] = -2
        out[slice1] = - f_data[slice3] / 2.0

    # one-side differences
    else:
        # Numerical differentiation: 1st order edges
        if f_data.shape[axis] == 2 or edge_order == 1:

            slice1[axis] = 0
            slice2[axis] = 1
            slice3[axis] = 0
            # 1D equivalent -- out[0] = (y[1] - y[0])
            out[slice1] = (f_data[slice2] - f_data[slice3])

            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            # 1D equivalent -- out[-1] = (y[-1] - y[-2])
            out[slice1] = (f_data[slice2] - f_data[slice3])

        # Numerical differentiation: 2nd order edges
        else:

            slice1[axis] = 0
            slice2[axis] = 0
            slice3[axis] = 1
            slice4[axis] = 2
            # 1D equivalent -- out[0] = -(3*y[0] - 4*y[1] + y[2]) / 2.0
            out[slice1] = -(3.0 * f_data[slice2] - 4.0 * f_data[slice3] +
                            f_data[slice4]) / 2.0

            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            slice4[axis] = -3
            # 1D equivalent -- out[-1] = (3*y[-1] - 4*y[-2] + y[-3]) / 2.0
            out[slice1] = (3.0 * f_data[slice2] - 4.0 * f_data[slice3] +
                           f_data[slice4]) / 2.0

    # divide by step size
    out /= dx

    return out


# noinspection PyAbstractClass
class Gradient(Operator):
    """Gradient operator for any number of dimension.

    Calls function 'partial_derivative' to calculate each component.
    """

    def __init__(self, space, voxel_size=(1,), edge_order=2,
                 zero_padding=True):
        """Initialize a `Gradient` operator instance.

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
        self.voxel_size = voxel_size
        self.edge_order = edge_order
        self.zero_padding = zero_padding
        super().__init__(domain=space,
                         range=ProductSpace(space, len(voxel_size)),
                         linear=True)

    def _apply(self, rhs, out):
        """Apply gradient operator to 'rhs' and store result in 'out'.

        >>> from odl import uniform_discr, FunctionSpace, IntervalProd
        >>> def ndvolume(vol_size, ndim, dtype=None):
        ...     s = [1]
        ...     vol = np.arange(vol_size, dtype=dtype)
        ...     for _ in range(ndim - 1):
        ...         s.insert(0, vol_size)
        ...         vol = vol * vol.reshape(s)
        ...     return vol
        >>> ndim = 2
        >>> vsize = 10
        >>> disc = uniform_discr(FunctionSpace(IntervalProd(
        ... [0.]*ndim, [vsize]*ndim)), [vsize]*ndim)
        >>> f = disc.element(ndvolume(vsize, ndim, np.int32))
        >>> A = Gradient(disc, (1.,)*ndim, zero_padding=True)
        >>> Af = A(f)
        >>> g = A.range.one()
        >>> Adg = A.adjoint(g)
        >>> g.inner(Af) - f.inner(Adg)
        0.0
        >>> B = Divergence(disc, (1.,)*ndim, zero_padding=True)
        >>> Bg = B(g)
        >>> Bdf = B.adjoint(f)
        >>> f.inner(Bg) - g.inner(Bdf)
        0.0
        >>> ndim = 3
        >>> vsize = 9
        >>> disc = uniform_discr(FunctionSpace(IntervalProd(
        ... [0.]*ndim, [vsize]*ndim)), [vsize]*ndim)
        >>> f = disc.element(ndvolume(vsize, ndim, np.int32))
        >>> A = Gradient(disc, (1.,)*ndim, zero_padding=True)
        >>> Af = A(f)
        >>> g = A.range.one()
        >>> Adg = A.adjoint(g)
        >>> g.inner(Af) - f.inner(Adg)
        0.0
        >>> B = Divergence(disc, (1.,)*ndim, zero_padding=True)
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
            out[axis][:] = partial_derivative(rhs_data, axis, dx[axis],
                                              self.edge_order,
                                              self.zero_padding)

    @property
    def adjoint(self):
        """Assuming zero padding, this returns the adjoint operator given by
        the negative of the `Divergence` operator.

        Note that the first argument of the 'Divergence' operator is the
        space the gradient is computed and not its domain. Thus, 'Divergence'
        takes the domain of 'Gradient' as space argument.
        """
        return -Divergence(self.domain, voxel_size=self.voxel_size,
                           edge_order=self.edge_order,
                           zero_padding=self.zero_padding)


# noinspection PyAbstractClass
class Divergence(Operator):
    """Divergence operator for any number of dimensions.

    Calls function 'partial_derivative' for each component of the input
    vector. For `-Divergence` to be the adjoint of `Gradient` requires
    `zero_padding`.
    """

    def __init__(self, space, voxel_size=(1,), edge_order=2,
                 zero_padding=True):
        """Initialize a `Divergence` operator instance.

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
        """Apply `Divergence` operator to `rhs` and store result in `out`.


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
        >>> B = Divergence(disc, (1.,)*ndim, zero_padding=True)
        >>> g = B.domain.one()
        >>> Bg = B(g)
        >>> Bdf = B.adjoint(f)
        >>> f.inner(Bg) - g.inner(Bdf)
        0.0
        """

        tmp = np.zeros_like(rhs[0].asarray())
        for axis in range(tmp.ndim):
            # tmp += self._partial(rhs[nn].asarray(), nn)
            tmp += partial_derivative(rhs[axis].asarray(), axis=axis,
                                      dx=self.voxel_size[axis],
                                      edge_order=self.edge_order,
                                      zero_padding=self.zero_padding)
        out[:] = tmp

    @property
    def adjoint(self):
        """Assuming zero padding, this returns the adjoint operator given by
        the negative of the `Gradient` operator.
        """
        return -Gradient(self.range, voxel_size=self.voxel_size,
                         edge_order=self.edge_order,
                         zero_padding=self.zero_padding)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
