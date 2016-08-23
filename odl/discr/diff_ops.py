﻿# Copyright 2014-2016 The ODL development group
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

"""Operators defined for tensor fields."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

from odl.discr.lp_discr import DiscreteLp
from odl.discr.tensor_ops import PointwiseTensorFieldOperator
from odl.space import ProductSpace


__all__ = ('PartialDerivative', 'Gradient', 'Divergence', 'Laplacian')

_SUPPORTED_DIFF_METHODS = ('central', 'forward', 'backward')
_SUPPORTED_PADDING_METHODS = ('constant', 'symmetric', 'symmetric_adjoint',
                              'periodic', None)
_ADJ_METHOD = {'central': 'central',
               'forward': 'backward',
               'backward': 'forward'}
_ADJ_PADDING = {'constant': 'constant',
                'symmetric': 'symmetric_adjoint',
                'symmetric_adjoint': 'symmetric',
                'periodic': 'periodic'}


class PartialDerivative(PointwiseTensorFieldOperator):

    """Calculate the discrete partial derivative along a given axis.

    Calls helper function `finite_diff` to calculate finite difference.
    Preserves the shape of the underlying grid.
    """

    def __init__(self, space, axis=0, method='forward', pad_mode=None,
                 pad_const=0, edge_order=None):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp`
            Space of elements which the operator is acting on.
        axis : `int`, optional
            Axis along which the partial derivative is evaluated.
        method : {'central', 'forward', 'backward'}, optional
            Finite difference method which is used in the interior of the
            domain of ``f``.
        pad_mode : {'constant', 'symmetric', 'periodic', None}, optional

            'constant' : Pads values outside the domain of ``f`` with a
            constant value given by ``pad_const``.

            'symmetric' : Pads with the reflection of the vector mirrored
            along the edge of the array.

            'periodic' : Pads with the values from the other side of the array.

            If `None` is given, one-sided forward or backward differences
            are used at the boundary.

        pad_const : `float`, optional
            If ``pad_mode`` is 'constant', ``f`` assumes
            ``pad_const`` for indices outside the domain of ``f``
        edge_order : {1, 2}, optional
            Edge-order accuracy at the boundaries if no padding is used. If
            `None` the edge-order accuracy at endpoints corresponds to the
            accuracy in the interior.
        """
        if not isinstance(space, DiscreteLp):
            raise TypeError('`space` {!r} is not a DiscreteLp instance'
                            ''.format(space))

        # Method is affine if nonzero padding is given.
        linear = not (pad_mode == 'constant' and pad_const != 0)
        super().__init__(domain=space, range=space, linear=linear)
        self.axis = axis
        self.dx = space.cell_sides[axis]
        self.method = method
        self.pad_mode = pad_mode
        self.pad_const = pad_const
        self.edge_order = edge_order

    def _call(self, x, out=None):
        """Apply gradient operator to ``x`` and store result in ``out``.

        Parameters
        ----------
        x : ``domain`` `element`
            Input vector to which the operator is applied to
        out : ``range`` element, optional
            Output vector to which the result is written

        Returns
        -------
        out : ``range`` `element`
            Result of the evaluation. If ``out`` is provided, the
            returned object is a reference to it.

        Examples
        --------
        >>> data = np.array([[ 0.,  1.,  2.,  3.,  4.],
        ...                  [ 0.,  2.,  4.,  6.,  8.]])
        >>> discr = odl.uniform_discr([0, 0], [2, 1], data.shape)
        >>> par_deriv = PartialDerivative(discr)
        >>> f = par_deriv.domain.element(data)
        >>> par_div_f = par_deriv(f)
        >>> print(par_div_f)
        [[0.0, 1.0, 2.0, 3.0, 4.0],
         [0.0, 1.0, 2.0, 3.0, 4.0]]
        """
        if out is None:
            out = self.range.element()

        # TODO: this pipes CUDA arrays through NumPy. Write native operator.
        out_arr = out.asarray()
        finite_diff(x.asarray(), out=out_arr, axis=self.axis, dx=self.dx,
                    method=self.method, pad_mode=self.pad_mode,
                    pad_const=self.pad_const,
                    edge_order=self.edge_order)

        # self assignment: no overhead in the case out_arr is a view
        out[:] = out_arr
        return out

    def derivative(self, point=None):
        """Return the derivative operator.

        The partial derivative is usually linear, but in case the 'constant'
        ``pad_mode`` is used with nonzero ``pad_const``, the
        derivative is given by the derivative with 0 ``pad_const``.

        Parameters
        ----------
        point : ``domain`` element, optional
            The point to take the derivative in. Does not change the result
            since the operator is affine.
        """
        if self.pad_mode == 'constant' and self.pad_const != 0:
            return PartialDerivative(self.domain, self.axis, self.method,
                                     self.pad_mode, 0, self.edge_order)
        else:
            return self

    @property
    def adjoint(self):
        """Return the adjoint operator."""
        if not self.is_linear:
            raise ValueError('operator with nonzero pad_const ({}) is not'
                             ' linear and has no adjoint'
                             ''.format(self.pad_const))

        return -PartialDerivative(self.domain, self.axis,
                                  _ADJ_METHOD[self.method],
                                  _ADJ_PADDING[self.pad_mode],
                                  self.pad_const, self.edge_order)


class Gradient(PointwiseTensorFieldOperator):

    """Spatial gradient operator for `DiscreteLp` spaces.

    Calls helper function `finite_diff` to calculate each component of the
    resulting product space vector. For the adjoint of the `Gradient`
    operator, zero padding is assumed to match the negative `Divergence`
    operator
    """

    def __init__(self, domain=None, range=None, method='forward',
                 pad_mode='constant', pad_const=0):
        """Initialize a new instance.

        Zero padding is assumed for the adjoint of the `Gradient`
        operator to match negative `Divergence` operator.

        Parameters
        ----------
        domain : `DiscreteLp`, optional
            Space of elements which the operator acts on.
            This is required if ``range`` is not given.
        range : power space of `DiscreteLp`, optional
            Space of elements to which the operator maps.
            This is required if ``domain`` is not given.
        method : {'central', 'forward', 'backward'}, optional
            Finite difference method to be used
        pad_mode : {'constant', 'symmetric', 'periodic'}, optional

            'constant' : Pads values outside the domain of ``f`` with a
            constant value given by ``pad_const``.

            'symmetric' : Pads with the reflection of the vector mirrored
            along the edge of the array.

            'periodic' : Pads with the values from the other side of the array.

        pad_const : `float`, optional
            If ``pad_mode`` is 'constant', ``f`` assumes
            ``pad_const`` for indices outside the domain of ``f``.

        Examples
        --------
        >>> dom = odl.uniform_discr([0, 0], [1, 1], (10, 20))
        >>> ran = odl.ProductSpace(dom, dom.ndim)  # 2-dimensional
        >>> grad_op = Gradient(dom)
        >>> grad_op.range == ran
        True
        >>> grad_op2 = Gradient(range=ran)
        >>> grad_op2.domain == dom
        True
        >>> grad_op3 = Gradient(domain=dom, range=ran)
        >>> grad_op3.domain == dom
        True
        >>> grad_op3.range == ran
        True
        """
        if domain is None and range is None:
            raise ValueError('either `domain` or `range` must be specified')

        if domain is None:
            if not isinstance(range, ProductSpace):
                raise TypeError('`range` {!r} is not a ProductSpace instance'
                                ''.format(range))
            domain = range[0]

        if range is None:
            if not isinstance(domain, DiscreteLp):
                raise TypeError('`domain` {!r} is not a `DiscreteLp` '
                                'instance'.format(domain))
            range = ProductSpace(domain, domain.ndim)

        linear = not (pad_mode == 'constant' and pad_const != 0)
        super().__init__(domain, range, linear=linear)
        self.method = method
        self.pad_mode = pad_mode
        self.pad_const = pad_const

    def _call(self, x, out=None):
        """Calculate the spatial gradient of ``x``.

        Parameters
        ----------
        x : ``domain`` `element`
            Input vector to which the `Gradient` operator is applied
        out : ``range`` `element`, optional
            Output vector to which the result is written

        Returns
        -------
        out : ``range`` `element`
            Result of the evaluation. If ``out`` is provided, the returned
            object is a reference to it.

        Examples
        --------
        >>> data = np.array([[ 0., 1., 2., 3., 4.],
        ...                  [ 0., 2., 4., 6., 8.]])
        >>> discr = odl.uniform_discr([0, 0], [2, 5], data.shape)
        >>> f = discr.element(data)
        >>> grad = Gradient(discr)
        >>> grad_f = grad(f)
        >>> print(grad_f[0])
        [[0.0, 1.0, 2.0, 3.0, 4.0],
         [0.0, -2.0, -4.0, -6.0, -8.0]]
        >>> print(grad_f[1])
        [[1.0, 1.0, 1.0, 1.0, -4.0],
         [2.0, 2.0, 2.0, 2.0, -8.0]]

        Verify adjoint:

        >>> g = grad.range.element((data, data ** 2))
        >>> adj_g = grad.adjoint(g)
        >>> print(adj_g)
        [[0.0, -2.0, -5.0, -8.0, -11.0],
         [0.0, -5.0, -14.0, -23.0, -32.0]]
        >>> g.inner(grad_f) / f.inner(adj_g)
        1.0
        """
        if out is None:
            out = self.range.element()

        x_arr = x.asarray()
        ndim = self.domain.ndim
        dx = self.domain.cell_sides

        for axis in range(ndim):
            out_arr = out[axis].asarray()

            finite_diff(x_arr, axis=axis, dx=dx[axis], method=self.method,
                        pad_mode=self.pad_mode,
                        pad_const=self.pad_const,
                        out=out_arr)

            out[axis][:] = out_arr

        return out

    def derivative(self, point=None):
        """Return the derivative operator.

        The gradient is usually linear, but in case the 'constant'
        ``pad_mode`` is used with nonzero ``pad_const``, the
        derivative is given by the Gradient with ``pad_const=0``.

        Parameters
        ----------
        point : ``domain`` element, optional
            The point to take the derivative in. Does not change the result
            since the operator is affine.
        """
        if self.pad_mode == 'constant' and self.pad_const != 0:
            return Gradient(self.domain, self.range, self.method,
                            pad_mode=self.pad_mode,
                            pad_const=0)
        else:
            return self

    @property
    def adjoint(self):
        """Adjoint of this operator.

        The adjoint is given by the negative `Divergence` with corrections for
        the method and padding.

        The Divergence is constructed from a ``space`` as a product space
        operator ``space^n --> space``, hence we need to provide the domain of
        this operator.
        """
        if not self.is_linear:
            raise ValueError('operator with nonzero pad_const ({}) is not'
                             ' linear and has no adjoint'
                             ''.format(self.pad_const))

        return - Divergence(domain=self.range, range=self.domain,
                            method=_ADJ_METHOD[self.method],
                            pad_mode=_ADJ_PADDING[self.pad_mode],
                            pad_const=self.pad_const)


class Divergence(PointwiseTensorFieldOperator):

    """Divergence operator for `DiscreteLp` spaces.

    Calls helper function `finite_diff` for each component of the input
    product space vector. For the adjoint of the `Divergence` operator to
    match the negative `Gradient` operator implicit zero is assumed.
    """

    def __init__(self, domain=None, range=None, method='forward',
                 pad_mode='constant', pad_const=0):
        """Initialize a new instance.

        Zero padding is assumed for the adjoint of the `Divergence`
        operator to match the negative `Gradient` operator.

        Parameters
        ----------
        domain : power space of `DiscreteLp`, optional
            Space of elements which the operator acts on.
            This is required if ``range`` is not given.
        range : `DiscreteLp`, optional
            Space of elements to which the operator maps.
            This is required if ``domain`` is not given.
        method : {'central', 'forward', 'backward'}, optional
            Finite difference method to be used
        pad_mode : {'constant', 'symmetric', 'periodic'}, optional

            'constant' : Pads values outside the domain of ``f`` with a
            constant value given by ``pad_const``.

            'symmetric' : Pads with the reflection of the vector mirrored
            along the edge of the array.

            'periodic' : Pads with the values from the other side of the array.

        pad_const : `float`, optional
            If ``pad_mode`` is 'constant', ``f`` assumes
            ``pad_const`` for indices outside the domain of ``f``.

        Examples
        --------
        >>> ran = odl.uniform_discr([0, 0], [1, 1], (10, 20))
        >>> dom = odl.ProductSpace(ran, ran.ndim)  # 2-dimensional
        >>> div_op = Divergence(dom)
        >>> div_op.range == ran
        True
        >>> div_op2 = Divergence(range=ran)
        >>> div_op2.domain == dom
        True
        >>> div_op3 = Divergence(domain=dom, range=ran)
        >>> div_op3.domain == dom
        True
        >>> div_op3.range == ran
        True
        """
        if domain is None and range is None:
            raise ValueError('either `domain` or `range` must be specified')

        if domain is None:
            if not isinstance(range, DiscreteLp):
                raise TypeError('`range` {!r} is not a DiscreteLp instance'
                                ''.format(range))
            domain = ProductSpace(range, range.ndim)

        if range is None:
            if not isinstance(domain, ProductSpace):
                raise TypeError('`domain` {!r} is not a ProductSpace instance'
                                ''.format(domain))
            range = domain[0]

        linear = not (pad_mode == 'constant' and pad_const != 0)
        super().__init__(domain, range, linear=linear)
        self.method = method
        self.pad_mode = pad_mode
        self.pad_const = pad_const

    def _call(self, x, out=None):
        """Calculate the divergence of ``x``.

        Parameters
        ----------
        x : ``domain`` `element`
            `ProductSpaceVector` to which the divergence operator
            is applied
        out : ``range`` `element`, optional
            Output vector to which the result is written

        Returns
        -------
        out : ``range`` `element`
            Result of the evaluation. If ``out`` is provided, the returned
            object is a reference to it.

        Examples
        --------
        >>> data = np.array([[0., 1., 2., 3., 4.],
        ...                  [1., 2., 3., 4., 5.],
        ...                  [2., 3., 4., 5., 6.]])
        >>> space = odl.uniform_discr([0, 0], [3, 5], data.shape)
        >>> div = Divergence(range=space)
        >>> f = div.domain.element([data, data])
        >>> div_f = div(f)
        >>> print(div_f)
        [[2.0, 2.0, 2.0, 2.0, -3.0],
         [2.0, 2.0, 2.0, 2.0, -4.0],
         [-1.0, -2.0, -3.0, -4.0, -12.0]]

        Verify adjoint:

        >>> g = div.range.element(data ** 2)
        >>> adj_div_g = div.adjoint(g)
        >>> g.inner(div_f) / f.inner(adj_div_g)
        1.0
        """
        if out is None:
            out = self.range.element()

        ndim = self.range.ndim
        dx = self.range.cell_sides

        out_arr = out.asarray()
        tmp = np.empty(out.shape, out.dtype, order=out.space.order)
        for axis in range(ndim):
            finite_diff(x[axis], axis=axis, dx=dx[axis], method=self.method,
                        pad_mode=self.pad_mode,
                        pad_const=self.pad_const,
                        out=tmp)
            if axis == 0:
                out_arr[:] = tmp
            else:
                out_arr += tmp

        # self assignment: no overhead in the case asarray is a view
        out[:] = out_arr
        return out

    def derivative(self, point=None):
        """Return the derivative operator.

        The Divergence is usually linear, but in case the 'constant'
        ``pad_mode`` is used with nonzero ``pad_const``, the
        derivative is given by the Divergence with ``pad_const=0``.

        Parameters
        ----------
        point : ``domain`` element, optional
            The point to take the derivative in. Does not change the result
            since the operator is affine.
        """
        if self.pad_mode == 'constant' and self.pad_const != 0:
            return Divergence(self.domain, self.range, self.method,
                              pad_mode=self.pad_mode,
                              pad_const=0)
        else:
            return self

    @property
    def adjoint(self):
        """Adjoint of this operator.

        The adjoint is given by the negative `Gradient` with corrections for
        the method and padding.
        """
        if not self.is_linear:
            raise ValueError('operator with nonzero pad_const ({}) is not'
                             ' linear and has no adjoint'
                             ''.format(self.pad_const))

        return - Gradient(domain=self.range, range=self.domain,
                          method=_ADJ_METHOD[self.method],
                          pad_mode=_ADJ_PADDING[self.pad_mode])


class Laplacian(PointwiseTensorFieldOperator):

    """Spatial Laplacian operator for `DiscreteLp` spaces.

    Calls helper function `finite_diff` to calculate each component of the
    resulting product space vector.

    Outside the domain zero padding is assumed.
    """

    def __init__(self, space, pad_mode='constant', pad_const=0):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp`
            Space of elements which the operator is acting on.
        pad_mode : {'constant', 'symmetric', 'periodic'}, optional

            'constant' : Pads values outside the domain of ``f`` with a
            constant value given by ``pad_const``.

            'symmetric' : Pads with the reflection of the vector mirrored
            along the edge of the array.

            'periodic' : Pads with the values from the other side of the array.

        pad_const : `float`, optional
            If ``pad_mode`` is 'constant', ``f`` assumes
            ``pad_const`` for indices outside the domain of ``f``.
        """
        if not isinstance(space, DiscreteLp):
            raise TypeError('`space` {!r} is not a DiscreteLp instance'
                            ''.format(space))
        super().__init__(domain=space, range=space, linear=True)
        self.pad_mode = pad_mode
        self.pad_const = pad_const

    def _call(self, x, out=None):
        """Calculate the spatial Laplacian of ``x``.

        Parameters
        ----------
        x : ``domain`` `element`
            Input vector to which the `Laplacian` operator is
            applied
        out : ``range`` `element`, optional
            Output vector to which the result is written

        Returns
        -------
        out : ``range`` `element`
            Result of the evaluation. If ``out`` is provided, the returned
            object is a reference to it.

        Examples
        --------
        >>> data = np.array([[ 0., 0., 0.],
        ...                  [ 0., 1., 0.],
        ...                  [ 0., 0., 0.]])
        >>> space = odl.uniform_discr([0, 0], [3, 3], data.shape)
        >>> f = space.element(data)
        >>> lap = Laplacian(space)
        >>> print(lap(f))
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]]
        """
        if out is None:
            out = self.range.zero()
        else:
            out.set_zero()

        x_arr = x.asarray()
        out_arr = out.asarray()
        tmp = np.empty(out.shape, out.dtype, order=out.space.order)

        ndim = self.domain.ndim
        dx = self.domain.cell_sides

        for axis in range(ndim):
            # TODO: this can be optimized

            finite_diff(x_arr, axis=axis, dx=dx[axis] ** 2,
                        method='forward',
                        pad_mode=self.pad_mode,
                        pad_const=self.pad_const, out=tmp)

            out_arr[:] += tmp

            finite_diff(x_arr, axis=axis, dx=dx[axis] ** 2,
                        method='backward',
                        pad_mode=self.pad_mode,
                        pad_const=self.pad_const, out=tmp)

            out_arr[:] -= tmp

        out[:] = out_arr
        return out

    def derivative(self, point=None):
        """Return the derivative operator.

        The Laplacian is usually linear, but in case the 'constant'
        ``pad_mode`` is used with nonzero ``pad_const``, the
        derivative is given by the derivative with 0 ``pad_const``.

        Parameters
        ----------
        point : ``domain`` element
            The point to take the derivative in. Does not change the result
            since the operator is affine.
        """
        if self.pad_mode == 'constant' and self.pad_const != 0:
            return Laplacian(self.domain,
                             pad_mode=self.pad_mode,
                             pad_const=0)
        else:
            return self

    @property
    def adjoint(self):
        """Return the adjoint operator.

        The laplacian is self-adjoint, so this returns ``self``.
        """
        return self


def finite_diff(f, axis=0, dx=1.0, method='forward', out=None, **kwargs):
    """Calculate the partial derivative of ``f`` along a given ``axis``.

    In the interior of the domain of f, the partial derivative is computed
    using first-order accurate forward or backward difference or
    second-order accurate central differences.

    With padding the same method and thus accuracy is used on endpoints as
    in the interior i.e. forward and backward differences use first-order
    accuracy on edges while central differences use second-order accuracy at
    edges.

    Without padding one-sided forward or backward differences are used at
    the boundaries. The accuracy at the endpoints can then also be
    triggered by the edge order.

    The returned array has the same shape as the input array ``f``.

    Per default forward difference with dx=1 and no padding is used.

    Parameters
    ----------
    f : `array-like`
         An N-dimensional array.
    axis : `int`, optional
        The axis along which the partial derivative is evaluated.
    dx : `float`, optional
        Scalar specifying the distance between sampling points along ``axis``.
    method : {'central', 'forward', 'backward'}, optional
        Finite difference method which is used in the interior of the domain
         of ``f``.
    pad_mode : {'constant', 'symmetric', 'periodic', None}, optional

        'constant' : Pads values outside the domain of ``f`` with a constant
        value given by ``pad_const``.

        'symmetric' : Pads with the reflection of the vector mirrored
        along the edge of the array.

        'periodic' : Pads with the values from the other side of the array.

        If `None` is given, one-sided forward or backward differences
        are used at the boundary.

    pad_const : `float`, optional
        If ``pad_mode`` is 'constant', ``f`` assumes ``pad_const``
        for indices outside the domain of ``f``.
    edge_order : {1, 2}, optional
        Edge-order accuracy at the boundaries if no padding is used. If
        `None` the edge-order accuracy at endpoints corresponds to the
        accuracy in the interior. Default: `None`
    out : `numpy.ndarray`, optional
         An N-dimensional array to which the output is written. Has to have
         the same shape as the input array ``f``. Default: `None`

    Returns
    -------
    out : `numpy.ndarray`
        N-dimensional array of the same shape as ``f``. If ``out`` is
        provided, the returned object is a reference to it.

    Notes
    -----
    Without padding the use of second-order accurate edges requires at
    least three elements.

    Central differences with padding cannot be used with first-order
    accurate edges.

    Forward and backward differences with padding use the first-order
    accuracy on edges (as in the interior).

    An edge-order accuracy different from the interior can only be triggered
    without padding i.e. when one-sided differences are used at the edges.

    Examples
    --------
    >>> f = np.array([ 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])

    >>> finite_diff(f)
    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])

    Without arguments the above defaults to:

    >>> finite_diff(f, axis=0, dx=1.0, method='forward', pad_mode=None,
    ... edge_order=None)
    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])

    >>> finite_diff(f, dx=0.5)
    array([ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.])
    >>> finite_diff(f, pad_mode='constant')
    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., -9.])

    Central differences and different edge orders:

    >>> finite_diff(1/2*f**2, method='central')
    array([-0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
    >>> finite_diff(1/2*f**2, method='central', edge_order=1)
    array([ 0.5,  1. ,  2. ,  3. ,  4. ,  5. ,  6. ,  7. ,  8. ,  8.5])

    In-place evaluation:

    >>> out = f.copy()
    >>> out is finite_diff(f, out=out)
    True
    """
    f_arr = np.asarray(f)
    ndim = f_arr.ndim

    if f_arr.shape[axis] < 2:
        raise ValueError('in axis {}: at least two elements required, got {}'
                         ''.format(axis, f_arr.shape[axis]))

    if axis < 0:
        axis += ndim
    if not (0 <= axis < ndim):
        raise IndexError('`axis` {} outside the valid range 0 ... {}'
                         ''.format(axis, ndim - 1))

    dx, dx_in = float(dx), dx
    if dx <= 0 or not np.isfinite(dx):
        raise ValueError("`dx` must be positive, got {}".format(dx_in))

    method, method_in = str(method).lower(), method
    if method not in _SUPPORTED_DIFF_METHODS:
        raise ValueError('`method` {} was not understood'.format(method_in))

    pad_mode = kwargs.pop('pad_mode', None)
    if pad_mode not in _SUPPORTED_PADDING_METHODS:
        raise ValueError('`pad_mode` {} not understood'
                         ''.format(pad_mode))
    if pad_mode == 'constant':
        pad_const = float(kwargs.pop('pad_const', 0))

    edge_order = kwargs.pop('edge_order', None)
    if edge_order is None:
        if method == 'central':
            edge_order = 2
        else:
            edge_order = 1
    elif edge_order not in (1, 2):
            raise ValueError('`edge_order` must be 1 or 2, got {}'
                             ''.format(edge_order))

    if out is None:
        out = np.empty_like(f_arr)
    else:
        if out.shape != f.shape:
            raise ValueError('expected output shape {}, got {}'
                             ''.format(f.shape, out.shape))

    if f_arr.shape[axis] == 2 and edge_order == 2:
        raise ValueError('shape of array to small to use edge order 2')

    if pad_mode is not None:
        if method == 'central' and edge_order == 1:
            raise ValueError('central differences with padding cannot be used '
                             'with first-order accurate edges')
        if method in ('forward', 'backward') and edge_order == 2:
            raise ValueError('{} differences with padding only use edge '
                             'order 1'.format(method))

    # create slice objects: initially all are [:, :, ..., :]

    # current slice
    slice_out = [slice(None)] * ndim

    # slices used to calculate finite differences
    slice_node1 = [slice(None)] * ndim
    slice_node2 = [slice(None)] * ndim
    slice_node3 = [slice(None)] * ndim

    # Interior of the domain of f

    if method == 'central':
        # 2nd order differences in the interior of the domain of f
        slice_out[axis] = slice(1, -1)
        slice_node1[axis] = slice(2, None)
        slice_node2[axis] = slice(None, -2)
        # 1D equivalent: out[1:-1] = (f[2:] - f[:-2])/2.0
        np.subtract(f_arr[slice_node1], f_arr[slice_node2], out[slice_out])
        out[slice_out] /= 2.0

    elif method == 'forward':
        # 1st order differences in the interior of the domain of f
        slice_out[axis] = slice(1, -1)
        slice_node1[axis] = slice(2, None)
        slice_node2[axis] = slice(1, -1)
        # 1D equivalent: out[1:-1] = (f[2:] - f[1:-1])
        np.subtract(f_arr[slice_node1], f_arr[slice_node2], out[slice_out])

    elif method == 'backward':
        # 1st order differences in the interior of the domain of f
        slice_out[axis] = slice(1, -1)
        slice_node1[axis] = slice(1, -1)
        slice_node2[axis] = slice(None, -2)
        # 1D equivalent: out[1:-1] = (f[1:-1] - f[:-2])
        np.subtract(f_arr[slice_node1], f_arr[slice_node2], out[slice_out])

    # Boundaries

    if pad_mode == 'constant':
        # Assume constant value c for indices outside the domain of ``f``

        # With padding the method used on endpoints is the same as in the
        # interior of the domain of f

        if method == 'central':
            # 2nd-order lower edge
            slice_out[axis] = 0
            slice_node1[axis] = 1
            # 1D equivalent: out[0] = (f[1] - c)/2.0
            out[slice_out] = (f_arr[slice_node1] - pad_const) / 2.0

            # 2nd-order upper edge
            slice_out[axis] = -1
            slice_node2[axis] = -2
            # 1D equivalent: out[-1] = (c - f[-2])/2.0
            out[slice_out] = (pad_const - f_arr[slice_node2]) / 2.0

        elif method == 'forward':
            # 1st-oder lower edge
            slice_out[axis] = 0
            slice_node1[axis] = 1
            slice_node2[axis] = 0
            # 1D equivalent: out[0] = f[1] - f[0]
            out[slice_out] = f_arr[slice_node1] - f_arr[slice_node2]

            # 1st-oder upper edge
            slice_out[axis] = -1
            slice_node2[axis] = -1
            # 1D equivalent: out[-1] = c - f[-1]
            out[slice_out] = pad_const - f_arr[slice_node2]

        elif method == 'backward':
            # 1st-oder lower edge
            slice_out[axis] = 0
            slice_node1[axis] = 0
            # 1D equivalent: out[0] = f[0] - c
            out[slice_out] = f_arr[slice_node1] - pad_const

            # 1st-oder upper edge
            slice_out[axis] = -1
            slice_node1[axis] = -1
            slice_node2[axis] = -2
            # 1D equivalent: out[-1] = f[-1] - f[-2]
            out[slice_out] = f_arr[slice_node1] - f_arr[slice_node2]

    elif pad_mode == 'symmetric':
        # Values of f for indices outside the domain of f are replicates of
        # the edge values

        # With padding the method used on endpoints is the same as in the
        # interior of the domain of f

        if method == 'central':
            # 2nd-order lower edge
            slice_out[axis] = 0
            slice_node1[axis] = 1
            slice_node2[axis] = 0
            # 1D equivalent: out[0] = (f[1] - f[0])/2.0
            out[slice_out] = (f_arr[slice_node1] - f_arr[slice_node2]) / 2.0

            # 2nd-order upper edge
            slice_out[axis] = -1
            slice_node1[axis] = -1
            slice_node2[axis] = -2
            # 1D equivalent: out[-1] = (f[-1] - f[-2])/2.0
            out[slice_out] = (f_arr[slice_node1] - f_arr[slice_node2]) / 2.0

        elif method == 'forward':
            # 1st-oder lower edge
            slice_out[axis] = 0
            slice_node1[axis] = 1
            slice_node2[axis] = 0
            # 1D equivalent: out[0] = f[1] - f[0]
            out[slice_out] = f_arr[slice_node1] - f_arr[slice_node2]

            # 1st-oder upper edge
            slice_out[axis] = -1
            # 1D equivalent: out[-1] = f[-1] - f[-1] = 0
            out[slice_out] = 0

        elif method == 'backward':
            # 1st-oder lower edge
            slice_out[axis] = 0
            # 1D equivalent: out[0] = f[0] - f[0] = 0
            out[slice_out] = 0

            # 1st-oder upper edge
            slice_out[axis] = -1
            slice_node1[axis] = -1
            slice_node2[axis] = -2
            # 1D equivalent: out[-1] = f[-1] - f[-2]
            out[slice_out] = f_arr[slice_node1] - f_arr[slice_node2]

    elif pad_mode == 'symmetric_adjoint':
        # The adjoint case of symmetric

        if method == 'central':
            # 2nd-order lower edge
            slice_out[axis] = 0
            slice_node1[axis] = 1
            slice_node2[axis] = 0
            # 1D equivalent: out[0] = (f[1] - f[0])/2.0
            out[slice_out] = (f_arr[slice_node1] + f_arr[slice_node2]) / 2.0

            # 2nd-order upper edge
            slice_out[axis] = -1
            slice_node1[axis] = -1
            slice_node2[axis] = -2
            # 1D equivalent: out[-1] = (f[-1] - f[-2])/2.0
            out[slice_out] = (-f_arr[slice_node1] - f_arr[slice_node2]) / 2.0

        elif method == 'forward':
            # 1st-oder lower edge
            slice_out[axis] = 0
            slice_node1[axis] = 1
            # 1D equivalent: out[0] = - f[1]
            out[slice_out] = f_arr[slice_node1]

            # 1st-oder upper edge
            slice_out[axis] = -1
            slice_node1[axis] = -1
            # 1D equivalent: out[-1] = - f[-1]
            out[slice_out] = -f_arr[slice_node1]

        elif method == 'backward':
            # 1st-oder lower edge
            slice_out[axis] = 0
            slice_node1[axis] = 0
            # 1D equivalent: out[0] = - f[0]
            out[slice_out] = f_arr[slice_node1]

            # 1st-oder upper edge
            slice_out[axis] = -1
            slice_node1[axis] = -2
            # 1D equivalent: out[-1] = f[-2]
            out[slice_out] = -f_arr[slice_node1]

    elif pad_mode == 'periodic':
        # Values of f for indices outside the domain of f are replicates of
        # the edge values on the other side

        if method == 'central':
            # 2nd-order lower edge
            slice_out[axis] = 0
            slice_node1[axis] = 1
            slice_node2[axis] = -1
            # 1D equivalent: out[0] = (f[1] - f[-1])/2.0
            out[slice_out] = (f_arr[slice_node1] - f_arr[slice_node2]) / 2.0

            # 2nd-order upper edge
            slice_out[axis] = -1
            slice_node1[axis] = 0
            slice_node2[axis] = -2
            # 1D equivalent: out[-1] = (f[0] - f[-2])/2.0
            out[slice_out] = (f_arr[slice_node1] - f_arr[slice_node2]) / 2.0

        elif method == 'forward':
            # 1st-order lower edge
            slice_out[axis] = 0
            slice_node1[axis] = 1
            slice_node2[axis] = 0
            # 1D equivalent: out[0] = f[1] - f[0]
            out[slice_out] = f_arr[slice_node1] - f_arr[slice_node2]

            # 1st-order upper edge
            slice_out[axis] = -1
            slice_node1[axis] = 0
            slice_node2[axis] = -1
            # 1D equivalent: out[-1] = f[0] - f[-1]
            out[slice_out] = f_arr[slice_node1] - f_arr[slice_node2]

        elif method == 'backward':
            # 1st-order lower edge
            slice_out[axis] = 0
            slice_node1[axis] = 0
            slice_node2[axis] = -1
            # 1D equivalent: out[0] = f[0] - f[-1]
            out[slice_out] = f_arr[slice_node1] - f_arr[slice_node2]

            # 1st-order upper edge
            slice_out[axis] = -1
            slice_node1[axis] = -1
            slice_node2[axis] = -2
            # 1D equivalent: out[-1] = f[-1] - f[-2]
            out[slice_out] = f_arr[slice_node1] - f_arr[slice_node2]

    # Use one-sided differences on the endpoints
    else:

        # Edge-order accuracy is triggered implicitly by the method used or
        # explicitly using ``edge_order``

        # 1st order edges
        if edge_order == 1:
            # lower boundary
            slice_out[axis] = 0
            slice_node1[axis] = 1
            slice_node2[axis] = 0
            # 1D equivalent: out[0] = (f[1] - f[0])
            out[slice_out] = f_arr[slice_node1] - f_arr[slice_node2]

            # upper boundary
            slice_out[axis] = -1
            slice_node1[axis] = -1
            slice_node2[axis] = -2
            # 1D equivalent: out[-1] = (f[-1] - f[-2])
            out[slice_out] = f_arr[slice_node1] - f_arr[slice_node2]

        # 2nd order edges
        elif edge_order == 2:
            # lower boundary
            slice_out[axis] = 0
            slice_node1[axis] = 0
            slice_node2[axis] = 1
            slice_node3[axis] = 2
            # 1D equivalent: out[0] = -(3*f[0] - 4*f[1] + f[2]) / 2.0
            out[slice_out] = -(3.0 * f_arr[slice_node1] - 4.0 * f_arr[
                slice_node2] + f_arr[slice_node3]) / 2.0

            # upper boundary
            slice_out[axis] = -1
            slice_node1[axis] = -1
            slice_node2[axis] = -2
            slice_node3[axis] = -3
            # 1D equivalent: out[-1] = (3*f[-1] - 4*f[-2] + f[-3]) / 2.0
            out[slice_out] = (3.0 * f_arr[slice_node1] - 4.0 * f_arr[
                slice_node2] + f_arr[slice_node3]) / 2.0

    # divide by step size
    out /= dx

    return out


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
