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


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library

standard_library.install_aliases()

# External module imports
import numpy as np
import pytest

# ODL imports
import odl
from odl.discr.discr_ops import (finite_diff, PartialDerivative,
                                 Gradient, Divergence)
from odl.util.testutils import almost_equal, all_equal, skip_if_no_cuda


def test_finite_diff():
    """Finite differences test."""

    # phantom data
    arr = np.array([0.5, 1, 3.5, 2, -.5, 3, -1, -1, 0, 3])

    # invalid parameter values
    # edge order in {1,2}
    with pytest.raises(ValueError):
        finite_diff(arr, edge_order=0)
    with pytest.raises(ValueError):
        finite_diff(arr, edge_order=3)
    # zero padding uses second-order accurate edges
    with pytest.raises(ValueError):
        finite_diff(arr, zero_padding=True, edge_order=1)
    # at least a two-element array is required
    with pytest.raises(ValueError):
        finite_diff(np.array([0.0]))
    # axis
    with pytest.raises(IndexError):
        finite_diff(arr, axis=2)
    # in-place argument
    out = np.zeros(arr.size + 1)
    with pytest.raises(ValueError):
        finite_diff(arr, out)
    with pytest.raises(ValueError):
        finite_diff(arr, dx=0)
    # wrong method
    with pytest.raises(ValueError):
        finite_diff(arr, method='non-method')

    # explicitly calculated finite difference
    findiff_ex = np.zeros_like(arr)

    # interior: second-order accurate differences
    findiff_ex[1:-1] = (arr[2:] - arr[:-2]) / 2.0

    # default: out=None, axis=0, dx=1.0, edge_order=2, zero_padding=False
    findiff_op = finite_diff(arr, out=None, axis=0, dx=1.0, edge_order=2,
                             zero_padding=False)
    assert all_equal(findiff_op, finite_diff(arr))

    # boundary: second-order accurate forward/backward difference
    findiff_ex[0] = -(3 * arr[0] - 4 * arr[1] + arr[2]) / 2.0
    findiff_ex[-1] = (3 * arr[-1] - 4 * arr[-2] + arr[-3]) / 2.0
    assert all_equal(findiff_op, findiff_ex)

    # non-unit step length
    dx = 0.5
    findiff_op = finite_diff(arr, dx=dx)
    assert all_equal(findiff_op, findiff_ex / dx)

    # boundary: second-order accurate central differences with zero padding
    findiff_op = finite_diff(arr, zero_padding=True)
    findiff_ex[0] = arr[1] / 2.0
    findiff_ex[-1] = -arr[-2] / 2.0
    assert all_equal(findiff_op, findiff_ex)

    # boundary: one-sided first-order forward/backward difference without zero
    # padding
    findiff_op = finite_diff(arr, zero_padding=False, edge_order=1)
    findiff_ex[0] = arr[1] - arr[0]  # 1st-order accurate forward difference
    findiff_ex[-1] = arr[-1] - arr[-2]  # 1st-order accurate backward diff.
    assert all_equal(findiff_op, findiff_ex)

    # different edge order really differ
    df1 = finite_diff(arr, edge_order=1)
    df2 = finite_diff(arr, edge_order=2)
    assert all_equal(df1[1:-1], findiff_ex[1:-1])
    assert all_equal(df2[1:-1], findiff_ex[1:-1])
    assert df1[0] != df2[0]
    assert df1[-1] != df2[-1]

    # in-place evaluation
    out = np.zeros_like(arr)
    assert out is finite_diff(arr, out)
    assert all_equal(out, finite_diff(arr))
    assert out is not finite_diff(arr)

    # axis
    arr = np.array([[0., 1., 2., 3., 4.],
                    [1., 2., 3., 4., 5.]])
    df0 = finite_diff(arr, axis=0)
    df1 = finite_diff(arr, axis=1)
    assert all_equal(df0, df1)

    # complex arrays
    arr = np.array([0., 1., 2., 3., 4.]) + 1j * np.array([10., 9., 8., 7., 6.])
    findiff_op = finite_diff(arr)
    assert all(findiff_op.real == 1)
    assert all(findiff_op.imag == -1)


def test_forward_diff():
    arr = np.array([0., 3., 5., 6.])

    findiff_op = finite_diff(arr, zero_padding=True, method='forward')
    assert all_equal(findiff_op, [3., 2., 1., -6.])


def test_backward_diff():
    arr = np.array([0., 3., 5., 6.])

    findiff_op = finite_diff(arr, zero_padding=True, method='backward')
    assert all_equal(findiff_op, [0., 3., 2., 1.])


def test_discr_part_deriv():
    """Discretized partial derivative."""

    discr_space = odl.Rn(1)
    with pytest.raises(TypeError):
        PartialDerivative(discr_space)

    # phantom data
    data = np.array([[0., 1., 2., 3., 4.],
                     [1., 2., 3., 4., 5.],
                     [2., 3., 4., 5., 6.]])

    # explicit calculation of finite difference
    # axis: 0
    dfe0 = np.zeros_like(data)
    # interior: second-order accurate differences
    dfe0[1:-1, :] = (data[2:, :] - data[:-2, :]) / 2.0
    # boundary: second-order accurate central differences with zero padding
    dfe0[0, :] = data[1, :] / 2.0
    dfe0[-1, :] = -data[-2, :] / 2.0
    # axis: 1
    dfe1 = np.zeros_like(data)
    # interior: second-order accurate differences
    dfe1[:, 1:-1] = (data[:, 2:] - data[:, :-2]) / 2.0
    # boundary: second-order accurate central differences with zero padding
    dfe1[:, 0] = data[:, 1] / 2.0
    dfe1[:, -1] = -data[:, -2] / 2.0

    # assert `dfe0` and `dfe1` do differ
    assert (dfe0 != dfe1).any()

    # discretized space
    discr_space = odl.uniform_discr([0, 0], [2, 1], data.shape)

    # operator
    partial_0 = PartialDerivative(discr_space, axis=0, zero_padding=True)
    partial_1 = PartialDerivative(discr_space, axis=1, zero_padding=True)

    # discretized space vector
    vec = partial_0.domain.element(data)

    # partial derivative
    partial_vec_0 = partial_0(vec)
    partial_vec_1 = partial_1(vec)

    assert partial_vec_0 != partial_vec_1
    assert all_equal(partial_vec_0.asarray(), dfe0)
    assert all_equal(partial_vec_1.asarray(), dfe1)

    # operator
    partial_0 = PartialDerivative(discr_space, axis=1, dx=0.2, edge_order=2,
                                  zero_padding=True)

    # adjoint not implemented
    with pytest.raises(NotImplementedError):
        partial_0.adjoint


@skip_if_no_cuda
def test_discr_part_deriv_cuda():
    """Discretized partial derivative using CUDA."""

    # phantom data
    data = np.array([0., 1., 2., 3., 4., 16., 25., 36.])

    # explicit calculation of finite difference
    dfe = np.zeros_like(data)
    # interior: second-order accurate differences
    dfe[1:-1] = (data[2:] - data[:-2]) / 2.0
    # boundary: second-order accurate central differences with zero padding
    dfe[0] = data[1] / 2.0
    dfe[-1] = -data[-2] / 2.0

    # discretized space using CUDA
    discr_space = odl.uniform_discr(0, data.size, data.shape, impl='cuda')

    # operator
    partial = PartialDerivative(discr_space, zero_padding=True)

    # discretized space vector
    discr_vec = partial.domain.element(data)

    # apply operator
    partial_vec = partial(discr_vec)

    assert all_equal(partial_vec, dfe)


def ndvolume(lin_size, ndim, dtype=np.float64):
    """Hypercube phantom.

    Parameters
    ----------
    lin_size : `int`
       Size of array in each dimension
    ndim : `int`
        Number of dimensions
    dtype : dtype
        The type of the output array

    """
    vec = [1]
    vol = np.arange(lin_size, dtype=dtype)
    for _ in range(ndim - 1):
        vec.insert(0, lin_size)
        vol = vol * vol.reshape(vec)
    return vol


def test_discrete_gradient():
    """Discretized spatial gradient operator."""

    discr_space = odl.Rn(1)
    with pytest.raises(TypeError):
        Gradient(discr_space)

    # Check result of operator with explicit summation
    # phantom data
    data = np.array([[0., 1., 2., 3., 4.],
                     [1., 2., 3., 4., 5.],
                     [2., 3., 4., 5., 6.]])

    data = np.array([[0., 1., 2., 3., 4.],
                     [0., 1., 2., 3., 4.],
                     [0., 1., 2., 3., 4.]])

    # DiscreteLp Vector
    discr_space = odl.uniform_discr([0, 0], [6, 2.5], data.shape)
    dom_vec = discr_space.element(data)

    # computation of gradient components with helper function
    dx0, dx1 = discr_space.cell_sides
    df0 = finite_diff(data, axis=0, dx=dx0, zero_padding=True, edge_order=2)
    df1 = finite_diff(data, axis=1, dx=dx1, zero_padding=True, edge_order=2)

    # gradient
    grad = Gradient(discr_space)
    grad_vec = grad(dom_vec)
    assert len(grad_vec) == data.ndim
    assert all_equal(grad_vec[0].asarray(), df0)
    assert all_equal(grad_vec[1].asarray(), df1)

    # adjoint operator
    ran_vec = grad.range.element([data, data ** 2])
    adj_vec = grad.adjoint(ran_vec)
    lhs = ran_vec.inner(grad_vec)
    rhs = dom_vec.inner(adj_vec)
    assert lhs != 0
    assert rhs != 0
    assert lhs == rhs

    # higher dimensional arrays
    lin_size = 3
    for ndim in range(1, 6):

        # DiscreteLp Vector
        discr_space = odl.uniform_discr([0.] * ndim, [lin_size] * ndim,
                                        [lin_size] * ndim)
        dom_vec = discr_space.element(ndvolume(lin_size, ndim))

        # gradient
        grad = Gradient(discr_space)
        grad(dom_vec)


@skip_if_no_cuda
def test_discrete_gradient_cuda():
    """Discretized spatial gradient operator using CUDA."""

    # Check result of operator with explicit summation
    # phantom data
    data = np.array([[0., 1., 2., 3., 4.],
                     [1., 2., 3., 4., 5.],
                     [2., 3., 4., 5., 6.]])

    # DiscreteLp Vector
    discr_space = odl.uniform_discr([0, 0], [6, 2.5], data.shape, impl='cuda')
    dom_vec = discr_space.element(data)

    # computation of gradient components with helper function
    dx0, dx1 = discr_space.cell_sides
    df0 = finite_diff(data, axis=0, dx=dx0, zero_padding=True, edge_order=2)
    df1 = finite_diff(data, axis=1, dx=dx1, zero_padding=True, edge_order=2)

    # gradient
    grad = Gradient(discr_space)
    grad_vec = grad(dom_vec)
    assert len(grad_vec) == data.ndim
    assert all_equal(grad_vec[0].asarray(), df0)
    assert all_equal(grad_vec[1].asarray(), df1)

    # adjoint operator
    ran_vec = grad.range.element([data, data ** 2])
    adj_vec = grad.adjoint(ran_vec)
    lhs = ran_vec.inner(grad_vec)
    rhs = dom_vec.inner(adj_vec)
    assert lhs != 0
    assert rhs != 0
    assert lhs == rhs


def test_discrete_divergence():
    """Discretized spatial divergence operator."""

    # Invalid arguments
    discr_space = odl.Rn(1)
    with pytest.raises(TypeError):
        Divergence(discr_space)

    # Check result of operator with explicit summation
    data = np.array([[0., 1., 2., 3., 4.],
                     [1., 2., 3., 4., 5.],
                     [2., 3., 4., 5., 6.]])

    # DiscreteLp
    discr_space = odl.uniform_discr([0, 0], [6, 2.5], data.shape)

    # Operator instance
    div = Divergence(discr_space)

    # Apply operator
    dom_vec = div.domain.element([data, data])
    div_dom_vec = div(dom_vec)

    # computation of divergence with helper function
    dx0, dx1 = discr_space.cell_sides
    df0 = finite_diff(data, axis=0, dx=dx0, zero_padding=True, edge_order=2)
    df1 = finite_diff(data, axis=1, dx=dx1, zero_padding=True, edge_order=2)

    assert all_equal(df0 + df1, div_dom_vec.asarray())

    # Adjoint operator
    adj_div = div.adjoint
    ran_vec = div.range.element(data ** 2)
    adj_div_ran_vec = adj_div(ran_vec)

    # Adjoint condition
    lhs = ran_vec.inner(div_dom_vec)
    rhs = dom_vec.inner(adj_div_ran_vec)
    assert lhs != 0
    assert rhs != 0
    assert almost_equal(lhs, rhs)

    # Higher dimensional arrays
    for ndim in range(1, 6):
        # DiscreteLp Vector
        lin_size = 3
        discr_space = odl.uniform_discr([0.] * ndim, [lin_size] * ndim,
                                        [lin_size] * ndim)
        # Divergence
        div = Divergence(discr_space)
        dom_vec = div.domain.element([ndvolume(lin_size, ndim)] * ndim)
        div(dom_vec)


@skip_if_no_cuda
def test_discrete_divergence_cuda():
    """Discretized spatial divergence operator using CUDA."""

    # Check result of operator with explicit summation
    # phantom data
    data = np.array([[0., 1., 2., 3., 4.],
                     [1., 2., 3., 4., 5.],
                     [2., 3., 4., 5., 6.]])

    # DiscreteLp
    discr_space = odl.uniform_discr([0, 0], [1.5, 10], data.shape, impl='cuda')

    # operator instance
    div = Divergence(discr_space)

    # apply operator
    dom_vec = div.domain.element([data, data])
    div_dom_vec = div(dom_vec)

    # computation of divergence with helper function
    dx0, dx1 = discr_space.cell_sides
    df0 = finite_diff(data, axis=0, dx=dx0, zero_padding=True, edge_order=2)
    df1 = finite_diff(data, axis=1, dx=dx1, zero_padding=True, edge_order=2)

    assert all_equal(df0 + df1, div_dom_vec.asarray())

    # Adjoint operator
    adj_div = div.adjoint
    ran_vec = div.range.element(data ** 2)
    adj_div_ran_vec = adj_div(ran_vec)

    # Adjoint condition
    lhs = ran_vec.inner(div_dom_vec)
    rhs = dom_vec.inner(adj_div_ran_vec)
    assert lhs != 0
    assert rhs != 0
    assert almost_equal(lhs, rhs)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
