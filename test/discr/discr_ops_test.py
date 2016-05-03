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

"""Tests for operators defined on `DiscreteLp`."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import pytest

import odl
from odl.discr.discr_ops import (finite_diff, PartialDerivative,
                                 Gradient, Divergence)
from odl.util.testutils import almost_equal, all_equal, skip_if_no_cuda


# Phantom data
DATA_1D = np.array([0.5, 1, 3.5, 2, -.5, 3, -1, -1, 0, 3])
DATA_2D = np.array([[0., 1., 2., 3., 4.],
                    [1., 2., 3., 4., 5.],
                    [2., 3., 4., 5., 6.]]) ** 1


def test_finite_diff_invalid_args():
    """Test finite difference function for invalid arguments."""

    # edge order in {1,2}
    with pytest.raises(ValueError):
        finite_diff(DATA_1D, edge_order=0)
    with pytest.raises(ValueError):
        finite_diff(DATA_1D, edge_order=3)

    # central differences and zero padding use second-order accurate edges
    with pytest.raises(ValueError):
        finite_diff(DATA_1D, method='central', padding_method=0, edge_order=1)
    with pytest.raises(ValueError):
        finite_diff(DATA_1D, method='forward', padding_method=0, edge_order=2)

    # at least a two-element array is required
    with pytest.raises(ValueError):
        finite_diff(np.array([0.0]))

    # axis
    with pytest.raises(IndexError):
        finite_diff(DATA_1D, axis=2)

    # in-place argument
    out = np.zeros(DATA_1D.size + 1)
    with pytest.raises(ValueError):
        finite_diff(DATA_1D, out=out)
    with pytest.raises(ValueError):
        finite_diff(DATA_1D, dx=0)

    # wrong method
    with pytest.raises(ValueError):
        finite_diff(DATA_1D, method='non-method')


def test_finite_diff_explicit():
    """Compare finite differences function to explicit computation."""

    # phantom data
    arr = DATA_1D

    # explicitly calculated finite difference
    diff_ex = np.zeros_like(arr)

    # interior: second-order accurate differences
    diff_ex[1:-1] = (arr[2:] - arr[:-2]) / 2.0

    # default: out=None, axis=0, dx=1.0, zero_padding=None, method='forward'
    diff = finite_diff(arr, out=None, axis=0, dx=1.0, padding_method=None)
    assert all_equal(diff, finite_diff(arr))

    # boundary: one-sided second-order accurate forward/backward difference
    diff = finite_diff(arr, out=None, axis=0, dx=1.0, method='central',
                       padding_method=None)
    diff_ex[0] = -(3 * arr[0] - 4 * arr[1] + arr[2]) / 2.0
    diff_ex[-1] = (3 * arr[-1] - 4 * arr[-2] + arr[-3]) / 2.0
    assert all_equal(diff, diff_ex)

    # non-unit step length
    dx = 0.5
    diff = finite_diff(arr, dx=dx, method='central')
    assert all_equal(diff, diff_ex / dx)

    # boundary: second-order accurate central differences with zero padding
    diff = finite_diff(arr, method='central', padding_method='constant',
                       padding_value=0)
    diff_ex[0] = arr[1] / 2.0
    diff_ex[-1] = -arr[-2] / 2.0
    assert all_equal(diff, diff_ex)

    # boundary: one-sided first-order forward/backward difference without zero
    # padding
    diff = finite_diff(arr, method='central', edge_order=1)
    diff_ex[0] = arr[1] - arr[0]  # 1st-order accurate forward difference
    diff_ex[-1] = arr[-1] - arr[-2]  # 1st-order accurate backward diff.
    assert all_equal(diff, diff_ex)

    # different edge order really differ
    df1 = finite_diff(arr, method='central', edge_order=1)
    df2 = finite_diff(arr, method='central', edge_order=2)
    assert all_equal(df1[1:-1], diff_ex[1:-1])
    assert all_equal(df2[1:-1], diff_ex[1:-1])
    assert df1[0] != df2[0]
    assert df1[-1] != df2[-1]

    # in-place evaluation
    out = np.zeros_like(arr)
    assert out is finite_diff(arr, out=out)
    assert all_equal(out, finite_diff(arr))
    assert out is not finite_diff(arr)

    # axis
    arr = np.array([[0., 1., 2., 3., 4.],
                    [1., 2., 3., 4., 5.]])
    df0 = finite_diff(arr, axis=0)
    df1 = finite_diff(arr, axis=1)
    assert all_equal(df0, df1)

    # complex arrays
    arr = np.array([0., 1., 2., 3., 4.]) + 1j * np.array([10., 9., 8., 7.,
                                                          6.])
    diff = finite_diff(arr)
    assert all(diff.real == 1)
    assert all(diff.imag == -1)


def test_finite_diff_symmetric_padding():
    """Finite difference using replicate padding."""

    # Using replicate padding forward and backward differences have zero
    # derivative at the upper or lower endpoint, respectively
    assert finite_diff(DATA_1D, method='forward',
                       padding_method='symmetric')[-1] == 0
    assert finite_diff(DATA_1D, method='backward',
                       padding_method='symmetric')[0] == 0

    diff = finite_diff(DATA_1D, method='central', padding_method='symmetric')
    assert diff[0] == (DATA_1D[1] - DATA_1D[0]) / 2
    assert diff[-1] == (DATA_1D[-1] - DATA_1D[-2]) / 2


def test_forward_diff():
    """Forward finite differences."""

    arr = np.array([0., 3., 5., 6.])

    findiff_op = finite_diff(arr, padding_method='constant', method='forward')
    assert all_equal(findiff_op, [3., 2., 1., -6.])


def test_backward_diff():
    """Backward finite differences."""
    arr = np.array([0., 3., 5., 6.])

    findiff_op = finite_diff(arr, padding_method='constant', method='backward')
    assert all_equal(findiff_op, [0., 3., 2., 1.])


def test_part_deriv_cpu():
    """Discretized partial derivative."""

    with pytest.raises(TypeError):
        PartialDerivative(odl.Rn(1))

    # discretized space
    space = odl.uniform_discr([0, 0], [2, 1], DATA_2D.shape)

    # operator
    partial_0 = PartialDerivative(space, axis=0, method='central',
                                  padding_method='constant')
    partial_1 = PartialDerivative(space, axis=1, method='central',
                                  padding_method='constant')

    # discretized space vector
    vec = partial_0.domain.element(DATA_2D)

    # partial derivative
    partial_vec_0 = partial_0(vec)
    partial_vec_1 = partial_1(vec)

    # explicit calculation of finite difference

    # axis: 0
    diff_0 = np.zeros_like(DATA_2D)
    # interior: second-order accurate differences
    diff_0[1:-1, :] = (DATA_2D[2:, :] - DATA_2D[:-2, :]) / 2.0
    # boundary: second-order accurate central differences with zero padding
    diff_0[0, :] = DATA_2D[1, :] / 2.0
    diff_0[-1, :] = -DATA_2D[-2, :] / 2.0
    diff_0 /= space.cell_sides[0]

    # axis: 1
    diff_1 = np.zeros_like(DATA_2D)
    # interior: second-order accurate differences
    diff_1[:, 1:-1] = (DATA_2D[:, 2:] - DATA_2D[:, :-2]) / 2.0
    # boundary: second-order accurate central differences with zero padding
    diff_1[:, 0] = DATA_2D[:, 1] / 2.0
    diff_1[:, -1] = -DATA_2D[:, -2] / 2.0
    diff_1 /= space.cell_sides[1]

    # assert `dfe0` and `dfe1` do differ
    assert (diff_0 != diff_1).any()

    assert partial_vec_0 != partial_vec_1
    assert all_equal(partial_vec_0.asarray(), diff_0)
    assert all_equal(partial_vec_1.asarray(), diff_1)

    # adjoint not implemented
    with pytest.raises(NotImplementedError):
        PartialDerivative(space).adjoint


@skip_if_no_cuda
def test_discr_deriv_cuda():
    """Discretized partial derivative using CUDA."""

    # explicit calculation of finite difference
    partial_vec_explicit = np.zeros_like(DATA_1D)
    # interior: second-order accurate differences
    partial_vec_explicit[1:-1] = (DATA_1D[2:] - DATA_1D[:-2]) / 2.0
    # boundary: second-order accurate central differences with zero padding
    partial_vec_explicit[0] = DATA_1D[1] / 2.0
    partial_vec_explicit[-1] = -DATA_1D[-2] / 2.0

    # discretized space using CUDA
    discr_space = odl.uniform_discr(0, DATA_1D.size, DATA_1D.shape,
                                    impl='cuda')

    # operator
    partial = PartialDerivative(discr_space, method='central',
                                padding_method='constant')

    # discretized space vector
    vec = partial.domain.element(DATA_1D)

    # apply operator
    partial_vec = partial(vec)

    assert all_equal(partial_vec, partial_vec_explicit)


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


def test_gradient_cpu():
    """Discretized spatial gradient operator."""

    with pytest.raises(TypeError):
        Gradient(odl.Rn(1))

    # DiscreteLp Vector
    discr_space = odl.uniform_discr([0, 0], [6, 2.5], DATA_2D.shape)
    dom_vec = discr_space.element(DATA_2D)

    # computation of gradient components with helper function
    dx0, dx1 = discr_space.cell_sides
    diff_0 = finite_diff(DATA_2D, axis=0, dx=dx0, method='forward',
                         padding_method='constant')
    diff_1 = finite_diff(DATA_2D, axis=1, dx=dx1, method='forward',
                         padding_method='constant')

    # gradient
    grad = Gradient(discr_space)
    grad_vec = grad(dom_vec)
    assert len(grad_vec) == DATA_2D.ndim
    assert all_equal(grad_vec[0].asarray(), diff_0)
    assert all_equal(grad_vec[1].asarray(), diff_1)

    # Test adjoint operator

    ran_vec = grad.range.element([DATA_2D, DATA_2D ** 2])
    adj_vec = grad.adjoint(ran_vec)
    lhs = ran_vec.inner(grad_vec)
    rhs = dom_vec.inner(adj_vec)
    # Check not to use trivial data
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
def test_gradient_cuda():
    """Discretized spatial gradient operator using CUDA."""

    # DiscreteLp Vector
    discr_space = odl.uniform_discr([0, 0], [6, 2.5], DATA_2D.shape,
                                    impl='cuda')
    dom_vec = discr_space.element(DATA_2D)

    # computation of gradient components with helper function
    dx0, dx1 = discr_space.cell_sides
    diff_0 = finite_diff(DATA_2D, axis=0, dx=dx0, padding_method='constant')
    diff_1 = finite_diff(DATA_2D, axis=1, dx=dx1, padding_method='constant')

    # gradient
    grad = Gradient(discr_space)
    grad_vec = grad(dom_vec)
    assert len(grad_vec) == DATA_2D.ndim
    assert all_equal(grad_vec[0].asarray(), diff_0)
    assert all_equal(grad_vec[1].asarray(), diff_1)

    # adjoint operator
    ran_vec = grad.range.element([DATA_2D, DATA_2D ** 2])
    adj_vec = grad.adjoint(ran_vec)
    lhs = ran_vec.inner(grad_vec)
    rhs = dom_vec.inner(adj_vec)
    assert lhs != 0
    assert rhs != 0
    assert lhs == rhs


def test_divergence_cpu():
    """Discretized spatial divergence operator."""

    # Invalid space
    with pytest.raises(TypeError):
        Divergence(odl.Rn(1))

    # DiscreteLp
    # space = odl.uniform_discr([0, 0], [6, 2.5], DATA.shape)
    space = odl.uniform_discr([0, 0], [3, 5], DATA_2D.shape)

    # Operator instance
    div = Divergence(space, method='forward')

    # Apply operator
    # dom_vec = div.domain.element([DATA / 2, DATA ** 3])
    dom_vec = div.domain.element([DATA_2D, DATA_2D])
    div_dom_vec = div(dom_vec)

    # computation of divergence with helper function
    dx0, dx1 = space.cell_sides
    diff_0 = finite_diff(dom_vec[0].asarray(), axis=0, dx=dx0,
                         padding_method='constant')
    diff_1 = finite_diff(dom_vec[1].asarray(), axis=1, dx=dx1,
                         padding_method='constant')

    assert all_equal(diff_0 + diff_1, div_dom_vec.asarray())

    # Adjoint operator
    adj_div = div.adjoint
    ran_vec = div.range.element(DATA_2D ** 2)
    adj_div_ran_vec = adj_div(ran_vec)

    # Adjoint condition
    lhs = ran_vec.inner(div_dom_vec)
    rhs = dom_vec.inner(adj_div_ran_vec)
    # Check not to use trivial data
    assert lhs != 0
    assert rhs != 0
    assert almost_equal(lhs, rhs)

    # Higher dimensional arrays
    for ndim in range(1, 6):
        # DiscreteLp Vector
        lin_size = 3
        space = odl.uniform_discr([0.] * ndim, [lin_size] * ndim,
                                  [lin_size] * ndim)

        # Divergence
        div = Divergence(space)
        dom_vec = div.domain.element([ndvolume(lin_size, ndim)] * ndim)
        div(dom_vec)


@skip_if_no_cuda
def test_discrete_divergence_cuda():
    """Discretized spatial divergence operator using CUDA."""

    # Check result of operator with explicit summation

    # DiscreteLp
    space = odl.uniform_discr([0, 0], [1.5, 10], DATA_2D.shape, impl='cuda')

    # operator instance
    div = Divergence(space)

    # apply operator
    dom_vec = div.domain.element([DATA_2D, DATA_2D])
    div_dom_vec = div(dom_vec)

    # computation of divergence with helper function
    dx0, dx1 = space.cell_sides
    diff_0 = finite_diff(dom_vec[0].asarray(), axis=0, dx=dx0,
                         padding_method='constant')
    diff_1 = finite_diff(dom_vec[1].asarray(), axis=1, dx=dx1,
                         padding_method='constant')

    assert all_equal(diff_0 + diff_1, div_dom_vec.asarray())

    # Adjoint operator
    adj_div = div.adjoint
    ran_vec = div.range.element(DATA_2D ** 2)
    adj_div_ran_vec = adj_div(ran_vec)

    # Adjoint condition
    lhs = ran_vec.inner(div_dom_vec)
    rhs = dom_vec.inner(adj_div_ran_vec)
    assert lhs != 0
    assert rhs != 0
    assert almost_equal(lhs, rhs)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
