# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for `diff_ops`."""

from __future__ import division
import pytest
import numpy as np

import odl
from odl.discr.diff_ops import (
    finite_diff, PartialDerivative, Gradient, Divergence, Laplacian)
from odl.util.testutils import (
    all_equal, all_almost_equal, almost_equal, noise_element, simple_fixture)


# --- pytest fixtures --- #


method = simple_fixture('method', ['central', 'forward', 'backward'])
padding = simple_fixture('padding', [('constant', 0), ('constant', 1),
                                     'symmetric', 'periodic',
                                     'order0', 'order1', 'order2'])


@pytest.fixture(scope="module", params=[1, 2, 3], ids=['1d', '2d', '3d'])
def space(request, tspace_impl):
    ndim = request.param

    return odl.uniform_discr([0] * ndim, [1] * ndim, [5] * ndim,
                             impl=tspace_impl)


# Test data
DATA_1D = np.array([0.5, 1, 3.5, 2, -.5, 3, -1, -1, 0, 3])


# --- finite_diff --- #


def test_finite_diff_invalid_args():
    """Test finite difference function for invalid arguments."""

    # Test that old "edge order" argument fails.
    with pytest.raises(ValueError):
        finite_diff(DATA_1D, axis=0, edge_order=0)
    with pytest.raises(ValueError):
        finite_diff(DATA_1D, axis=0, edge_order=3)

    # at least a two-element array is required
    with pytest.raises(ValueError):
        finite_diff(np.array([0.0]), axis=0)

    # axis
    with pytest.raises(IndexError):
        finite_diff(DATA_1D, axis=2)

    # in-place argument
    out = np.zeros(DATA_1D.size + 1)
    with pytest.raises(ValueError):
        finite_diff(DATA_1D, axis=0, out=out)
    with pytest.raises(ValueError):
        finite_diff(DATA_1D, axis=0, dx=0)

    # wrong method
    with pytest.raises(ValueError):
        finite_diff(DATA_1D, axis=0, method='non-method')


def test_finite_diff_explicit():
    """Compare finite differences function to explicit computation."""

    # phantom data
    arr = DATA_1D

    # explicitly calculated finite difference
    diff_ex = np.zeros_like(arr)

    # interior: second-order accurate differences
    diff_ex[1:-1] = (arr[2:] - arr[:-2]) / 2.0

    # default: out=None, axis=0, dx=1.0, zero_padding=None, method='forward'
    diff = finite_diff(arr, axis=0, dx=1.0, out=None,
                       pad_mode='constant')
    assert all_equal(diff, finite_diff(arr, axis=0))

    # boundary: one-sided second-order accurate forward/backward difference
    diff = finite_diff(arr, axis=0, dx=1.0, out=None,
                       method='central', pad_mode='order2')
    diff_ex[0] = -(3 * arr[0] - 4 * arr[1] + arr[2]) / 2.0
    diff_ex[-1] = (3 * arr[-1] - 4 * arr[-2] + arr[-3]) / 2.0
    assert all_equal(diff, diff_ex)

    # non-unit step length
    dx = 0.5
    diff = finite_diff(arr, axis=0, dx=dx, method='central', out=None,
                       pad_mode='order2')
    assert all_equal(diff, diff_ex / dx)

    # boundary: second-order accurate central differences with zero padding
    diff = finite_diff(arr, axis=0, method='central', pad_mode='constant',
                       pad_const=0)
    diff_ex[0] = arr[1] / 2.0
    diff_ex[-1] = -arr[-2] / 2.0
    assert all_equal(diff, diff_ex)

    # boundary: one-sided first-order forward/backward difference without zero
    # padding
    diff = finite_diff(arr, axis=0, method='central', pad_mode='order1')
    diff_ex[0] = arr[1] - arr[0]  # 1st-order accurate forward difference
    diff_ex[-1] = arr[-1] - arr[-2]  # 1st-order accurate backward diff.
    assert all_equal(diff, diff_ex)

    # different edge order really differ
    df1 = finite_diff(arr, axis=0, method='central', pad_mode='order1')
    df2 = finite_diff(arr, axis=0, method='central', pad_mode='order2')
    assert all_equal(df1[1:-1], diff_ex[1:-1])
    assert all_equal(df2[1:-1], diff_ex[1:-1])
    assert df1[0] != df2[0]
    assert df1[-1] != df2[-1]

    # in-place evaluation
    out = np.zeros_like(arr)
    assert out is finite_diff(arr, axis=0, out=out)
    assert all_equal(out, finite_diff(arr, axis=0))
    assert out is not finite_diff(arr, axis=0)

    # axis
    arr = np.array([[0., 2., 4., 6., 8.],
                    [1., 3., 5., 7., 9.]])
    df0 = finite_diff(arr, axis=0, pad_mode='order1')
    darr0 = 1 * np.ones(arr.shape)
    assert all_equal(df0, darr0)
    darr1 = 2 * np.ones(arr.shape)
    df1 = finite_diff(arr, axis=1, pad_mode='order1')
    assert all_equal(df1, darr1)

    # complex arrays
    arr = np.array([0., 1., 2., 3., 4.]) + 1j * np.array([10., 9., 8., 7.,
                                                          6.])
    diff = finite_diff(arr, axis=0, pad_mode='order1')
    assert all(diff.real == 1)
    assert all(diff.imag == -1)


def test_finite_diff_symmetric_padding():
    """Finite difference using replicate padding."""

    # Using replicate padding forward and backward differences have zero
    # derivative at the upper or lower endpoint, respectively
    assert finite_diff(DATA_1D, axis=0, method='forward',
                       pad_mode='symmetric')[-1] == 0
    assert finite_diff(DATA_1D, axis=0, method='backward',
                       pad_mode='symmetric')[0] == 0

    diff = finite_diff(DATA_1D, axis=0, method='central', pad_mode='symmetric')
    assert diff[0] == (DATA_1D[1] - DATA_1D[0]) / 2
    assert diff[-1] == (DATA_1D[-1] - DATA_1D[-2]) / 2


def test_finite_diff_constant_padding():
    """Finite difference using constant padding."""

    for pad_const in [-1, 0, 1]:
        diff_forward = finite_diff(DATA_1D, axis=0, method='forward',
                                   pad_mode='constant',
                                   pad_const=pad_const)

        assert diff_forward[0] == DATA_1D[1] - DATA_1D[0]
        assert diff_forward[-1] == pad_const - DATA_1D[-1]

        diff_backward = finite_diff(DATA_1D, axis=0, method='backward',
                                    pad_mode='constant',
                                    pad_const=pad_const)

        assert diff_backward[0] == DATA_1D[0] - pad_const
        assert diff_backward[-1] == DATA_1D[-1] - DATA_1D[-2]

        diff_central = finite_diff(DATA_1D, axis=0, method='central',
                                   pad_mode='constant',
                                   pad_const=pad_const)

        assert diff_central[0] == (DATA_1D[1] - pad_const) / 2
        assert diff_central[-1] == (pad_const - DATA_1D[-2]) / 2


def test_finite_diff_periodic_padding():
    """Finite difference using periodic padding."""

    diff_forward = finite_diff(DATA_1D, axis=0, method='forward',
                               pad_mode='periodic')

    assert diff_forward[0] == DATA_1D[1] - DATA_1D[0]
    assert diff_forward[-1] == DATA_1D[0] - DATA_1D[-1]

    diff_backward = finite_diff(DATA_1D, axis=0, method='backward',
                                pad_mode='periodic')

    assert diff_backward[0] == DATA_1D[0] - DATA_1D[-1]
    assert diff_backward[-1] == DATA_1D[-1] - DATA_1D[-2]

    diff_central = finite_diff(DATA_1D, axis=0, method='central',
                               pad_mode='periodic')

    assert diff_central[0] == (DATA_1D[1] - DATA_1D[-1]) / 2
    assert diff_central[-1] == (DATA_1D[0] - DATA_1D[-2]) / 2


# --- PartialDerivative --- #


def test_part_deriv(space, method, padding):
    """Discretized partial derivative."""

    with pytest.raises(TypeError):
        PartialDerivative(odl.rn(1))

    if isinstance(padding, tuple):
        pad_mode, pad_const = padding
    else:
        pad_mode, pad_const = padding, 0

    # discretized space
    dom_vec = noise_element(space)
    dom_vec_arr = dom_vec.asarray()

    # operator
    for axis in range(space.ndim):
        partial = PartialDerivative(space, axis=axis, method=method,
                                    pad_mode=pad_mode,
                                    pad_const=pad_const)

        # Compare to helper function
        dx = space.cell_sides[axis]
        diff = finite_diff(dom_vec_arr, axis=axis, dx=dx, method=method,
                           pad_mode=pad_mode,
                           pad_const=pad_const)

        partial_vec = partial(dom_vec)
        assert all_almost_equal(partial_vec, diff)

        # Test adjoint operator
        derivative = partial.derivative()
        ran_vec = noise_element(space)
        deriv_vec = derivative(dom_vec)
        adj_vec = derivative.adjoint(ran_vec)
        lhs = ran_vec.inner(deriv_vec)
        rhs = dom_vec.inner(adj_vec)

        # Check not to use trivial data
        assert lhs != 0
        assert rhs != 0
        assert almost_equal(lhs, rhs, places=4)


# --- Gradient --- #


def test_gradient(space, method, padding):
    """Discretized spatial gradient operator."""

    places = 2 if space.dtype == np.float32 else 4

    with pytest.raises(TypeError):
        Gradient(odl.rn(1), method=method)

    if isinstance(padding, tuple):
        pad_mode, pad_const = padding
    else:
        pad_mode, pad_const = padding, 0

    # DiscreteLp Vector
    dom_vec = noise_element(space)
    dom_vec_arr = dom_vec.asarray()

    # gradient
    grad = Gradient(space, method=method,
                    pad_mode=pad_mode,
                    pad_const=pad_const)
    grad_vec = grad(dom_vec)
    assert len(grad_vec) == space.ndim

    # computation of gradient components with helper function
    for axis, dx in enumerate(space.cell_sides):
        diff = finite_diff(dom_vec_arr, axis=axis, dx=dx, method=method,
                           pad_mode=pad_mode,
                           pad_const=pad_const)

        assert all_almost_equal(grad_vec[axis].asarray(), diff)

    # Test adjoint operator
    derivative = grad.derivative()
    ran_vec = noise_element(derivative.range)
    deriv_grad_vec = derivative(dom_vec)
    adj_grad_vec = derivative.adjoint(ran_vec)
    lhs = ran_vec.inner(deriv_grad_vec)
    rhs = dom_vec.inner(adj_grad_vec)

    # Check not to use trivial data
    assert lhs != 0
    assert rhs != 0
    assert almost_equal(lhs, rhs, places=places)

    # higher dimensional arrays
    lin_size = 3
    for ndim in [1, 3, 6]:

        # DiscreteLpElement
        space = odl.uniform_discr([0.] * ndim, [1.] * ndim, [lin_size] * ndim)
        dom_vec = odl.phantom.cuboid(space, [0.2] * ndim, [0.8] * ndim)

        # gradient
        grad = Gradient(space, method=method,
                        pad_mode=pad_mode,
                        pad_const=pad_const)
        grad(dom_vec)

# --- Divergence --- #


def test_divergence(space, method, padding):
    """Discretized spatial divergence operator."""

    # Invalid space
    with pytest.raises(TypeError):
        Divergence(range=odl.rn(1), method=method)

    if isinstance(padding, tuple):
        pad_mode, pad_const = padding
    else:
        pad_mode, pad_const = padding, 0

    # Operator instance
    div = Divergence(range=space, method=method,
                     pad_mode=pad_mode,
                     pad_const=pad_const)

    # Apply operator
    dom_vec = noise_element(div.domain)
    div_dom_vec = div(dom_vec)

    # computation of divergence with helper function
    expected_result = np.zeros(space.shape)
    for axis, dx in enumerate(space.cell_sides):
        expected_result += finite_diff(dom_vec[axis], axis=axis, dx=dx,
                                       method=method, pad_mode=pad_mode,
                                       pad_const=pad_const)

    assert all_almost_equal(expected_result, div_dom_vec.asarray())

    # Adjoint operator
    derivative = div.derivative()
    deriv_div_dom_vec = derivative(dom_vec)
    ran_vec = noise_element(div.range)
    adj_div_ran_vec = derivative.adjoint(ran_vec)

    # Adjoint condition
    lhs = ran_vec.inner(deriv_div_dom_vec)
    rhs = dom_vec.inner(adj_div_ran_vec)
    # Check not to use trivial data
    assert lhs != 0
    assert rhs != 0
    assert almost_equal(lhs, rhs, places=4)

    # Higher dimensional arrays
    for ndim in range(1, 6):
        # DiscreteLpElement
        lin_size = 3
        space = odl.uniform_discr([0.] * ndim, [1.] * ndim, [lin_size] * ndim)


def test_laplacian(space, padding):
    """Discretized spatial laplacian operator."""

    # Invalid space
    with pytest.raises(TypeError):
        Laplacian(range=odl.rn(1))

    if isinstance(padding, tuple):
        pad_mode, pad_const = padding
    else:
        pad_mode, pad_const = padding, 0

    if pad_mode in ('order1', 'order2'):
        return  # these pad modes not supported for laplacian

    # Operator instance
    lap = Laplacian(space, pad_mode=pad_mode, pad_const=pad_const)

    # Apply operator
    dom_vec = noise_element(space)
    div_dom_vec = lap(dom_vec)

    # computation of divergence with helper function
    expected_result = np.zeros(space.shape)
    for axis, dx in enumerate(space.cell_sides):
        diff_f = finite_diff(dom_vec.asarray(), axis=axis, dx=dx ** 2,
                             method='forward', pad_mode=pad_mode,
                             pad_const=pad_const)
        diff_b = finite_diff(dom_vec.asarray(), axis=axis, dx=dx ** 2,
                             method='backward', pad_mode=pad_mode,
                             pad_const=pad_const)
        expected_result += diff_f - diff_b

    assert all_almost_equal(expected_result, div_dom_vec.asarray())

    # Adjoint operator
    derivative = lap.derivative()
    deriv_lap_dom_vec = derivative(dom_vec)
    ran_vec = noise_element(lap.range)
    adj_lap_ran_vec = derivative.adjoint(ran_vec)

    # Adjoint condition
    lhs = ran_vec.inner(deriv_lap_dom_vec)
    rhs = dom_vec.inner(adj_lap_ran_vec)

    # Check not to use trivial data
    assert lhs != 0
    assert rhs != 0
    assert almost_equal(lhs, rhs, places=4)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
