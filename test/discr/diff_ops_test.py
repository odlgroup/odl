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

"""Unit tests for `diff_ops`."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import pytest
import numpy as np

import odl
from odl.discr.diff_ops import (
    finite_diff, PartialDerivative, Gradient, Divergence, Laplacian)
from odl.util.testutils import (
    all_equal, all_almost_equal, almost_equal, never_skip)


methods = ['central', 'forward', 'backward']
method_ids = [' method={} '.format(p) for p in methods]


@pytest.fixture(scope="module", params=methods, ids=method_ids)
def method(request):
    return request.param


paddings = [('constant', 0), ('constant', 1), 'symmetric', 'periodic']
padding_ids = [' constant=0 ', ' constant=1 ', ' symmetric ', ' periodic ']


@pytest.fixture(scope="module", params=paddings, ids=padding_ids)
def padding(request):
    return request.param


# Test data
DATA_1D = np.array([0.5, 1, 3.5, 2, -.5, 3, -1, -1, 0, 3])
DATA_2D = np.array([[0., 1., 2., 3., 4.],
                    [1., 2., 3., 4., 5.],
                    [2., 3., 4., 5., 6.]])


# --- finite_diff --- #


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


def test_finite_diff_constant_padding():
    """Finite difference using constant padding."""

    for padding_value in [-1, 0, 1]:
        diff_forward = finite_diff(DATA_1D, method='forward',
                                   padding_method='constant',
                                   padding_value=padding_value)

        assert diff_forward[0] == DATA_1D[1] - DATA_1D[0]
        assert diff_forward[-1] == padding_value - DATA_1D[-1]

        diff_backward = finite_diff(DATA_1D, method='backward',
                                    padding_method='constant',
                                    padding_value=padding_value)

        assert diff_backward[0] == DATA_1D[0] - padding_value
        assert diff_backward[-1] == DATA_1D[-1] - DATA_1D[-2]

        diff_central = finite_diff(DATA_1D, method='central',
                                   padding_method='constant',
                                   padding_value=padding_value)

        assert diff_central[0] == (DATA_1D[1] - padding_value) / 2
        assert diff_central[-1] == (padding_value - DATA_1D[-2]) / 2


def test_finite_diff_periodic_padding():
    """Finite difference using periodic padding."""

    diff_forward = finite_diff(DATA_1D, method='forward',
                               padding_method='periodic')

    assert diff_forward[0] == DATA_1D[1] - DATA_1D[0]
    assert diff_forward[-1] == DATA_1D[0] - DATA_1D[-1]

    diff_backward = finite_diff(DATA_1D, method='backward',
                                padding_method='periodic')

    assert diff_backward[0] == DATA_1D[0] - DATA_1D[-1]
    assert diff_backward[-1] == DATA_1D[-1] - DATA_1D[-2]

    diff_central = finite_diff(DATA_1D, method='central',
                               padding_method='periodic')

    assert diff_central[0] == (DATA_1D[1] - DATA_1D[-1]) / 2
    assert diff_central[-1] == (DATA_1D[0] - DATA_1D[-2]) / 2


# --- PartialDerivative --- #


def test_part_deriv(fn_impl, method, padding):
    """Discretized partial derivative."""

    with pytest.raises(TypeError):
        PartialDerivative(odl.rn(1))

    if isinstance(padding, tuple):
        padding_method, padding_value = padding
    else:
        padding_method, padding_value = padding, None

    # discretized space
    space = odl.uniform_discr([0, 0], [2, 1], DATA_2D.shape, impl=fn_impl)
    dom_vec = space.element(DATA_2D)

    # operator
    for axis in range(space.ndim):
        partial = PartialDerivative(space, axis=axis, method=method,
                                    padding_method=padding_method,
                                    padding_value=padding_value)

        # Compare to helper function
        dx = space.cell_sides[axis]
        diff = finite_diff(DATA_2D, axis=axis, dx=dx, method=method,
                           padding_method=padding_method,
                           padding_value=padding_value)

        partial_vec = partial(dom_vec)
        assert all_almost_equal(partial_vec.asarray(), diff)

        # Test adjoint operator
        derivative = partial.derivative()
        ran_vec = derivative.range.element(DATA_2D ** 2)
        deriv_vec = derivative(dom_vec)
        adj_vec = derivative.adjoint(ran_vec)
        lhs = ran_vec.inner(deriv_vec)
        rhs = dom_vec.inner(adj_vec)

        # Check not to use trivial data
        assert lhs != 0
        assert rhs != 0
        assert almost_equal(lhs, rhs)


# --- Gradient --- #


def test_gradient(method, fn_impl, padding):
    """Discretized spatial gradient operator."""

    with pytest.raises(TypeError):
        Gradient(odl.rn(1), method=method)

    if isinstance(padding, tuple):
        padding_method, padding_value = padding
    else:
        padding_method, padding_value = padding, None

    # DiscreteLp Vector
    space = odl.uniform_discr([0, 0], [1, 1], DATA_2D.shape, impl=fn_impl)
    dom_vec = space.element(DATA_2D)

    # computation of gradient components with helper function
    dx0, dx1 = space.cell_sides
    diff_0 = finite_diff(DATA_2D, axis=0, dx=dx0, method=method,
                         padding_method=padding_method,
                         padding_value=padding_value)
    diff_1 = finite_diff(DATA_2D, axis=1, dx=dx1, method=method,
                         padding_method=padding_method,
                         padding_value=padding_value)

    # gradient
    grad = Gradient(space, method=method,
                    padding_method=padding_method,
                    padding_value=padding_value)
    grad_vec = grad(dom_vec)
    assert len(grad_vec) == DATA_2D.ndim
    assert all_almost_equal(grad_vec[0].asarray(), diff_0)
    assert all_almost_equal(grad_vec[1].asarray(), diff_1)

    # Test adjoint operator
    derivative = grad.derivative()
    ran_vec = derivative.range.element([DATA_2D, DATA_2D ** 2])
    deriv_grad_vec = derivative(dom_vec)
    adj_grad_vec = derivative.adjoint(ran_vec)
    lhs = ran_vec.inner(deriv_grad_vec)
    rhs = dom_vec.inner(adj_grad_vec)
    # Check not to use trivial data
    assert lhs != 0
    assert rhs != 0
    assert almost_equal(lhs, rhs)

    # higher dimensional arrays
    lin_size = 3
    for ndim in [1, 3, 6]:

        # DiscreteLp Vector
        space = odl.uniform_discr([0.] * ndim, [1.] * ndim, [lin_size] * ndim)
        dom_vec = odl.phantom.cuboid(space, [0.2] * ndim, [0.8] * ndim)

        # gradient
        grad = Gradient(space, method=method,
                        padding_method=padding_method,
                        padding_value=padding_value)
        grad(dom_vec)


# --- Divergence --- #


def test_divergence(method, fn_impl, padding):
    """Discretized spatial divergence operator."""

    # Invalid space
    with pytest.raises(TypeError):
        Divergence(range=odl.rn(1), method=method)

    if isinstance(padding, tuple):
        padding_method, padding_value = padding
    else:
        padding_method, padding_value = padding, None

    # DiscreteLp
    space = odl.uniform_discr([0, 0], [1, 1], DATA_2D.shape, impl=fn_impl)

    # Operator instance
    div = Divergence(range=space, method=method,
                     padding_method=padding_method,
                     padding_value=padding_value)

    # Apply operator
    dom_vec = div.domain.element([DATA_2D, DATA_2D])
    div_dom_vec = div(dom_vec)

    # computation of divergence with helper function
    dx0, dx1 = space.cell_sides
    diff_0 = finite_diff(dom_vec[0].asarray(), axis=0, dx=dx0, method=method,
                         padding_method=padding_method,
                         padding_value=padding_value)
    diff_1 = finite_diff(dom_vec[1].asarray(), axis=1, dx=dx1, method=method,
                         padding_method=padding_method,
                         padding_value=padding_value)

    assert all_almost_equal(diff_0 + diff_1, div_dom_vec.asarray())

    # Adjoint operator
    derivative = div.derivative()
    deriv_div_dom_vec = derivative(dom_vec)
    ran_vec = div.range.element(DATA_2D ** 2)
    adj_div_ran_vec = derivative.adjoint(ran_vec)

    # Adjoint condition
    lhs = ran_vec.inner(deriv_div_dom_vec)
    rhs = dom_vec.inner(adj_div_ran_vec)
    # Check not to use trivial data
    assert lhs != 0
    assert rhs != 0
    assert almost_equal(lhs, rhs)

    # Higher dimensional arrays
    for ndim in range(1, 6):
        # DiscreteLp Vector
        lin_size = 3
        space = odl.uniform_discr([0.] * ndim, [1.] * ndim, [lin_size] * ndim)

        # Divergence
        div = Divergence(range=space, method=method,
                         padding_method=padding_method,
                         padding_value=padding_value)
        dom_vec = odl.phantom.cuboid(space, [0.2] * ndim, [0.8] * ndim)
        div([dom_vec] * ndim)


def test_laplacian(fn_impl, padding):
    """Discretized spatial laplacian operator."""

    # Invalid space
    with pytest.raises(TypeError):
        Divergence(range=odl.rn(1))

    if isinstance(padding, tuple):
        padding_method, padding_value = padding
    else:
        padding_method, padding_value = padding, None

    # DiscreteLp
    space = odl.uniform_discr([0, 0], [1, 1], DATA_2D.shape, impl=fn_impl)

    # Operator instance
    lap = Laplacian(space,
                    padding_method=padding_method,
                    padding_value=padding_value)

    # Apply operator
    dom_vec = lap.domain.element(DATA_2D)
    div_dom_vec = lap(dom_vec)

    # computation of divergence with helper function
    dx0, dx1 = space.cell_sides

    expected_result = np.zeros(space.shape)
    for axis, dx in enumerate(space.cell_sides):
        diff_f = finite_diff(dom_vec.asarray(), axis=axis, dx=dx ** 2,
                             method='forward',
                             padding_method=padding_method,
                             padding_value=padding_value)
        diff_b = finite_diff(dom_vec.asarray(), axis=axis, dx=dx ** 2,
                             method='backward',
                             padding_method=padding_method,
                             padding_value=padding_value)
        expected_result += diff_f - diff_b

    assert all_almost_equal(expected_result, div_dom_vec.asarray())

    # Adjoint operator
    derivative = lap.derivative()
    deriv_lap_dom_vec = derivative(dom_vec)
    ran_vec = lap.range.element(DATA_2D ** 2)
    adj_lap_ran_vec = derivative.adjoint(ran_vec)

    # Adjoint condition
    lhs = ran_vec.inner(deriv_lap_dom_vec)
    rhs = dom_vec.inner(adj_lap_ran_vec)

    # Check not to use trivial data
    assert lhs != 0
    assert rhs != 0
    assert almost_equal(lhs, rhs)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
