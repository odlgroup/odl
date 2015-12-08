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
from odl.discr.lp_discr import uniform_discr, FunctionSpace, IntervalProd
from odl.discr.discr_ops import (finite_diff, DiscretePartDeriv,
                                 DiscreteGradient, DiscreteDivergence)
from odl.space.ntuples import Rn
from odl.set.domain import Rectangle
from odl.util.testutils import almost_equal, all_equal


def test_finite_diff():
    """Finite differences test."""

    # some array
    f = np.array([0.5, 1, 3.5, 2, -.5, 3, -1, -1, 0, 3])

    # invalid parameter values
    # edge order in {1,2}
    with pytest.raises(ValueError):
        finite_diff(f, edge_order=0)
    with pytest.raises(ValueError):
        finite_diff(f, edge_order=3)
    # zero padding uses second-order accurate edges
        with pytest.raises(ValueError):
            finite_diff(f, zero_padding=True, edge_order=1)
    # at least a two-element array is required
        with pytest.raises(ValueError):
            finite_diff(np.array([0.0]))
    # axis
    with pytest.raises(IndexError):
        finite_diff(f, axis=2)
    # in-place argument
    out = np.zeros(f.size + 1)
    with pytest.raises(ValueError):
        finite_diff(f, out)
    with pytest.raises(ValueError):
        finite_diff(f, dx=0)

    # finite difference array
    dfe = np.zeros_like(f)

    # interior: second-order accurate differences
    dfe[1:-1] = (f[2:] - f[:-2]) / 2.0

    # default: out=None, axis=0, dx=1.0, edge_order=2, zero_padding=False
    df = finite_diff(f, out=None, axis=0, dx=1.0, edge_order=2,
                     zero_padding=False)
    assert all_equal(df, finite_diff(f))

    # boundary: second-order accurate forward/backward difference
    dfe[0] = -(3 * f[0] - 4 * f[1] + f[2]) / 2.0
    dfe[-1] = (3 * f[-1] - 4 * f[-2] + f[-3]) / 2.0
    assert all_equal(df, dfe)

    # non-unit step length
    dx = 0.5
    df = finite_diff(f, dx=dx)
    assert all_equal(df, dfe / dx)

    # boundary: second-order accurate central differences with zero padding
    df = finite_diff(f, zero_padding=True)
    dfe[0] = f[1] / 2.0
    dfe[-1] = -f[-2] / 2.0
    assert all_equal(df, dfe)

    # boundary: one-sided first-order forward/backward difference without zero
    # padding
    df = finite_diff(f, zero_padding=False, edge_order=1)
    dfe[0] = f[1] - f[0]  # 1st-order accurate forward difference
    dfe[-1] = f[-1] - f[-2]  # 1st-order accurate backward difference
    assert all_equal(df, dfe)

    # different edge order really differ
    df1 = finite_diff(f, edge_order=1)
    df2 = finite_diff(f, edge_order=2)
    assert all_equal(df1[1:-1], dfe[1:-1])
    assert all_equal(df2[1:-1], dfe[1:-1])
    assert not df1[0] == df2[0]
    assert not df1[-1] == df2[-1]

    # in-place evaluation
    out = np.zeros_like(f)
    assert out is finite_diff(f, out)
    assert all_equal(out, finite_diff(f))
    assert out is not finite_diff(f)

    # axis
    f = np.array([[0., 1., 2., 3., 4.],
                  [1., 2., 3., 4., 5.]])
    df0 = finite_diff(f, axis=0)
    df1 = finite_diff(f, axis=1)
    assert all_equal(df0, df1)

    # complex arrays
    f = np.array([0., 1., 2., 3., 4.]) + 1j * np.array([10., 9., 8., 7., 6.])
    df = finite_diff(f)
    assert all(df.real == 1)
    assert all(df.imag == -1)


def test_discr_part_deriv():
    """Discretized partial derivative."""

    discr_space = Rn(1)
    with pytest.raises(TypeError):
        DiscretePartDeriv(discr_space)

    # phantom data
    data = np.array([[1.2, 0, 3, 5, 7],
                     [4, -1, 3, -2.1, -4]])

    # discretized space
    space = FunctionSpace(Rectangle([0, 0], [2, 1]))
    discr_space = uniform_discr(space, data.shape)

    # operator
    par_div = DiscretePartDeriv(discr_space)

    # discretized space vector
    f = par_div.domain.element(data)

    # partial derivative
    par_div_f1 = par_div(f)

    # operator
    par_div = DiscretePartDeriv(discr_space, axis=1, dx=0.2, edge_order=2,
                                zero_padding=True)

    # partial derivative
    par_div_f2 = par_div(f)

    assert not par_div_f1 == par_div_f2


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
    s = [1]
    vol = np.arange(lin_size, dtype=dtype)
    for _ in range(ndim - 1):
        s.insert(0, lin_size)
        vol = vol * vol.reshape(s)
    return vol


def test_discrete_gradient():
    """Discretized spatial gradient operator."""

    discr_space = Rn(1)
    with pytest.raises(TypeError):
        DiscreteGradient(discr_space)

    for ndim in range(1, 6):
        # DiscreteLp Vector
        vsize = 3
        intvl = IntervalProd([0.] * ndim, [vsize] * ndim)
        space = FunctionSpace(intvl)
        discr_space = uniform_discr(space, [vsize] * ndim)
        dom_vec = discr_space.element(ndvolume(vsize, ndim))

        # Gradient
        grad = DiscreteGradient(discr_space)
        grad(dom_vec)

    # DiscreteLp Vector
    ndim = 2
    vsize = 3
    intvl = IntervalProd([0.] * ndim, [vsize] * ndim)
    space = FunctionSpace(intvl)
    discr_space = uniform_discr(space, [vsize] * ndim)
    dom_vec = discr_space.element(ndvolume(vsize, ndim))

    # Gradient
    grad = DiscreteGradient(discr_space)
    grad_vec = grad(dom_vec)
    assert len(grad_vec) is ndim

    # Adjoint operator
    ran_vec = grad.range.element(
        [ndvolume(vsize, ndim) ** 2] * ndim)
    adj = grad.adjoint
    adj_vec = adj(ran_vec)
    lhs = ran_vec.inner(grad_vec)
    rhs = dom_vec.inner(adj_vec)
    assert not lhs == 0
    assert not rhs == 0
    assert lhs == rhs


def test_discrete_divergence():
    """Discretized spatial divergence operator."""

    # Invalid arguments
    discr_space = Rn(1)
    with pytest.raises(TypeError):
        DiscreteDivergence(discr_space)

    # Check result of operator with explicit summation
    data = np.array([[0., 1., 2., 3., 4.],
                     [1., 2., 3., 4., 5.],
                     [2., 3., 4., 5., 6.]])

    # DiscreteLp
    space = FunctionSpace(Rectangle([0, 0], [6, 2.5]))
    discr_space = uniform_discr(space, data.shape)

    # Operator instance
    div = DiscreteDivergence(discr_space)

    # Apply operator
    dom_vec = div.domain.element([data, data])
    div_dom_vec = div(dom_vec)

    # computation of divergence with helper function
    dx0, dx1 = discr_space.grid.stride
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
    assert not lhs == 0
    assert not rhs == 0
    assert almost_equal(lhs, rhs)

    # Higher dimensional arrays
    for ndim in range(1, 6):

        # DiscreteLp Vector
        lin_size = 3
        space = FunctionSpace(IntervalProd([0.] * ndim, [lin_size] * ndim))
        discr_space = uniform_discr(space, [lin_size] * ndim)

        # Divergence
        div = DiscreteDivergence(discr_space)
        dom_vec = div.domain.element([ndvolume(lin_size, ndim)] * ndim)
        div(dom_vec)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
