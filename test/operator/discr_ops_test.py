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
# import odl
from odl.discr.lp_discr import uniform_discr, FunctionSpace, IntervalProd
from odl.operator.discr_ops import (finite_diff, DiscretePartDeriv,
                                    DiscreteGradient, DiscreteDivergence)
from odl.discr.lp_discr import DiscreteLp
from odl.space.ntuples import Rn
from odl.set.pspace import ProductSpace
# from odl.util.testutils import all_almost_equal


def test_finite_diff():
    """Finite differences test. """

    ndim = 10
    f = np.arange(ndim, dtype=float)

    df = finite_diff(f)
    assert all(df == np.ones(ndim))

    dx = 0.5
    df = finite_diff(f, dx=dx)
    assert all(df == np.ones(ndim) / dx)

    # central differences with zero padding at boundaries
    df = finite_diff(f, zero_padding=True)
    df0 = np.ones(ndim)
    df0[0] = f[1] / 2.
    df0[-1] = -f[-2] / 2
    assert all(df == df0)

    # one-sided difference at boundaries without zero padding
    df = finite_diff(f, zero_padding=False, edge_order=1)
    df0 = np.ones(ndim)
    df0[0] = f[1] - f[0]
    df0[-1] = f[-1] - f[-2]
    assert all(df == df0)

    # edge order
    df1 = finite_diff(np.sin(f / 10 * np.pi), edge_order=1)
    df2 = finite_diff(np.sin(f / 10 * np.pi), edge_order=2)
    assert all(df1[1:-1] == df2[1:-1])
    assert df1[0] is not df2[0]
    assert df1[-1] is not df2[-1]
    with pytest.raises(ValueError):
        finite_diff(f, edge_order=3)

    # axis parameter
    ndim = 5
    f = np.arange(ndim, dtype=float)
    f = f * f.reshape((ndim, 1))
    df = finite_diff(f, axis=0)
    f0 = np.array([[-0., 1., 2., 3., 4.],
                   [0., 1., 2., 3., 4.],
                   [0., 1., 2., 3., 4.],
                   [0., 1., 2., 3., 4.],
                   [0., 1., 2., 3., 4.]])
    assert (df == f0).all()
    df = finite_diff(f, axis=1)
    assert (df == f0.T).all()
    with pytest.raises(IndexError):
        finite_diff(f, axis=2)

    # zero padding uses second-order accurate edges
    with pytest.raises(ValueError):
        finite_diff(f, zero_padding=True, edge_order=1)

    # at lease two elements are required
    with pytest.raises(ValueError):
        finite_diff(np.array([0.0]))

    # in-place evaluation
    ndim = 10
    f = np.arange(ndim, dtype=float)
    out = np.zeros_like(f)
    assert out is not finite_diff(f)
    assert out is finite_diff(f, out)
    assert all(out == finite_diff(f))
    assert out is not finite_diff(f)
    out = np.zeros(2)
    with pytest.raises(TypeError):
        finite_diff(f, out)


def test_discr_part_deriv():
    pass


def ndvolume(vol_size, ndim, dtype=None):
    s = [1]
    vol = np.arange(vol_size, dtype=dtype)
    for _ in range(ndim - 1):
        s.insert(0, vol_size)
        vol = vol * vol.reshape(s)
    return vol


def test_discrete_gradient():
    """Discretized spatial gradient operator"""

    discr_space = Rn(10)
    with pytest.raises(TypeError):
        DiscreteGradient(discr_space)

    for ndim in range(1, 6):
        # DiscreteLp Vector
        vsize = 3
        intvl = IntervalProd([0.] * ndim, [vsize] * ndim)
        space = FunctionSpace(intvl)
        discr_space = uniform_discr(space, [vsize] * ndim)
        dom_vec = discr_space.element(ndvolume(vsize, ndim, np.float32))

        # Gradient
        grad = DiscreteGradient(discr_space)
        grad(dom_vec)

    # DiscreteLp Vector
    ndim = 2
    vsize = 3
    intvl = IntervalProd([0.] * ndim, [vsize] * ndim)
    space = FunctionSpace(intvl)
    discr_space = uniform_discr(space, [vsize] * ndim)
    dom_vec = discr_space.element(ndvolume(vsize, ndim, np.float32))

    # Gradient
    grad = DiscreteGradient(discr_space)
    grad_vec = grad(dom_vec)
    assert len(grad_vec) is ndim

    # Adjoint operator
    ran_vec = grad.range.element(
        [ndvolume(vsize, ndim, np.float32) ** 2] * ndim)
    adj = grad.adjoint
    adj_vec = adj(ran_vec)
    lhs = ran_vec.inner(grad_vec)
    rhs = dom_vec.inner(adj_vec)
    assert not lhs == 0
    assert not rhs == 0
    assert lhs == rhs


def test_discrete_divergence():
    """Discretized spatial divergence operator"""

    discr_space = Rn(10)
    with pytest.raises(TypeError):
        DiscreteDivergence(discr_space)

    for ndim in range(2, 6):
        # DiscreteLp Vector
        vsize = 3
        intvl = IntervalProd([0.] * ndim, [vsize] * ndim)
        space = FunctionSpace(intvl)
        discr_space = uniform_discr(space, [vsize] * ndim)

        # Divergence
        div = DiscreteDivergence(discr_space)
        dom_vec = div.domain.element(
            [ndvolume(vsize, ndim, np.float32)] * ndim)
        div(dom_vec)

    # DiscreteLp Vector
    ndim = 2
    vsize = 3
    intvl = IntervalProd([0.] * ndim, [vsize] * ndim)
    space = FunctionSpace(intvl)
    discr_space = uniform_discr(space, [vsize] * ndim)
    # dom = ProductSpace(discr_space, ndim)
    # dom_vec0 = dom.element([ndvolume(vsize, ndim, np.float32)] * ndim)

    # Divergence
    div = DiscreteDivergence(discr_space)
    dom_vec = div.domain.element([ndvolume(vsize, ndim, np.float32)] * ndim)
    div_vec = div(dom_vec)

    # Adjoint operator
    ran_vec = div.range.element(ndvolume(vsize, ndim, np.float32)**2)
    adj = div.adjoint
    adj_vec = adj(ran_vec)
    lhs = ran_vec.inner(div_vec)
    rhs = dom_vec.inner(adj_vec)
    assert not lhs == 0
    assert not rhs == 0
    assert lhs == rhs


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
