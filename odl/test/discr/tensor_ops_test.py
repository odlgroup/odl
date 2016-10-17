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

"""Unit tests for `tensor_ops`."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import pytest
import numpy as np

import odl
from odl.discr.tensor_ops import PointwiseNorm, PointwiseInner
from odl.space.pspace import ProductSpace
from odl.util.testutils import all_almost_equal, all_equal


exp_params = [2.0, 1.0, float('inf'), 3.5, 1.5]
exp_ids = [' p = {} '.format(p) for p in exp_params]


@pytest.fixture(scope="module", ids=exp_ids, params=exp_params)
def exponent(request):
    return request.param


# ---- PointwiseNorm ----


def test_pointwise_norm_init_properties():
    # 1d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2))
    vfspace = ProductSpace(fspace, 1, exponent=1)

    # Make sure the code runs and test the properties
    pwnorm = PointwiseNorm(vfspace)
    assert pwnorm.base_space == fspace
    assert all_equal(pwnorm.weights, [1])
    assert not pwnorm.is_weighted
    assert pwnorm.exponent == 1.0
    repr(pwnorm)

    pwnorm = PointwiseNorm(vfspace, exponent=2)
    assert pwnorm.exponent == 2

    pwnorm = PointwiseNorm(vfspace, weight=2)
    assert all_equal(pwnorm.weights, [2])
    assert pwnorm.is_weighted

    # 3d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2))
    vfspace = ProductSpace(fspace, 3, exponent=1)

    # Make sure the code runs and test the properties
    pwnorm = PointwiseNorm(vfspace)
    assert pwnorm.base_space == fspace
    assert all_equal(pwnorm.weights, [1, 1, 1])
    assert not pwnorm.is_weighted
    assert pwnorm.exponent == 1.0
    repr(pwnorm)

    pwnorm = PointwiseNorm(vfspace, exponent=2)
    assert pwnorm.exponent == 2

    pwnorm = PointwiseNorm(vfspace, weight=[1, 2, 3])
    assert all_equal(pwnorm.weights, [1, 2, 3])
    assert pwnorm.is_weighted

    # Bad input
    with pytest.raises(TypeError):
        PointwiseNorm(odl.rn(3))  # No power space

    with pytest.raises(ValueError):
        PointwiseNorm(vfspace, exponent=0.5)  # < 1 not allowed

    with pytest.raises(ValueError):
        PointwiseNorm(vfspace, weight=-1)  # < 0 not allowed

    with pytest.raises(ValueError):
        PointwiseNorm(vfspace, weight=[1, 0, 1])  # 0 invalid


def test_pointwise_norm_real(exponent):
    # 1d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2))
    vfspace = ProductSpace(fspace, 1)
    pwnorm = PointwiseNorm(vfspace, exponent)

    testarr = np.array([[[1, 2],
                         [3, 4]]])

    true_norm = np.linalg.norm(testarr, ord=exponent, axis=0)

    func = vfspace.element(testarr)
    func_pwnorm = pwnorm(func)
    assert all_almost_equal(func_pwnorm, true_norm.reshape(-1))

    out = fspace.element()
    pwnorm(func, out=out)
    assert all_almost_equal(out, true_norm.reshape(-1))

    # 3d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2))
    vfspace = ProductSpace(fspace, 3)
    pwnorm = PointwiseNorm(vfspace, exponent)

    testarr = np.array([[[1, 2],
                         [3, 4]],
                        [[0, -1],
                         [0, 1]],
                        [[1, 1],
                         [1, 1]]])

    true_norm = np.linalg.norm(testarr, ord=exponent, axis=0)

    func = vfspace.element(testarr)
    func_pwnorm = pwnorm(func)
    assert all_almost_equal(func_pwnorm, true_norm.reshape(-1))

    out = fspace.element()
    pwnorm(func, out=out)
    assert all_almost_equal(out, true_norm.reshape(-1))


def test_pointwise_norm_complex(exponent):
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), dtype=complex)
    vfspace = ProductSpace(fspace, 3)
    pwnorm = PointwiseNorm(vfspace, exponent)

    testarr = np.array([[[1 + 1j, 2],
                         [3, 4 - 2j]],
                        [[0, -1],
                         [0, 1]],
                        [[1j, 1j],
                         [1j, 1j]]])

    true_norm = np.linalg.norm(testarr, ord=exponent, axis=0)

    func = vfspace.element(testarr)
    func_pwnorm = pwnorm(func)
    assert all_almost_equal(func_pwnorm, true_norm.reshape(-1))

    out = fspace.element()
    pwnorm(func, out=out)
    assert all_almost_equal(out, true_norm.reshape(-1))


def test_pointwise_norm_weighted(exponent):
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2))
    vfspace = ProductSpace(fspace, 3)
    weight = np.array([1.0, 2.0, 3.0])
    pwnorm = PointwiseNorm(vfspace, exponent, weight=weight)

    testarr = np.array([[[1, 2],
                         [3, 4]],
                        [[0, -1],
                         [0, 1]],
                        [[1, 1],
                         [1, 1]]])

    if exponent in (1.0, float('inf')):
        true_norm = np.linalg.norm(weight[:, None, None] * testarr,
                                   ord=exponent, axis=0)
    else:
        true_norm = np.linalg.norm(
            weight[:, None, None] ** (1 / exponent) * testarr, ord=exponent,
            axis=0)

    func = vfspace.element(testarr)
    func_pwnorm = pwnorm(func)
    assert all_almost_equal(func_pwnorm, true_norm.reshape(-1))

    out = fspace.element()
    pwnorm(func, out=out)
    assert all_almost_equal(out, true_norm.reshape(-1))


# ---- PointwiseInner ----


def test_pointwise_inner_init_properties():
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2))
    vfspace = ProductSpace(fspace, 3, exponent=2)

    # Make sure the code runs and test the properties
    pwinner = PointwiseInner(vfspace, vfspace.one())
    assert pwinner.base_space == fspace
    assert all_equal(pwinner.weights, [1, 1, 1])
    assert not pwinner.is_weighted
    repr(pwinner)

    pwinner = PointwiseInner(vfspace, vfspace.one(), weight=[1, 2, 3])
    assert all_equal(pwinner.weights, [1, 2, 3])
    assert pwinner.is_weighted

    # Bad input
    with pytest.raises(TypeError):
        PointwiseInner(odl.rn(3), odl.rn(3).one())  # No power space

    # TODO: Does not raise currently, although bad_vecfield not in vfspace!
    """
    bad_vecfield = ProductSpace(fspace, 3, exponent=1).one()
    with pytest.raises(TypeError):
        PointwiseInner(vfspace, bad_vecfield)
    """

    with pytest.raises(ValueError):
        PointwiseInner(vfspace, vfspace.one(), weight=-1)  # < 0 not allowed

    with pytest.raises(ValueError):
        PointwiseInner(vfspace, vfspace.one(), weight=[1, 0, 1])  # 0 invalid


def test_pointwise_inner_real():
    # 1d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2))
    vfspace = ProductSpace(fspace, 1)
    array = np.array([[[-1, -3],
                       [2, 0]]])
    pwinner = PointwiseInner(vfspace, vecfield=array)

    testarr = np.array([[[1, 2],
                         [3, 4]]])

    true_inner = np.sum(testarr * array, axis=0)

    func = vfspace.element(testarr)
    func_pwinner = pwinner(func)
    assert all_almost_equal(func_pwinner, true_inner.reshape(-1))

    out = fspace.element()
    pwinner(func, out=out)
    assert all_almost_equal(out, true_inner.reshape(-1))

    # 3d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2))
    vfspace = ProductSpace(fspace, 3)
    array = np.array([[[-1, -3],
                       [2, 0]],
                      [[0, 0],
                       [0, 1]],
                      [[-1, 1],
                       [1, 1]]])
    pwinner = PointwiseInner(vfspace, vecfield=array)

    testarr = np.array([[[1, 2],
                         [3, 4]],
                        [[0, -1],
                         [0, 1]],
                        [[1, 1],
                         [1, 1]]])

    true_inner = np.sum(testarr * array, axis=0)

    func = vfspace.element(testarr)
    func_pwinner = pwinner(func)
    assert all_almost_equal(func_pwinner, true_inner.reshape(-1))

    out = fspace.element()
    pwinner(func, out=out)
    assert all_almost_equal(out, true_inner.reshape(-1))


def test_pointwise_inner_complex():
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), dtype=complex)
    vfspace = ProductSpace(fspace, 3)
    array = np.array([[[-1 - 1j, -3],
                       [2, 2j]],
                      [[-1j, 0],
                       [0, 1]],
                      [[-1, 1 + 2j],
                       [1, 1]]])
    pwinner = PointwiseInner(vfspace, vecfield=array)

    testarr = np.array([[[1 + 1j, 2],
                         [3, 4 - 2j]],
                        [[0, -1],
                         [0, 1]],
                        [[1j, 1j],
                         [1j, 1j]]])

    true_inner = np.sum(testarr * array.conj(), axis=0)

    func = vfspace.element(testarr)
    func_pwinner = pwinner(func)
    assert all_almost_equal(func_pwinner, true_inner.reshape(-1))

    out = fspace.element()
    pwinner(func, out=out)
    assert all_almost_equal(out, true_inner.reshape(-1))


def test_pointwise_inner_weighted():
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2))
    vfspace = ProductSpace(fspace, 3)
    array = np.array([[[-1, -3],
                       [2, 0]],
                      [[0, 0],
                       [0, 1]],
                      [[-1, 1],
                       [1, 1]]])

    weight = np.array([1.0, 2.0, 3.0])
    pwinner = PointwiseInner(vfspace, vecfield=array, weight=weight)

    testarr = np.array([[[1, 2],
                         [3, 4]],
                        [[0, -1],
                         [0, 1]],
                        [[1, 1],
                         [1, 1]]])

    true_inner = np.sum(weight[:, None, None] * testarr * array, axis=0)

    func = vfspace.element(testarr)
    func_pwinner = pwinner(func)
    assert all_almost_equal(func_pwinner, true_inner.reshape(-1))

    out = fspace.element()
    pwinner(func, out=out)
    assert all_almost_equal(out, true_inner.reshape(-1))


def test_pointwise_inner_adjoint():
    # 1d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), dtype=complex)
    vfspace = ProductSpace(fspace, 1)
    array = np.array([[[-1, -3],
                       [2, 0]]])
    pwinner = PointwiseInner(vfspace, vecfield=array)

    testarr = np.array([[1 + 1j, 2],
                        [3, 4 - 2j]])

    true_inner_adj = testarr[None, :, :] * array

    testfunc = fspace.element(testarr)
    testfunc_pwinner_adj = pwinner.adjoint(testfunc)
    assert all_almost_equal(testfunc_pwinner_adj,
                            true_inner_adj.reshape([1, -1]))

    out = vfspace.element()
    pwinner.adjoint(testfunc, out=out)
    assert all_almost_equal(out, true_inner_adj.reshape([1, -1]))

    # 3d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), dtype=complex)
    vfspace = ProductSpace(fspace, 3)
    array = np.array([[[-1 - 1j, -3],
                       [2, 2j]],
                      [[-1j, 0],
                       [0, 1]],
                      [[-1, 1 + 2j],
                       [1, 1]]])
    pwinner = PointwiseInner(vfspace, vecfield=array)

    testarr = np.array([[1 + 1j, 2],
                        [3, 4 - 2j]])

    true_inner_adj = testarr[None, :, :] * array

    testfunc = fspace.element(testarr)
    testfunc_pwinner_adj = pwinner.adjoint(testfunc)
    assert all_almost_equal(testfunc_pwinner_adj,
                            true_inner_adj.reshape([3, -1]))

    out = vfspace.element()
    pwinner.adjoint(testfunc, out=out)
    assert all_almost_equal(out, true_inner_adj.reshape([3, -1]))


def test_pointwise_inner_adjoint_weighted():
    # Weighted product space only
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), dtype=complex)
    vfspace = ProductSpace(fspace, 3, weight=[2, 4, 6])
    array = np.array([[[-1 - 1j, -3],
                       [2, 2j]],
                      [[-1j, 0],
                       [0, 1]],
                      [[-1, 1 + 2j],
                       [1, 1]]])
    pwinner = PointwiseInner(vfspace, vecfield=array)

    testarr = np.array([[1 + 1j, 2],
                        [3, 4 - 2j]])

    true_inner_adj = testarr[None, :, :] * array  # same as unweighted case

    testfunc = fspace.element(testarr)
    testfunc_pwinner_adj = pwinner.adjoint(testfunc)
    assert all_almost_equal(testfunc_pwinner_adj,
                            true_inner_adj.reshape([3, -1]))

    out = vfspace.element()
    pwinner.adjoint(testfunc, out=out)
    assert all_almost_equal(out, true_inner_adj.reshape([3, -1]))

    # Using different weighting in the inner product
    pwinner = PointwiseInner(vfspace, vecfield=array, weight=[4, 8, 12])

    testarr = np.array([[1 + 1j, 2],
                        [3, 4 - 2j]])

    true_inner_adj = 2 * testarr[None, :, :] * array  # w / v = (2, 2, 2)

    testfunc = fspace.element(testarr)
    testfunc_pwinner_adj = pwinner.adjoint(testfunc)
    assert all_almost_equal(testfunc_pwinner_adj,
                            true_inner_adj.reshape([3, -1]))

    out = vfspace.element()
    pwinner.adjoint(testfunc, out=out)
    assert all_almost_equal(out, true_inner_adj.reshape([3, -1]))


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
