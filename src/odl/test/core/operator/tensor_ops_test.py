# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for `tensor_ops`."""

from __future__ import division

import numpy as np
import scipy.sparse

import odl
import pytest
from odl.core.operator.tensor_ops import (
    MatrixOperator, PointwiseInner, PointwiseNorm, PointwiseSum)
from odl.core.space.pspace import ProductSpace
from odl.core.util.testutils import (
    all_almost_equal, all_equal, noise_element, noise_elements, simple_fixture, skip_if_no_pytorch)
from odl.core.space.entry_points import tensor_space_impl_names
from odl.core.sparse import SparseMatrix
from odl.core.array_API_support import lookup_array_backend, get_array_and_backend
from odl.core.operator.tensor_ops import DeviceChange, ArrayBackendChange

matrix_dtype = simple_fixture(
    name='matrix_dtype',
    params=['float32', 'complex64', 'float64', 'complex128'])


@pytest.fixture(scope='module')
def matrix(matrix_dtype, odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    space = odl.rn((3, 4), impl=impl, device=device, dtype=matrix_dtype)
    return space.one()

    # else:
    #     assert 0

@pytest.fixture(scope='module')
def matrix_space(odl_impl_device_pairs, odl_floating_dtype):
    impl, device = odl_impl_device_pairs
    return odl.tensor_space(
        shape=(4, 4), 
        dtype=odl_floating_dtype, 
        impl=impl, 
        device=device
        )

def test_npy_fallback_operator(matrix_space):
    """ Test of the lambertw function which is know not to work for pytorch """
    impl, device = matrix_space.impl, matrix_space.device
    
    x = noise_element(matrix_space)

    to_cpu_op = DeviceChange(domain_device=device, range_device='cpu')

    to_npy_op = ArrayBackendChange(domain_impl=impl, range_impl='numpy')

    op = to_npy_op @ to_cpu_op @ MatrixOperator(x.data)

    assert op(x[0]).device == 'cpu'
    assert op(x[0]).impl   == 'numpy'

    to_orig_device = DeviceChange(domain_device='cpu', range_device=device)

    to_orig_impl = ArrayBackendChange(domain_impl='numpy', range_impl=impl)

    op_back = to_orig_impl @ to_orig_device @ op

    assert op_back(x[0]).device == device
    assert op_back(x[0]).impl   == impl


exponent = simple_fixture('exponent', [2.0, 1.0, float('inf'), 3.5, 1.5])

sparse_matrix_backend = simple_fixture('backend', ['scipy', 'pytorch'])
sparse_matrix_format = simple_fixture('format', ['COO'])
# ---- PointwiseNorm ----


def test_pointwise_norm_init_properties(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    # 1d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), impl=impl, device=device)
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

    pwnorm = PointwiseNorm(vfspace, weighting=2)
    assert all_equal(pwnorm.weights, [2])
    assert pwnorm.is_weighted

    # 3d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), impl=impl, device=device)
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

    pwnorm = PointwiseNorm(vfspace, weighting=[1, 2, 3])
    assert all_equal(pwnorm.weights, [1, 2, 3])
    assert pwnorm.is_weighted

    # Bad input
    with pytest.raises(TypeError):
        PointwiseNorm(odl.rn(3))  # No power space

    with pytest.raises(ValueError):
        PointwiseNorm(vfspace, exponent=0.5)  # < 1 not allowed

    with pytest.raises(ValueError):
        PointwiseNorm(vfspace, weighting=-1)  # < 0 not allowed

    with pytest.raises(ValueError):
        PointwiseNorm(vfspace, weighting=[1, 0, 1])  # 0 invalid


def test_pointwise_norm_real(exponent, odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    # 1d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), impl=impl, device=device)
    vfspace = ProductSpace(fspace, 1)
    pwnorm = PointwiseNorm(vfspace, exponent)

    testarr = fspace.array_backend.array_constructor([[[1, 2],
                         [3, 4]]], dtype=float, device=device)

    true_norm = fspace.array_namespace.linalg.norm(testarr, ord=exponent, axis=0)

    func = vfspace.element(testarr)
    func_pwnorm = pwnorm(func)
    assert all_almost_equal(func_pwnorm, true_norm)

    out = fspace.element()
    pwnorm(func, out=out)
    assert all_almost_equal(out, true_norm)

    # 3d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), impl=impl, device=device)
    vfspace = ProductSpace(fspace, 3)
    pwnorm = PointwiseNorm(vfspace, exponent)

    testarr = fspace.array_backend.array_constructor([[[1, 2],
                         [3, 4]],
                        [[0, -1],
                         [0, 1]],
                        [[1, 1],
                         [1, 1]]], dtype=float, device=device)

    true_norm = fspace.array_namespace.linalg.norm(testarr, ord=exponent, axis=0)

    func = vfspace.element(testarr)
    func_pwnorm = pwnorm(func)
    assert all_almost_equal(func_pwnorm, true_norm)

    out = fspace.element()
    pwnorm(func, out=out)
    assert all_almost_equal(out, true_norm)


def test_pointwise_norm_complex(exponent, odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), dtype=complex, impl=impl, device=device)
    vfspace = ProductSpace(fspace, 3)
    pwnorm = PointwiseNorm(vfspace, exponent)

    testarr = fspace.array_backend.array_constructor([[[1 + 1j, 2],
                         [3, 4 - 2j]],
                        [[0, -1],
                         [0, 1]],
                        [[1j, 1j],
                         [1j, 1j]]], device=device, dtype=complex)

    true_norm = fspace.array_namespace.linalg.norm(testarr, ord=exponent, axis=0)

    func = vfspace.element(testarr)
    func_pwnorm = pwnorm(func)
    assert all_almost_equal(func_pwnorm, true_norm)

    out = pwnorm.range.element()
    pwnorm(func, out=out)
    assert all_almost_equal(out.real, true_norm)


def test_pointwise_norm_weighted(exponent, odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), impl=impl, device=device)

    ns = fspace.array_namespace
    backend = fspace.array_backend

    vfspace = ProductSpace(fspace, 3)
    weight = backend.array_constructor([1.0, 2.0, 3.0], device=device)
    pwnorm = PointwiseNorm(vfspace, exponent, weighting=weight)

    testarr = backend.array_constructor([[[1, 2],
                         [3, 4]],
                        [[0, -1],
                         [0, 1]],
                        [[1, 1],
                         [1, 1]]], device=device, dtype=float)

    if exponent in (1.0, float('inf')):
        true_norm = ns.linalg.norm(weight[:, None, None] * testarr,
                                   ord=exponent, axis=0)
    else:
        true_norm = ns.linalg.norm(
            weight[:, None, None] ** (1 / exponent) * testarr, ord=exponent,
            axis=0)

    func = vfspace.element(testarr)
    func_pwnorm = pwnorm(func)
    assert all_almost_equal(func_pwnorm, true_norm)

    out = fspace.element()
    pwnorm(func, out=out)
    assert all_almost_equal(out, true_norm)


def test_pointwise_norm_gradient_real(exponent, odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    # The operator is not differentiable for exponent 'inf'
    if exponent == float('inf'):
        fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), impl=impl, device=device)
        vfspace = ProductSpace(fspace, 1)
        pwnorm = PointwiseNorm(vfspace, exponent)
        point = vfspace.one()
        with pytest.raises(NotImplementedError):
            pwnorm.derivative(point)
        return

    # 1d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), impl=impl, device=device)
    vfspace = ProductSpace(fspace, 1)
    pwnorm = PointwiseNorm(vfspace, exponent)

    point = noise_element(vfspace)
    direction = noise_element(vfspace)

    # Computing expected result
    tmp = odl.pow(pwnorm(point), 1 - exponent)
    v_field = vfspace.element()
    for i in range(len(v_field)):
        v_field[i] = tmp * point[i] * odl.abs(point[i]) ** (exponent - 2)
    pwinner = odl.PointwiseInner(vfspace, v_field)
    expected_result = pwinner(direction)

    func_pwnorm = pwnorm.derivative(point)

    assert all_almost_equal(func_pwnorm(direction), expected_result)

    # 3d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), impl=impl, device=device)
    vfspace = ProductSpace(fspace, 3)
    pwnorm = PointwiseNorm(vfspace, exponent)

    point = noise_element(vfspace)
    direction = noise_element(vfspace)

    # Computing expected result
    tmp = odl.pow(pwnorm(point), 1 - exponent)
    v_field = vfspace.element()
    for i in range(len(v_field)):
        v_field[i] = tmp * point[i] * odl.abs(point[i]) ** (exponent - 2)
    pwinner = odl.PointwiseInner(vfspace, v_field)
    expected_result = pwinner(direction)

    func_pwnorm = pwnorm.derivative(point)
    assert all_almost_equal(func_pwnorm(direction), expected_result)


def test_pointwise_norm_gradient_real_with_zeros(
        exponent,
        odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    # The gradient is only well-defined in points with zeros if the exponent is
    # >= 2 and < inf
    if exponent < 2 or exponent == float('inf'):
        pytest.skip('differential of operator has singularity for this '
                    'exponent')

    # 1d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), impl=impl, device=device)
    vfspace = ProductSpace(fspace, 1)
    pwnorm = PointwiseNorm(vfspace, exponent)

    backend = fspace.array_backend

    # This makes the point singular for p < 2
    test_point = backend.array_constructor(
        [[[0, 0],  [1, 2]]], device=device)
    test_direction = backend.array_constructor(
        [[[1, 2], [4, 5]]], device=device)

    point = vfspace.element(test_point)
    direction = vfspace.element(test_direction)
    func_pwnorm = pwnorm.derivative(point)

    assert not odl.any(odl.isnan(func_pwnorm(direction)))

    # 3d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), impl=impl, device=device)
    vfspace = ProductSpace(fspace, 3)
    pwnorm = PointwiseNorm(vfspace, exponent)

    # This makes the point singular for p < 2
    test_point = backend.array_constructor(
        [[[0, 0],  
        [1, 2]],
        [[3, 4],
        [0, 0]], 
        [[5, 6],
        [7, 8]]], device=device)
    test_direction = backend.array_constructor(
        [[[0, 1],
        [2, 3]],
        [[4, 5],
        [6, 7]],
        [[8, 9],
        [0, 1]]], device=device)

    point = vfspace.element(test_point)
    direction = vfspace.element(test_direction)
    func_pwnorm = pwnorm.derivative(point)

    assert not odl.any(odl.isnan(func_pwnorm(direction)))

# ---- PointwiseInner ----


def test_pointwise_inner_init_properties(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), impl=impl, device=device)
    vfspace = ProductSpace(fspace, 3, exponent=2)

    # Make sure the code runs and test the properties
    pwinner = PointwiseInner(vfspace, vfspace.one())
    assert pwinner.base_space == fspace
    assert all_equal(pwinner.weights, [1, 1, 1])
    assert not pwinner.is_weighted
    repr(pwinner)

    pwinner = PointwiseInner(vfspace, vfspace.one(), weighting=[1, 2, 3])
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


def test_pointwise_inner_real(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    # 1d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), impl=impl, device=device)
    
    backend = fspace.array_backend
    
    vfspace = ProductSpace(fspace, 1)
    array = backend.array_constructor(
        [[[-1, -3], [2, 0]]], device=device)

    pwinner = PointwiseInner(vfspace, vecfield=array)

    testarr = backend.array_constructor(
        [[[1, 2], [3, 4]]], device=device)

    true_inner = backend.array_namespace.sum(testarr * array, axis=0)

    func = vfspace.element(testarr)
    func_pwinner = pwinner(func)
    assert all_almost_equal(func_pwinner, true_inner)

    out = fspace.element()
    pwinner(func, out=out)
    assert all_almost_equal(out, true_inner)

    # 3d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), impl=impl, device=device)
    vfspace = ProductSpace(fspace, 3)
    array =  backend.array_constructor([[[-1, -3],
                       [2, 0]],
                      [[0, 0],
                       [0, 1]],
                      [[-1, 1],
                       [1, 1]]], device=device)
    pwinner = PointwiseInner(vfspace, vecfield=array)

    testarr =  backend.array_constructor([[[1, 2],
                         [3, 4]],
                        [[0, -1],
                         [0, 1]],
                        [[1, 1],
                         [1, 1]]], device=device)

    true_inner =  backend.array_namespace.sum(testarr * array, axis=0)

    func = vfspace.element(testarr)
    func_pwinner = pwinner(func)
    assert all_almost_equal(func_pwinner, true_inner)

    out = fspace.element()
    pwinner(func, out=out)
    assert all_almost_equal(out, true_inner)


def test_pointwise_inner_complex(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), dtype=complex, impl=impl, device=device)
    vfspace = ProductSpace(fspace, 3)

    backend = fspace.array_backend

    array = backend.array_constructor([[[-1 - 1j, -3],
                       [2, 2j]],
                      [[-1j, 0],
                       [0, 1]],
                      [[-1, 1 + 2j],
                       [1, 1]]], device=device)
    
    pwinner = PointwiseInner(vfspace, vecfield=array)

    testarr = backend.array_constructor([[[1 + 1j, 2],
                         [3, 4 - 2j]],
                        [[0, -1],
                         [0, 1]],
                        [[1j, 1j],
                         [1j, 1j]]], device=device)

    true_inner = backend.array_namespace.sum(testarr * array.conj(), axis=0)

    func = vfspace.element(testarr)
    func_pwinner = pwinner(func)
    assert all_almost_equal(func_pwinner, true_inner)

    out = fspace.element()
    pwinner(func, out=out)
    assert all_almost_equal(out, true_inner)


def test_pointwise_inner_weighted(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), impl=impl, device=device)

    backend = fspace.array_backend

    vfspace = ProductSpace(fspace, 3)
    array = backend.array_constructor([[[-1, -3],
                       [2, 0]],
                      [[0, 0],
                       [0, 1]],
                      [[-1, 1],
                       [1, 1]]], device=device)

    weight = backend.array_constructor([1.0, 2.0, 3.0], device=device)
    pwinner = PointwiseInner(vfspace, vecfield=array, weighting=weight)

    testarr = backend.array_constructor([[[1, 2],
                         [3, 4]],
                        [[0, -1],
                         [0, 1]],
                        [[1, 1],
                         [1, 1]]], device=device)

    true_inner = backend.array_namespace.sum(weight[:, None, None] * testarr * array, axis=0)

    func = vfspace.element(testarr)
    func_pwinner = pwinner(func)
    assert all_almost_equal(func_pwinner, true_inner)

    out = fspace.element()
    pwinner(func, out=out)
    assert all_almost_equal(out, true_inner)


def test_pointwise_inner_adjoint(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    # 1d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), dtype=complex, impl=impl, device=device)

    backend = fspace.array_backend

    vfspace = ProductSpace(fspace, 1)
    array = backend.array_constructor([[[-1, -3],
                       [2, 0]]], device=device)
    pwinner = PointwiseInner(vfspace, vecfield=array)

    testarr = backend.array_constructor([[1 + 1j, 2],
                        [3, 4 - 2j]], device=device)

    true_inner_adj = testarr[None, :, :] * array

    testfunc = fspace.element(testarr)
    testfunc_pwinner_adj = pwinner.adjoint(testfunc)
    assert all_almost_equal(testfunc_pwinner_adj, true_inner_adj)

    out = vfspace.element()
    pwinner.adjoint(testfunc, out=out)
    assert all_almost_equal(out, true_inner_adj)

    # 3d
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), dtype=complex, impl=impl, device=device)
    vfspace = ProductSpace(fspace, 3)
    array = backend.array_constructor([[[-1 - 1j, -3],
                       [2, 2j]],
                      [[-1j, 0],
                       [0, 1]],
                      [[-1, 1 + 2j],
                       [1, 1]]], device=device)
    pwinner = PointwiseInner(vfspace, vecfield=array)

    testarr = backend.array_constructor([[1 + 1j, 2],
                        [3, 4 - 2j]], device=device)

    true_inner_adj = testarr[None, :, :] * array

    testfunc = fspace.element(testarr)
    testfunc_pwinner_adj = pwinner.adjoint(testfunc)
    assert all_almost_equal(testfunc_pwinner_adj, true_inner_adj)

    out = vfspace.element()
    pwinner.adjoint(testfunc, out=out)
    assert all_almost_equal(out, true_inner_adj)


def test_pointwise_inner_adjoint_weighted(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    # Weighted product space only
    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), dtype=complex, impl=impl, device=device)
    backend = fspace.array_backend
    vfspace = ProductSpace(fspace, 3, weighting=[2, 4, 6])
    array = backend.array_constructor([[[-1 - 1j, -3],
                       [2, 2j]],
                      [[-1j, 0],
                       [0, 1]],
                      [[-1, 1 + 2j],
                       [1, 1]]], device=device)
    pwinner = PointwiseInner(vfspace, vecfield=array)

    testarr = backend.array_constructor([[1 + 1j, 2],
                        [3, 4 - 2j]], device=device)

    true_inner_adj = testarr[None, :, :] * array  # same as unweighted case

    testfunc = fspace.element(testarr)
    testfunc_pwinner_adj = pwinner.adjoint(testfunc)
    assert all_almost_equal(testfunc_pwinner_adj, true_inner_adj)

    out = vfspace.element()
    pwinner.adjoint(testfunc, out=out)
    assert all_almost_equal(out, true_inner_adj)

    # Using different weighting in the inner product
    pwinner = PointwiseInner(vfspace, vecfield=array, weighting=[4, 8, 12])

    testarr = backend.array_constructor([[1 + 1j, 2],
                        [3, 4 - 2j]], device=device)

    true_inner_adj = 2 * testarr[None, :, :] * array  # w / v = (2, 2, 2)

    testfunc = fspace.element(testarr)
    testfunc_pwinner_adj = pwinner.adjoint(testfunc)
    assert all_almost_equal(testfunc_pwinner_adj, true_inner_adj)

    out = vfspace.element()
    pwinner.adjoint(testfunc, out=out)
    assert all_almost_equal(out, true_inner_adj)


# ---- PointwiseSum ----


def test_pointwise_sum(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    """PointwiseSum currently depends on PointwiseInner, we verify that."""

    fspace = odl.uniform_discr([0, 0], [1, 1], (2, 2), impl=impl, device=device)
    vfspace = ProductSpace(fspace, 3, exponent=2)

    # Make sure the code runs and test the properties
    psum = PointwiseSum(vfspace)
    assert isinstance(psum, PointwiseInner)
    assert psum.base_space == fspace
    assert all_equal(psum.weights, [1, 1, 1])
    assert all_equal(psum.vecfield, psum.domain.one())


# ---- MatrixOperator ---- #
def sparse_scipy_input(sparse_matrix_format):
    dense_matrix  = np.ones((3, 4))
    if sparse_matrix_format == 'COO':
        sparse_matrix = SparseMatrix('COO', 'scipy', dense_matrix)
    else:
        raise NotImplementedError
    return dense_matrix, sparse_matrix
    
def sparse_pytorch_input(sparse_matrix_format, cuda_device):
    assert sparse_matrix_format == 'COO', NotImplementedError
    indices = [
        #1st row|2nd row|3rd row
        [0,0,0,0,1,1,1,1,2,2,2,2],
        [0,1,2,3,0,1,2,3,0,1,2,3]
        ]
    values = [
        1.0,1.0,1.0,1.0,
        1.0,1.0,1.0,1.0,
        1.0,1.0,1.0,1.0
        ]
    array = [
        [1.0,1.0,1.0,1.0],
        [1.0,1.0,1.0,1.0],
        [1.0,1.0,1.0,1.0]
        ]
    backend = lookup_array_backend('pytorch')
    dense_matrix = backend.array_constructor(array, device=cuda_device)
    sparse_matrix = SparseMatrix('COO', 'pytorch', indices, values, device=cuda_device)
    return dense_matrix, sparse_matrix   



sparse_configs = []
sparse_configs.extend(
    (pytest.param(proj_cfg)
     for proj_cfg in ['COO scipy cpu'])
)

if 'pytorch' in tensor_space_impl_names():
    pytorch_cfgs = []
    for device in lookup_array_backend('pytorch').available_devices:
        pytorch_cfgs.append(f'COO pytorch {device}')

    sparse_configs.extend(
        (pytest.param(proj_cfg, marks=skip_if_no_pytorch)
         for proj_cfg in pytorch_cfgs)
    )

sparse_ids = [
    " format='{}' - backend='{}' - device='{}' ".format(*s.values[0].split())
    for s in sparse_configs
]

@pytest.fixture(scope='module', params=sparse_configs, ids=sparse_ids)
def matrix_input(request):
    format, backend, device = request.param.split()
    if backend == 'scipy':
        return sparse_scipy_input(format)
    elif backend == 'pytorch':
        return sparse_pytorch_input(format, device)
    else:
        raise ValueError
    
def invertible_sparse_scipy_input(sparse_matrix_format):
    assert sparse_matrix_format == 'COO', NotImplementedError
    dense_matrix  = np.ones((3, 3)) + 4.0 * np.eye(3)  # invertible
    sparse_matrix = SparseMatrix('COO', 'scipy', dense_matrix)
    return dense_matrix, sparse_matrix

def invertible_sparse_pytorch_input(sparse_matrix_format, cuda_device):
    assert sparse_matrix_format == 'COO', NotImplementedError
    indices = [
        #1st row|2nd row|3rd row
        [0,0,0,1,1,1,2,2,2],
        [0,1,2,0,1,2,0,1,2]
        ]
    values = [
        5.0,1.0,1.0,
        1.0,5.0,1.0,
        1.0,1.0,5.0
        ]
    array = [
        [5.0,1.0,1.0],
        [1.0,5.0,1.0],
        [1.0,1.0,5.0]
        ]
    backend = lookup_array_backend('pytorch')
    dense_matrix = backend.array_constructor(array, device=cuda_device)
    sparse_matrix = SparseMatrix('COO', 'pytorch', indices, values, device=cuda_device)
    return dense_matrix, sparse_matrix   
    
@pytest.fixture(scope='module', params=sparse_configs, ids=sparse_ids)
def invertible_matrix_input(request):
    format, backend, device = request.param.split()
    if backend == 'scipy':
        return invertible_sparse_scipy_input(format)
    elif backend == 'pytorch':
        return invertible_sparse_pytorch_input(format, device)
    else:
        raise ValueError

def test_matrix_op_init(matrix_input):
    """Test initialization and properties of matrix operators."""
    dense_matrix, sparse_matrix = matrix_input

    dense_matrix, backend = get_array_and_backend(dense_matrix)
    impl = backend.impl
    device = dense_matrix.device
    # Just check if the code runs
    MatrixOperator(dense_matrix)
    MatrixOperator(sparse_matrix)

    # Test default domain and range
    mat_op = MatrixOperator(dense_matrix)
    assert mat_op.domain == odl.tensor_space(4, dense_matrix.dtype, impl=impl, device=device)
    assert mat_op.range == odl.tensor_space(3, dense_matrix.dtype, impl=impl, device=device)
    assert odl.all(mat_op.matrix == dense_matrix)

    mat_op = MatrixOperator(sparse_matrix)
    assert mat_op.domain == odl.tensor_space(4, dense_matrix.dtype, impl=impl, device=device)
    assert mat_op.range == odl.tensor_space(3, dense_matrix.dtype, impl=impl, device=device)
    if impl == 'numpy':
        assert (mat_op.matrix != sparse_matrix).getnnz() == 0
    # Pytorch does not support == and != betweend sparse tensors
    elif impl == 'pytorch':
        assert len(mat_op.matrix) == len(sparse_matrix) 
    else:
        raise NotImplementedError
    # Explicit domain and range
    dom = odl.tensor_space(4, dense_matrix.dtype, impl=impl, device=device)
    ran = odl.tensor_space(3, dense_matrix.dtype, impl=impl, device=device)

    mat_op = MatrixOperator(dense_matrix, domain=dom, range=ran)
    assert mat_op.domain == dom
    assert mat_op.range == ran

    mat_op = MatrixOperator(sparse_matrix, domain=dom, range=ran)
    assert mat_op.domain == dom
    assert mat_op.range == ran

    # Bad 1d sizes
    with pytest.raises(ValueError):
        MatrixOperator(dense_matrix, domain=odl.cn(4, impl=impl, device=device), range=odl.cn(4, impl=impl, device=device))
    with pytest.raises(ValueError):
        MatrixOperator(dense_matrix, range=odl.cn(4, impl=impl, device=device))
    # Invalid range dtype
    with pytest.raises(ValueError):
        if impl == 'numpy':
            MatrixOperator(dense_matrix.astype(complex), range=odl.rn(4, impl=impl, device=device))
        elif impl == 'pytorch':
            MatrixOperator(dense_matrix.to(complex), range=odl.rn(4, impl=impl, device=device))
        else:
            raise NotImplementedError

    # Data type promotion
    # real space, complex matrix -> complex space
    dom = odl.rn(4, impl=impl, device=device)
    if impl == 'numpy':
        mat_op = MatrixOperator(dense_matrix.astype(complex), domain=dom, impl=impl, device=device)

    elif impl == 'pytorch':
        mat_op = MatrixOperator(dense_matrix.to(complex), domain=dom,
        impl=impl, device=device)
    else:
        raise NotImplementedError
    assert mat_op.domain == dom
    assert mat_op.range == odl.cn(3, impl=impl, device=device)

    # complex space, real matrix -> complex space
    dom = odl.cn(4, impl=impl, device=device)
    mat_op = MatrixOperator(dense_matrix.real, domain=dom)
    assert mat_op.domain == dom
    assert mat_op.range == odl.cn(3, impl=impl, device=device)

    # Multi-dimensional spaces
    dom = odl.tensor_space((6, 5, 4), dense_matrix.dtype, impl=impl, device=device)
    ran = odl.tensor_space((6, 5, 3), dense_matrix.dtype, impl=impl, device=device)
    mat_op = MatrixOperator(dense_matrix, domain=dom, axis=2)
    assert mat_op.range == ran
    mat_op = MatrixOperator(dense_matrix, domain=dom, range=ran, axis=2)
    assert mat_op.range == ran

    with pytest.raises(ValueError):
        bad_dom = odl.tensor_space((6, 6, 6), dense_matrix.dtype)  # wrong shape
        MatrixOperator(dense_matrix, domain=bad_dom)
    with pytest.raises(ValueError):
        dom = odl.tensor_space((6, 5, 4), dense_matrix.dtype)
        bad_ran = odl.tensor_space((6, 6, 6), dense_matrix.dtype)  # wrong shape
        MatrixOperator(dense_matrix, domain=dom, range=bad_ran)
    with pytest.raises(ValueError):
        MatrixOperator(dense_matrix, domain=dom, axis=1)
    with pytest.raises(ValueError):
        MatrixOperator(dense_matrix, domain=dom, axis=0)
    with pytest.raises(ValueError):
        bad_ran = odl.tensor_space((6, 3, 4), dense_matrix.dtype, impl=impl, device=device)
        MatrixOperator(dense_matrix, domain=dom, range=bad_ran, axis=2)
    with pytest.raises(ValueError):
        bad_dom_for_sparse = odl.rn((6, 5, 4), impl=impl, device=device)
        MatrixOperator(sparse_matrix, domain=bad_dom_for_sparse, axis=2, impl=impl, device=device)

    # Init with uniform_discr space (subclass of TensorSpace)
    dom = odl.uniform_discr(0, 1, 4, dtype=dense_matrix.dtype, impl=impl, device=device)
    ran = odl.uniform_discr(0, 1, 3, dtype=dense_matrix.dtype, impl=impl, device=device)
    MatrixOperator(dense_matrix, domain=dom, range=ran)

    # Make sure this runs and returns something string-like
    assert str(mat_op) > ''
    assert repr(mat_op) > ''


def test_matrix_op_call_implicit(matrix_input):
    """Validate result from calls to matrix operators against Numpy."""
    dense_matrix, sparse_matrix = matrix_input

    dense_matrix, backend = get_array_and_backend(dense_matrix)
    impl = backend.impl
    device = dense_matrix.device
    ns = backend.array_namespace

    # Default 1d case
    dmat_op = MatrixOperator(dense_matrix)
    smat_op = MatrixOperator(sparse_matrix)
    xarr, x = noise_elements(dmat_op.domain)
    # if impl == 'numpy':
    #     true_result = dense_matrix.dot(xarr)
    # elif impl == 'pytorch':
    
    true_result = ns.tensordot(dense_matrix, xarr, axes=([1], [0]))
    assert all_almost_equal(dmat_op(x), true_result)
    assert all_almost_equal(smat_op(x), true_result)
    out = dmat_op.range.element()
    dmat_op(x, out=out)
    assert all_almost_equal(out, true_result)
    smat_op(x, out=out)
    assert all_almost_equal(out, true_result)

    # Multi-dimensional case
    

    domain = odl.rn((2, 2, 4),impl=impl,device=device)
    mat_op = MatrixOperator(dense_matrix, domain, axis=2)
    xarr, x = noise_elements(mat_op.domain)
    true_result = ns.moveaxis(ns.tensordot(dense_matrix, xarr, axes=([1], [2])), 0, 2)
    assert all_almost_equal(mat_op(x), true_result)
    out = mat_op.range.element()
    mat_op(x, out=out)
    assert all_almost_equal(out, true_result)


def test_matrix_op_call_explicit(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    """Validate result from call to matrix op against explicit calculation."""

    space = odl.rn((3,2), impl=impl, device=device)
    mat = space.one().data

    backend = space.array_backend
    ns = space.array_namespace

    xarr = backend.array_constructor([[[0, 1],
                      [2, 3]],
                     [[4, 5],
                      [6, 7]]], dtype=float, device=device)

    # Multiplication along `axis` with `mat` is the same as summation
    # along `axis` and stacking 3 times along the same axis
    for axis in range(3):
        mat_op = MatrixOperator(mat, domain=odl.rn(xarr.shape, impl=impl, device=device),
                                axis=axis)
        result = mat_op(xarr)
        if impl == 'numpy':
            true_result = ns.repeat(ns.sum(xarr, axis=axis, keepdims=True),
                                repeats=3, axis=axis)
        elif impl == 'pytorch':
            true_result = ns.repeat_interleave(ns.sum(xarr, axis=axis, keepdims=True),
                                repeats=3, axis=axis)
        else:
            raise ValueError(f'Not implemented for impl = {impl}')
        assert result.shape == true_result.shape
        assert odl.allclose(result, true_result)


def test_matrix_op_adjoint(matrix_input):
    """Test if the adjoint of matrix operators is correct."""
    dense_matrix, sparse_matrix = matrix_input

    dense_matrix, backend = get_array_and_backend(dense_matrix)
    impl = backend.impl
    device = dense_matrix.device
    ns = backend.array_namespace
    tol = 2 * len(dense_matrix) * ns.finfo(dense_matrix.dtype).resolution
    # Default 1d case
    dmat_op = MatrixOperator(dense_matrix)
    smat_op = MatrixOperator(sparse_matrix)
    x = noise_element(dmat_op.domain)
    y = noise_element(dmat_op.range)

    inner_ran = dmat_op(x).inner(y)
    inner_dom = x.inner(dmat_op.adjoint(y))
    assert inner_ran == pytest.approx(inner_dom, rel=tol, abs=tol)
    inner_ran = smat_op(x).inner(y)
    inner_dom = x.inner(smat_op.adjoint(y))
    assert inner_ran == pytest.approx(inner_dom, rel=tol, abs=tol)

    # Multi-dimensional case
    domain = odl.tensor_space((2, 2, 4), impl=impl, device=device)
    mat_op = MatrixOperator(dense_matrix, domain, axis=2, impl=impl, device=device)
    x = noise_element(mat_op.domain)
    y = noise_element(mat_op.range)
    inner_ran = mat_op(x).inner(y)
    inner_dom = x.inner(mat_op.adjoint(y))
    assert inner_ran == pytest.approx(inner_dom, rel=tol, abs=tol)


def test_matrix_op_inverse(invertible_matrix_input):
    """Test if the inverse of matrix operators is correct."""
    dense_matrix, sparse_matrix = invertible_matrix_input

    # Default 1d case
    dmat_op = MatrixOperator(dense_matrix)
    smat_op = MatrixOperator(sparse_matrix)
    x = noise_element(dmat_op.domain)
    md_x = dmat_op(x)
    mdinv_md_x = dmat_op.inverse(md_x)
    assert all_almost_equal(x, mdinv_md_x)
    ms_x = smat_op(x)
    msinv_ms_x = smat_op.inverse(ms_x)
    assert all_almost_equal(x, msinv_ms_x)

    # Multi-dimensional case
    dense_matrix, backend = get_array_and_backend(dense_matrix)
    impl = backend.impl
    device = dense_matrix.device
    domain = odl.tensor_space((2, 2, 3), impl=impl, device=device)
    mat_op = MatrixOperator(dense_matrix, domain, axis=2)
    x = noise_element(mat_op.domain)
    m_x = mat_op(x)
    minv_m_x = mat_op.inverse(m_x)
    assert all_almost_equal(x, minv_m_x)


def test_sampling_operator_adjoint(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    """Validate basic properties of `SamplingOperator.adjoint`."""
    # 1d space
    space = odl.uniform_discr([-1], [1], shape=(3), impl=impl, device=device)
    sampling_points = [[0, 1, 1, 0]]
    x = space.element([1, 2, 3])
    op = odl.SamplingOperator(space, sampling_points)
    assert op.adjoint(op(x)).inner(x) == pytest.approx(op(x).inner(op(x)))

    op = odl.SamplingOperator(space, sampling_points, variant='integrate')
    assert op.adjoint(op(x)).inner(x) == pytest.approx(op(x).inner(op(x)))

    # 2d space
    space = odl.uniform_discr([-1, -1], [1, 1], shape=(2, 3), impl=impl, device=device)
    x = space.element([[1, 2, 3],
                       [4, 5, 6]])
    sampling_points = [[0, 1, 1, 0],
                       [0, 1, 2, 0]]
    op = odl.SamplingOperator(space, sampling_points)
    assert op.adjoint(op(x)).inner(x) == pytest.approx(op(x).inner(op(x)))

    # The ``'integrate'`` variant adjoint puts ones at the indices in
    # `sampling_points``, multiplied by their multiplicity:
    op = odl.SamplingOperator(space, sampling_points, variant='integrate')
    assert op.adjoint(op(x)).inner(x) == pytest.approx(op(x).inner(op(x)))


if __name__ == '__main__':
    odl.core.util.test_file(__file__)
