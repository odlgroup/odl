# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for `DiscretizedSpace`."""

from __future__ import division

import numpy as np

import odl
import pytest
from odl.discr.discr_space import DiscretizedSpace, DiscretizedSpaceElement
from odl.space.base_tensors import TensorSpace, default_dtype
from odl.space.npy_tensors import NumpyTensor
from odl.util.dtype_utils import COMPLEX_DTYPES
from odl.util.testutils import (
    all_almost_equal, all_equal, noise_elements, simple_fixture, default_precision_dict)
from odl.array_API_support import lookup_array_backend
# --- Pytest fixtures --- #


exponent = simple_fixture('exponent', [2.0, 1.0, float('inf'), 0.5, 1.5])
power = simple_fixture('power', [1.0, 2.0, 0.5, -0.5, -1.0, -2.0])
shape = simple_fixture('shape', [(2, 3, 4), (3, 4), (2,), (1,), (1, 1, 1)])
power = simple_fixture('power', [1.0, 2.0, 0.5, -0.5, -1.0, -2.0])


# --- DiscretizedSpace --- #


def test_discretizedspace_init(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    """Test initialization and basic properties of DiscretizedSpace."""
    # Real space
    part = odl.uniform_partition([0, 0], [1, 1], (2, 4))
    tspace = odl.rn(part.shape, impl=impl, device=device)

    discr = DiscretizedSpace(part, tspace)
    assert discr.tspace == tspace
    assert discr.partition == part
    assert discr.exponent == tspace.exponent
    assert discr.axis_labels == ('$x$', '$y$')
    assert discr.is_real

    # Complex space
    tspace_c = odl.cn(part.shape, impl=impl, device=device)
    discr = DiscretizedSpace(part, tspace_c)
    assert discr.is_complex

    # Make sure repr shows something
    assert repr(discr) != ''

    # Error scenarios
    part_1d = odl.uniform_partition(0, 1, 2)
    with pytest.raises(ValueError):
        DiscretizedSpace(part_1d, tspace)  # wrong dimensionality

    part_diffshp = odl.uniform_partition([0, 0], [1, 1], (3, 4))
    with pytest.raises(ValueError):
        DiscretizedSpace(part_diffshp, tspace)  # shape mismatch


def test_empty(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    """Check if empty spaces behave as expected and all methods work."""
    discr = odl.uniform_discr([], [], (), impl=impl, device=device)

    assert discr.axis_labels == ()
    assert discr.tangent_bundle == odl.ProductSpace(field=odl.RealNumbers())
    assert discr.complex_space == odl.uniform_discr([], [], (), dtype=complex, impl=impl, device=device)
    hash(discr)
    assert repr(discr) != ''

    elem = discr.element(1.0)
    assert all_equal(elem.asarray(), 1.0)
    assert all_equal(elem.real, 1.0)
    assert all_equal(elem.imag, 0.0)
    assert all_equal(elem.conj(), 1.0)


# --- uniform_discr --- #


def test_factory_dtypes(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    """Check dtypes of spaces from factory function."""
    real_float_dtypes = ["float32", "float64"]
    nonfloat_dtypes = ["int8", "int16", "int32", "int64",
                       "uint8", "uint16", "uint32", "uint64"]
    complex_float_dtypes = ["complex64", "complex128"]

    for dtype in real_float_dtypes:
        try:
            discr = odl.uniform_discr(0, 1, 10, impl=impl, dtype=dtype, device=device)
        except TypeError:
            continue
        else:
            assert isinstance(discr.tspace, TensorSpace)
            assert discr.tspace.impl == impl
            assert discr.is_real

    for dtype in nonfloat_dtypes:
        try:
            discr = odl.uniform_discr(0, 1, 10, impl=impl, dtype=dtype, device=device)
        except TypeError:
            continue
        else:
            assert isinstance(discr.tspace, TensorSpace)
            assert discr.tspace.impl == impl
            assert discr.tspace.element().space.dtype_identifier == dtype

    for dtype in complex_float_dtypes:
        try:
            discr = odl.uniform_discr(0, 1, 10, impl=impl, dtype=dtype, device=device)
        except TypeError:
            continue
        else:
            assert isinstance(discr.tspace, TensorSpace)
            assert discr.tspace.impl == impl
            assert discr.is_complex
            assert discr.tspace.element().space.dtype_identifier == dtype


def test_uniform_discr_init_real(odl_impl_device_pairs):
    """Test initialization and basic properties with uniform_discr, real."""
    impl, device = odl_impl_device_pairs

    # 1D
    discr = odl.uniform_discr(0, 1, 10, impl=impl, device=device)
    assert isinstance(discr, DiscretizedSpace)
    assert isinstance(discr.tspace, TensorSpace)
    assert discr.impl == impl
    assert discr.is_real
    assert discr.tspace.exponent == 2.0
    assert discr.dtype == default_dtype(impl, field=odl.RealNumbers())
    assert discr.is_real
    assert not discr.is_complex
    assert all_equal(discr.min_pt, [0])
    assert all_equal(discr.max_pt, [1])
    assert discr.shape == (10,)
    assert repr(discr)

    discr = odl.uniform_discr(0, 1, 10, impl=impl, exponent=1.0, device=device)
    assert discr.exponent == 1.0

    # 2D
    discr = odl.uniform_discr([0, 0], [1, 1], (5, 5), impl=impl, device=device)
    assert all_equal(discr.min_pt, np.array([0, 0]))
    assert all_equal(discr.max_pt, np.array([1, 1]))
    assert discr.shape == (5, 5)

    # nd
    discr = odl.uniform_discr([0] * 10, [1] * 10, (5,) * 10, impl=impl, device=device)
    assert all_equal(discr.min_pt, np.zeros(10))
    assert all_equal(discr.max_pt, np.ones(10))
    assert discr.shape == (5,) * 10


# ## Why does this test fail if impl != numpy?
# def test_uniform_discr_init_complex(odl_tspace_impl):
#     """Test initialization and basic properties with uniform_discr, complex."""
#     impl = odl_tspace_impl
#     if impl != 'numpy':
#         pytest.xfail(reason='complex dtypes not supported')

#     discr = odl.uniform_discr(0, 1, 10, dtype='complex', impl=impl)
#     assert discr.is_complex
#     assert discr.dtype == default_dtype(impl, field=odl.ComplexNumbers())


# --- DiscretizedSpace methods --- #


def test_discretizedspace_element(odl_impl_device_pairs):
    """Test creation and membership of DiscretizedSpace elements."""
    impl, device = odl_impl_device_pairs
    # Creation from scratch
    # 1D
    discr = odl.uniform_discr(0, 1, 3, impl=impl, device=device)
    weight = 1.0 if exponent == float('inf') else discr.cell_volume
    tspace = odl.rn(3, weighting=weight, impl=impl, device=device)
    elem = discr.element()
    assert elem in discr
    assert elem.tensor in tspace

    # 2D
    discr = odl.uniform_discr([0, 0], [1, 1], (3, 3), impl=impl, device=device)
    weight = 1.0 if exponent == float('inf') else discr.cell_volume
    tspace = odl.rn((3, 3), weighting=weight, impl=impl, device=device)
    elem = discr.element()
    assert elem in discr
    assert elem.tensor in tspace


def test_discretizedspace_element_from_array(odl_impl_device_pairs):
    """Test creation of DiscretizedSpace elements from arrays."""
    impl, device = odl_impl_device_pairs
    # 1D
    discr = odl.uniform_discr(0, 1, 3, impl=impl, device=device)
    elem = discr.element([1, 2, 3])
    assert all_equal(elem.tensor, [1, 2, 3])

    assert isinstance(elem, DiscretizedSpaceElement)
    assert isinstance(elem.tensor, discr.tspace.element_type)
    assert all_equal(elem.tensor, [1, 2, 3])

# That should be deprecated
# def test_element_from_array_2d(odl_elem_order, odl_impl_device_pairs):
#     """Test element in 2d with different orderings."""
#     impl, device = odl_impl_device_pairs
#     order = odl_elem_order
#     discr = odl.uniform_discr([0, 0], [1, 1], [2, 2], impl=impl, device=device)
#     elem = discr.element([[1, 2],
#                           [3, 4]], order=order)

#     assert isinstance(elem, DiscretizedSpaceElement)
#     assert isinstance(elem.tensor, NumpyTensor)
#     assert all_equal(elem, [[1, 2],
#                             [3, 4]])

#     assert elem.tensor.data.flags['C_CONTIGUOUS']

#     with pytest.raises(ValueError):
#         discr.element([1, 2, 3])  # wrong size & shape
#     with pytest.raises(ValueError):
#         discr.element([1, 2, 3, 4])  # wrong shape
#     with pytest.raises(ValueError):
#         discr.element([[1],
#                        [2],
#                        [3],
#                        [4]])  # wrong shape


def test_element_from_function_1d(odl_impl_device_pairs):
    """Test creation of DiscretizedSpace elements from functions in 1D."""
    impl, device = odl_impl_device_pairs
    space = odl.uniform_discr(-1, 1, 4, impl=impl, device=device)
    points = space.points().squeeze()

    # Without parameter
    def f(x):
        return x * 2 + np.maximum(x, 0)

    elem_f = space.element(f)
    true_elem = [x * 2 + max(x, 0) for x in points]
    assert all_equal(elem_f, true_elem)

    # Without parameter, using same syntax as in higher dimensions
    def f(x):
        return x[0] * 2 + np.maximum(x[0], 0)

    elem_f = space.element(f)
    true_elem = [x * 2 + max(x, 0) for x in points]
    assert all_equal(elem_f, true_elem)

    # With parameter
    def f(x, **kwargs):
        c = kwargs.pop('c', 0)
        return x * c + np.maximum(x, 0)

    elem_f_default = space.element(f)
    true_elem = [x * 0 + max(x, 0) for x in points]
    assert all_equal(elem_f_default, true_elem)

    elem_f_2 = space.element(f, c=2)
    true_elem = [x * 2 + max(x, 0) for x in points]
    assert all_equal(elem_f_2, true_elem)

    # Using a lambda
    elem_lam = space.element(lambda x: -x ** 2)
    true_elem = [-x ** 2 for x in points]
    assert all_equal(elem_lam, true_elem)

    # Broadcast from constant function
    elem_lam = space.element(lambda x: 1.0)
    true_elem = [1.0 for _ in points]
    assert all_equal(elem_lam, true_elem)


def test_element_from_function_2d(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    """Test creation of DiscretizedSpace elements from functions in 2D."""
    space = odl.uniform_discr([-1, -1], [1, 1], (2, 3), impl=impl, device=device)
    points = space.points()

    # Without parameter
    def f(x):
        return x[0] ** 2 + np.maximum(x[1], 0)

    elem_f = space.element(f)
    true_elem = np.reshape(
        [x[0] ** 2 + max(x[1], 0) for x in points], space.shape
    )
    assert all_equal(elem_f, true_elem)

    # With parameter
    def f(x, **kwargs):
        c = kwargs.pop('c', 0)
        return x[0] ** 2 + np.maximum(x[1], c)

    elem_f_default = space.element(f)
    true_elem = np.reshape(
        [x[0] ** 2 + max(x[1], 0) for x in points], space.shape
    )
    assert all_equal(elem_f_default, true_elem)

    elem_f_2 = space.element(f, c=1)
    true_elem = np.reshape(
        [x[0] ** 2 + max(x[1], 1) for x in points], space.shape
    )
    assert all_equal(elem_f_2, true_elem)

    # Using a lambda
    elem_lam = space.element(lambda x: x[0] - x[1])
    true_elem = np.reshape([x[0] - x[1] for x in points], space.shape)
    assert all_equal(elem_lam, true_elem)

    # Using broadcasting
    elem_lam = space.element(lambda x: x[0])
    true_elem = np.reshape([x[0] for x in points], space.shape)
    assert all_equal(elem_lam, true_elem)

    elem_lam = space.element(lambda x: x[1])
    true_elem = np.reshape([x[1] for x in points], space.shape)
    assert all_equal(elem_lam, true_elem)

    # Broadcast from constant function
    elem_lam = space.element(lambda x: 1.0)
    true_elem = np.reshape([1.0 for _ in points], space.shape)
    assert all_equal(elem_lam, true_elem)


def test_discretizedspace_zero_one(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    """Test the zero and one element creators of DiscretizedSpace."""
    discr = odl.uniform_discr(0, 1, 3, impl=impl, device=device)

    zero = discr.zero()
    assert zero in discr
    assert all_equal(zero, [0, 0, 0])

    one = discr.one()
    assert one in discr
    assert all_equal(one, [1, 1, 1])


def test_equals_space(exponent, odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    x1 = odl.uniform_discr(0, 1, 3, exponent=exponent, impl=impl, device=device)
    x2 = odl.uniform_discr(0, 1, 3, exponent=exponent, impl=impl, device=device)
    y = odl.uniform_discr(0, 1, 4, exponent=exponent, impl=impl, device=device)

    assert x1 is x1
    assert x1 is not x2
    assert x1 is not y
    assert x1 == x1
    assert x1 == x2
    assert x1 != y
    assert hash(x1) == hash(x2)
    assert hash(x1) != hash(y)


def test_equals_vec(exponent, odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    discr = odl.uniform_discr(0, 1, 3, exponent=exponent, impl=impl, device=device)
    discr2 = odl.uniform_discr(0, 1, 4, exponent=exponent, impl=impl, device=device)
    x1 = discr.element([1, 2, 3])
    x2 = discr.element([1, 2, 3])
    y = discr.element([2, 2, 3])
    z = discr2.element([1, 2, 3, 4])

    assert x1 is x1
    assert x1 is not x2
    assert x1 is not y
    assert x1 == x1
    assert x1 == x2
    assert x1 != y
    assert x1 != z


def _test_unary_operator(discr, function):
    # Verify that the statement y=function(x) gives equivalent results
    # to NumPy
    x_arr, x = noise_elements(discr)
    y_arr = function(x_arr)
    y = function(x)
    assert all_almost_equal([x, y], [x_arr, y_arr])


def _test_binary_operator(discr, function):
    # Verify that the statement z=function(x,y) gives equivalent results
    # to NumPy
    [x_arr, y_arr], [x, y] = noise_elements(discr, 2)
    z_arr = function(x_arr, y_arr)
    z = function(x, y)
    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr])


def test_operators(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    # Test of all operator overloads against the corresponding NumPy
    # implementation
    discr = odl.uniform_discr(0, 1, 10, impl=impl, device=device)

    # Unary operators
    _test_unary_operator(discr, lambda x: +x)
    _test_unary_operator(discr, lambda x: -x)

    # Scalar addition
    for scalar in [-31.2, -1, 0, 1, 2.13]:
        def iadd(x):
            x += scalar
        _test_unary_operator(discr, iadd)
        _test_unary_operator(discr, lambda x: x + scalar)

    # Scalar subtraction
    for scalar in [-31.2, -1, 0, 1, 2.13]:
        def isub(x):
            x -= scalar
        _test_unary_operator(discr, isub)
        _test_unary_operator(discr, lambda x: x - scalar)

    # Scalar multiplication
    for scalar in [-31.2, -1, 0, 1, 2.13]:
        def imul(x):
            x *= scalar
        _test_unary_operator(discr, imul)
        _test_unary_operator(discr, lambda x: x * scalar)

    # Scalar division
    for scalar in [-31.2, -1, 1, 2.13]:
        def idiv(x):
            x /= scalar
        _test_unary_operator(discr, idiv)
        _test_unary_operator(discr, lambda x: x / scalar)

    # Incremental operations
    def iadd(x, y):
        x += y

    def isub(x, y):
        x -= y

    def imul(x, y):
        x *= y

    def idiv(x, y):
        x /= y

    _test_binary_operator(discr, iadd)
    _test_binary_operator(discr, isub)
    _test_binary_operator(discr, imul)
    _test_binary_operator(discr, idiv)

    # Incremental operators with aliased inputs
    def iadd_aliased(x):
        x += x

    def isub_aliased(x):
        x -= x

    def imul_aliased(x):
        x *= x

    def idiv_aliased(x):
        x /= x

    _test_unary_operator(discr, iadd_aliased)
    _test_unary_operator(discr, isub_aliased)
    _test_unary_operator(discr, imul_aliased)
    _test_unary_operator(discr, idiv_aliased)

    # Binary operators
    _test_binary_operator(discr, lambda x, y: x + y)
    _test_binary_operator(discr, lambda x, y: x - y)
    _test_binary_operator(discr, lambda x, y: x * y)
    _test_binary_operator(discr, lambda x, y: x / y)

    # Binary with aliased inputs
    _test_unary_operator(discr, lambda x: x + x)
    _test_unary_operator(discr, lambda x: x - x)
    _test_unary_operator(discr, lambda x: x * x)
    _test_unary_operator(discr, lambda x: x / x)


def test_getitem(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    discr = odl.uniform_discr(0, 1, 3, impl=impl, device=device)
    elem = discr.element([1, 2, 3])

    assert all_equal(elem, [1, 2, 3])


def test_getslice(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    discr = odl.uniform_discr(0, 1, 3, impl=impl, device=device)
    elem = discr.element([1, 2, 3])
    tspace_impl = discr.tspace.element_type
    assert isinstance(elem[:], tspace_impl)
    assert all_equal(elem[:], [1, 2, 3])

    discr = odl.uniform_discr(0, 1, 3, dtype=complex)
    tspace_impl = discr.tspace.element_type
    elem = discr.element([1 + 2j, 2 - 2j, 3])

    assert isinstance(elem[:], tspace_impl)
    assert all_equal(elem[:], [1 + 2j, 2 - 2j, 3])


def test_setitem(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    discr = odl.uniform_discr(0, 1, 3, impl=impl, device=device)
    elem = discr.element([1, 2, 3])
    elem[0] = 4
    elem[1] = 5
    elem[2] = 6

    assert all_equal(elem, [4, 5, 6])


def test_setitem_nd(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs

    # 1D
    discr = odl.uniform_discr(0, 1, 3, impl=impl, device=device)
    elem = discr.element([1, 2, 3])

    backend = discr.array_backend

    elem[:] = [4, 5, 6]
    assert all_equal(elem, [4, 5, 6])

    elem[:] = backend.array_constructor([3, 2, 1], device=device)
    assert all_equal(elem, [3, 2, 1])

    elem[:] = 0
    assert all_equal(elem, [0, 0, 0])

    elem[:] = [1]
    assert all_equal(elem, [1, 1, 1])

    error = ValueError if impl =='numpy' else RuntimeError
    with pytest.raises(error):
        elem[:] = [0, 0]  # bad shape

    with pytest.raises(error):
        elem[:] = [0, 0, 1, 2]  # bad shape

    # 2D
    discr = odl.uniform_discr([0, 0], [1, 1], [3, 2])

    elem = discr.element([[1, 2],
                          [3, 4],
                          [5, 6]])

    elem[:] = [[-1, -2],
               [-3, -4],
               [-5, -6]]
    assert all_equal(elem, [[-1, -2],
                            [-3, -4],
                            [-5, -6]])

    # arr = np.arange(6, 12).reshape([3, 2])
    arr = odl.arange(impl=impl, start=6, stop=12).reshape([3, 2])
    elem[:] = arr
    assert all_equal(elem, arr)

    elem[:] = 0
    assert all_equal(elem, odl.zeros(impl=impl, shape=elem.shape))

    elem[:] = [1]
    assert all_equal(elem, odl.ones(impl=impl, shape=elem.shape))

    elem[:] = [0, 0]  # broadcasting assignment
    assert all_equal(elem,odl.zeros(impl=impl, shape=elem.shape))

    with pytest.raises(ValueError):
        elem[:] = [0, 0, 0]  # bad shape

    with pytest.raises(ValueError):
        elem[:] = odl.arange(impl=impl, start=6)  # bad shape (6,)

    with pytest.raises(ValueError):
        elem[:] = odl.ones(impl=impl, shape=(2, 3))[..., None]  # bad shape (2, 3, 1)

    with pytest.raises(ValueError):
        arr = odl.arange(impl=impl, start=6, stop=12).reshape([3, 2])
        elem[:] = arr.T  # bad shape (2, 3)

    # nD
    shape = (3,) * 3 + (4,) * 3
    discr = odl.uniform_discr([0] * 6, [1] * 6, shape, impl=impl, device=device)
    size = np.prod(shape)
    elem = discr.element(np.zeros(shape))

    arr = odl.arange(impl=impl, start=size).reshape(shape)

    elem[:] = arr
    assert all_equal(elem, arr)

    elem[:] = 0
    assert all_equal(elem, odl.zeros(impl=impl, shape=elem.shape))

    elem[:] = [1]
    assert all_equal(elem, odl.ones(impl=impl, shape=elem.shape))

    error = ValueError if impl =='numpy' else RuntimeError
    with pytest.raises(error):
        # Reversed shape -> bad
        elem[:] = odl.arange(impl=impl, start=size).reshape((4,) * 3 + (3,) * 3)


def test_setslice(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    discr = odl.uniform_discr(0, 1, 3, impl=impl, device=device)
    elem = discr.element([1, 2, 3])

    elem[:] = [4, 5, 6]
    assert all_equal(elem, [4, 5, 6])


def test_asarray_2d(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    """Test the asarray method."""
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 2], impl=impl, device=device)
    elem = discr.element([[1, 2],
                          [3, 4]])

    arr = elem.asarray()
    assert all_equal(arr, [[1, 2],
                           [3, 4]])
    
    # test out parameter
    out_c = odl.empty(impl=impl, shape=[2, 2])
    result_c = elem.asarray(out=out_c)
    assert result_c is out_c
    assert all_equal(out_c, [[1, 2],
                             [3, 4]])
    # Try wrong shape
    out_wrong_shape = odl.empty(impl=impl, shape=[2, 3])
    error = ValueError if impl =='numpy' else RuntimeError
    with pytest.raises(error):
        elem.asarray(out=out_wrong_shape)


def test_transpose(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 2], impl=impl, device=device)
    x = discr.element([[1, 2], [3, 4]])
    y = discr.element([[5, 6], [7, 8]])

    assert isinstance(x.T, odl.Operator)
    assert x.T.is_linear

    assert x.T(y) == x.inner(y)
    assert x.T.T == x
    assert all_equal(x.T.adjoint(1.0), x)


def test_cell_sides(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    # Non-degenerated case, should be same as cell size
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 2], impl=impl, device=device)
    elem = discr.element()

    assert all_equal(discr.cell_sides, [0.5] * 2)
    assert all_equal(elem.cell_sides, [0.5] * 2)

    # Degenerated case, uses interval size in 1-point dimensions
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 1], impl=impl, device=device)
    elem = discr.element()

    assert all_equal(discr.cell_sides, [0.5, 1])
    assert all_equal(elem.cell_sides, [0.5, 1])


def test_cell_volume(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    # Non-degenerated case
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 2], impl=impl, device=device)
    elem = discr.element()

    assert discr.cell_volume == 0.25
    assert elem.cell_volume == 0.25

    # Degenerated case, uses interval size in 1-point dimensions
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 1])
    elem = discr.element()

    assert discr.cell_volume == 0.5
    assert elem.cell_volume == 0.5


def test_astype(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    rdiscr = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype='float64', impl=impl, device=device)
    cdiscr = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype='complex128', impl=impl, device=device)
    rdiscr_s = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype='float32', impl=impl, device=device)
    cdiscr_s = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype='complex64', impl=impl, device=device)

    # Real
    assert rdiscr.astype('float32') == rdiscr_s
    assert rdiscr.astype('float64') is rdiscr
    assert rdiscr.real_space is rdiscr
    assert rdiscr.astype('complex64') == cdiscr_s
    assert rdiscr.astype('complex128') == cdiscr
    assert rdiscr.complex_space == cdiscr

    # Complex
    assert cdiscr.astype('complex64') == cdiscr_s
    assert cdiscr.astype('complex128') is cdiscr
    assert cdiscr.complex_space is cdiscr
    assert cdiscr.astype('float32') == rdiscr_s
    assert cdiscr.astype('float64') == rdiscr
    assert cdiscr.real_space == rdiscr

    # More exotic dtype
    # @leftaroundabout why was that even supported?
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype=bool, impl=impl, device=device)
    as_float = discr.astype(float)
    assert as_float.dtype_identifier == default_precision_dict[impl]['float']
    assert not as_float.is_weighted
    as_complex = discr.astype(complex)
    assert as_complex.dtype_identifier == 'complex128'
    assert not as_complex.is_weighted



def test_real_imag(odl_elem_order, odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    """Check if real and imaginary parts can be read and written to."""
    order = odl_elem_order
    tspace_cls = odl.space.entry_points.tensor_space_impl(impl)
    for dtype in COMPLEX_DTYPES:
        cdiscr = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype=dtype, impl=impl, device=device)
        rdiscr = cdiscr.real_space

        # Get real and imag
        x = cdiscr.element([[1 - 1j, 2 - 2j],
                            [3 - 3j, 4 - 4j]])
        assert x.real in rdiscr
        assert all_equal(x.real, [[1, 2],
                                  [3, 4]])
        assert x.imag in rdiscr
        assert all_equal(x.imag, [[-1, -2],
                                  [-3, -4]])

        # Set with different data types and shapes
        for assigntype in [ lambda x: x, tuple, rdiscr.element ]:


            # Using setters
            x = cdiscr.zero()
            new_real = assigntype([[2, 3],
                                 [4, 5]])
            x.real = new_real
            assert all_equal(x.real, [[2, 3],
                                      [4, 5]])

            x = cdiscr.zero()
            x.imag = assigntype([[4, 5],
                                 [6, 7]])
            assert all_equal(x.imag, [[4, 5],
                                      [6, 7]])

            # With [:] assignment
            x = cdiscr.zero()
            x.real[:] = assigntype([[2, 3],
                                    [4, 5]])
            assert all_equal(x.real, [[2, 3],
                                      [4, 5]])

            x = cdiscr.zero()
            x.imag[:] = assigntype([[2, 3],
                                    [4, 5]])
            assert all_equal(x.imag, [[2, 3],
                                      [4, 5]])

        # Setting with scalars
        x = cdiscr.zero()
        x.real = 1
        assert all_equal(x.real, [[1, 1],
                                  [1, 1]])

        x = cdiscr.zero()
        x.imag = -1
        assert all_equal(x.imag, [[-1, -1],
                                  [-1, -1]])

    # Incompatible shapes
    error = ValueError if impl =='numpy' else RuntimeError
    with pytest.raises(error):
        x.real = [4, 5, 6, 7]
    with pytest.raises(error):
        x.imag = [4, 5, 6, 7]


def test_reduction(odl_reduction, odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    name = odl_reduction
    space = odl.uniform_discr([0, 0], [1, 1], [2, 2], impl=impl, device=device)

    reduction = getattr(odl, name)
    backend_reduction = getattr(space.array_namespace, name)

    # Create some data
    x_arr, x = noise_elements(space, 1)
    arr_red = space.array_backend.to_cpu(backend_reduction(x_arr))
    odl_red = space.array_backend.to_cpu(reduction(x))
    assert arr_red == pytest.approx(odl_red)


def test_power(power, odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    space = odl.uniform_discr([0, 0], [1, 1], [2, 2], impl=impl, device=device)
    ns = space.array_namespace
    x_arr, x = noise_elements(space, 1)
    x_pos_arr = ns.abs(x_arr)
    x_neg_arr = -x_pos_arr
    x_pos = odl.abs(x)
    x_neg = -x_pos

    power_keyword = 'power' if impl == 'numpy' else 'pow'
    power_function = getattr(ns, power_keyword)

    if int(power) != power:
        # Make input positive to get real result
        for y in [x_pos_arr, x_neg_arr, x_pos, x_neg]:
            y += 0.1

    with np.errstate(invalid='ignore'):
        true_pos_pow = power_function(x_pos_arr, power)
        true_neg_pow = power_function(x_neg_arr, power)

    if int(power) != power and impl == 'cuda':
        with pytest.raises(ValueError):
            x_pos ** power
        with pytest.raises(ValueError):
            x_pos **= power
    else:
        with np.errstate(invalid='ignore'):
            assert all_almost_equal(x_pos ** power, true_pos_pow)
            if int(power) == power:
                assert all_almost_equal(x_neg ** power, true_neg_pow)

            x_pos **= power
            assert all_almost_equal(x_pos, true_pos_pow)

            if int(power) == power:
                x_neg **= power
                assert all_almost_equal(x_neg, true_neg_pow)


def test_inner_nonuniform(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    """Check if inner products are correct in non-uniform discretizations."""
    part = odl.nonuniform_partition([0, 2, 3, 5], min_pt=0, max_pt=5)
    backend = lookup_array_backend(impl)
    weights = backend.array_constructor(part.cell_sizes_vecs[0], device=device)
    tspace = odl.rn(part.size, weighting=weights, impl=impl, device=device)
    discr = odl.DiscretizedSpace(part, tspace)

    one = discr.one()
    linear = discr.element(lambda x: x)

    # Exact inner product is the integral from 0 to 5 of x, which is 5**2 / 2
    exact_inner = 5 ** 2 / 2.0
    inner = one.inner(linear)
    assert inner == pytest.approx(exact_inner)


def test_norm_nonuniform(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    """Check if norms are correct in non-uniform discretizations."""
    part = odl.nonuniform_partition([0, 2, 3, 5], min_pt=0, max_pt=5)

    backend = lookup_array_backend(impl)
    weights = backend.array_constructor(part.cell_sizes_vecs[0], device=device)
    
    tspace = odl.rn(part.size, weighting=weights, impl=impl, device=device)
    discr = odl.DiscretizedSpace(part, tspace)

    sqrt = discr.element(lambda x: np.sqrt(x))

    # Exact norm is the square root of the integral from 0 to 5 of x,
    # which is sqrt(5**2 / 2)
    exact_norm = np.sqrt(5 ** 2 / 2.0)
    norm = sqrt.norm()
    assert norm == pytest.approx(exact_norm)


def test_norm_interval(exponent, odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    # Test the function f(x) = x^2 on the interval (0, 1). Its
    # L^p-norm is (1 + 2*p)^(-1/p) for finite p and 1 for p=inf
    p = exponent
    discr = odl.uniform_discr(0, 1, 10, exponent=p, impl=impl, device=device)

    func = discr.element(lambda x: x ** 2)
    if p == float('inf'):
        assert func.norm() <= 1  # Max at boundary not hit
    else:
        true_norm = (1 + 2 * p) ** (-1 / p)
        assert func.norm() == pytest.approx(true_norm, rel=1e-2)


def test_norm_rectangle(exponent, odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    # Test the function f(x) = x_0^2 * x_1^3 on (0, 1) x (-1, 1). Its
    # L^p-norm is ((1 + 2*p) * (1 + 3 * p) / 2)^(-1/p) for finite p
    # and 1 for p=inf
    p = exponent
    discr = odl.uniform_discr([0, -1], [1, 1], (20, 30), exponent=p, impl=impl, device=device)

    func = discr.element(lambda x: x[0] ** 2 * x[1] ** 3)
    if p == float('inf'):
        assert func.norm() <= 1  # Max at boundary not hit
    else:
        true_norm = ((1 + 2 * p) * (1 + 3 * p) / 2) ** (-1 / p)
        assert func.norm() == pytest.approx(true_norm, rel=1e-2)


def test_norm_rectangle_boundary(exponent, odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    # Check the constant function 1 in different situations regarding the
    # placement of the outermost grid points.
    dtype = 'float32'

    # Standard case
    discr = odl.uniform_discr(
        [-1, -2], [1, 2], (4, 8), dtype=dtype, impl=impl, device=device, exponent=exponent
    )
    if exponent == float('inf'):
        assert discr.one().norm() == 1
    else:
        assert (
            discr.one().norm()
            == pytest.approx(discr.domain.volume ** (1 / exponent))
        )

    # Nodes on the boundary (everywhere)
    discr = odl.uniform_discr(
        [-1, -2], [1, 2], (4, 8), dtype=dtype, impl=impl, device=device, exponent=exponent,
        nodes_on_bdry=True
    )
    if exponent == float('inf'):
        assert discr.one().norm() == 1
    else:
        assert (
            discr.one().norm()
            == pytest.approx(discr.domain.volume ** (1 / exponent))
        )

    # Nodes on the boundary (selective)
    discr = odl.uniform_discr(
        [-1, -2], [1, 2], (4, 8), dtype=dtype, impl=impl, device=device, exponent=exponent,
        nodes_on_bdry=((False, True), False)
    )
    if exponent == float('inf'):
        assert discr.one().norm() == 1
    else:
        assert (
            discr.one().norm()
            == pytest.approx(discr.domain.volume ** (1 / exponent))
        )

    discr = odl.uniform_discr(
        [-1, -2], [1, 2], (4, 8), dtype=dtype, impl=impl, device=device, exponent=exponent,
        nodes_on_bdry=(False, (True, False))
    )
    if exponent == float('inf'):
        assert discr.one().norm() == 1
    else:
        assert (
            discr.one().norm()
            == pytest.approx(discr.domain.volume ** (1 / exponent))
        )

    # Completely arbitrary boundary
    part = odl.RectPartition(
        odl.IntervalProd([-1, -2], [1, 2]),
        odl.uniform_grid([0, 0], [1, 1], (4, 4))
    )
    weight = 1.0 if exponent == float('inf') else part.cell_volume
    tspace = odl.rn(part.shape, dtype=dtype, impl=impl,
                    exponent=exponent, weighting=weight, device=device)
    discr = DiscretizedSpace(part, tspace)

    if exponent == float('inf'):
        assert discr.one().norm() == 1
    else:
        assert (
            discr.one().norm()
            == pytest.approx(discr.domain.volume ** (1 / exponent))
        )


def test_uniform_discr_fromdiscr_one_attr(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    # Change 1 attribute
    discr = odl.uniform_discr([0, -1], [1, 1], [10, 5], impl=impl, device=device)
    # csides = [0.1, 0.4]

    # min_pt -> translate, keep cells
    new_min_pt = [3, 7]
    true_new_max_pt = [4, 9]

    new_discr = odl.uniform_discr_fromdiscr(discr, min_pt=new_min_pt)
    assert all_almost_equal(new_discr.min_pt, new_min_pt)
    assert all_almost_equal(new_discr.max_pt, true_new_max_pt)
    assert all_equal(new_discr.shape, discr.shape)
    assert all_almost_equal(new_discr.cell_sides, discr.cell_sides)

    # max_pt -> translate, keep cells
    new_max_pt = [3, 7]
    true_new_min_pt = [2, 5]

    new_discr = odl.uniform_discr_fromdiscr(discr, max_pt=new_max_pt)
    assert all_almost_equal(new_discr.min_pt, true_new_min_pt)
    assert all_almost_equal(new_discr.max_pt, new_max_pt)
    assert all_equal(new_discr.shape, discr.shape)
    assert all_almost_equal(new_discr.cell_sides, discr.cell_sides)

    # shape -> resize cells, keep corners
    new_shape = (5, 20)
    true_new_csides = [0.2, 0.1]
    new_discr = odl.uniform_discr_fromdiscr(discr, shape=new_shape)
    assert all_almost_equal(new_discr.min_pt, discr.min_pt)
    assert all_almost_equal(new_discr.max_pt, discr.max_pt)
    assert all_equal(new_discr.shape, new_shape)
    assert all_almost_equal(new_discr.cell_sides, true_new_csides)

    # cell_sides -> resize cells, keep corners
    new_csides = [0.5, 0.2]
    true_new_shape = (2, 10)
    new_discr = odl.uniform_discr_fromdiscr(discr, cell_sides=new_csides)
    assert all_almost_equal(new_discr.min_pt, discr.min_pt)
    assert all_almost_equal(new_discr.max_pt, discr.max_pt)
    assert all_equal(new_discr.shape, true_new_shape)
    assert all_almost_equal(new_discr.cell_sides, new_csides)


def test_uniform_discr_fromdiscr_two_attrs(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    # Change 2 attributes -> resize and translate

    discr = odl.uniform_discr([0, -1], [1, 1], [10, 5], impl=impl, device=device)
    # csides = [0.1, 0.4]

    new_min_pt = [-2, 1]
    new_max_pt = [4, 2]
    true_new_csides = [0.6, 0.2]
    new_discr = odl.uniform_discr_fromdiscr(discr, min_pt=new_min_pt,
                                            max_pt=new_max_pt)
    assert all_almost_equal(new_discr.min_pt, new_min_pt)
    assert all_almost_equal(new_discr.max_pt, new_max_pt)
    assert all_equal(new_discr.shape, discr.shape)
    assert all_almost_equal(new_discr.cell_sides, true_new_csides)

    new_min_pt = [-2, 1]
    new_shape = (5, 20)
    true_new_max_pt = [-1.5, 9]
    new_discr = odl.uniform_discr_fromdiscr(discr, min_pt=new_min_pt,
                                            shape=new_shape)
    assert all_almost_equal(new_discr.min_pt, new_min_pt)
    assert all_almost_equal(new_discr.max_pt, true_new_max_pt)
    assert all_equal(new_discr.shape, new_shape)
    assert all_almost_equal(new_discr.cell_sides, discr.cell_sides)

    new_min_pt = [-2, 1]
    new_csides = [0.6, 0.2]
    true_new_max_pt = [4, 2]
    new_discr = odl.uniform_discr_fromdiscr(discr, min_pt=new_min_pt,
                                            cell_sides=new_csides)
    assert all_almost_equal(new_discr.min_pt, new_min_pt)
    assert all_almost_equal(new_discr.max_pt, true_new_max_pt)
    assert all_equal(new_discr.shape, discr.shape)
    assert all_almost_equal(new_discr.cell_sides, new_csides)

    new_max_pt = [4, 2]
    new_shape = (5, 20)
    true_new_min_pt = [3.5, -6]
    new_discr = odl.uniform_discr_fromdiscr(discr, max_pt=new_max_pt,
                                            shape=new_shape)
    assert all_almost_equal(new_discr.min_pt, true_new_min_pt)
    assert all_almost_equal(new_discr.max_pt, new_max_pt)
    assert all_equal(new_discr.shape, new_shape)
    assert all_almost_equal(new_discr.cell_sides, discr.cell_sides)

    new_max_pt = [4, 2]
    new_csides = [0.6, 0.2]
    true_new_min_pt = [-2, 1]
    new_discr = odl.uniform_discr_fromdiscr(discr, max_pt=new_max_pt,
                                            cell_sides=new_csides)
    assert all_almost_equal(new_discr.min_pt, true_new_min_pt)
    assert all_almost_equal(new_discr.max_pt, new_max_pt)
    assert all_equal(new_discr.shape, discr.shape)
    assert all_almost_equal(new_discr.cell_sides, new_csides)


def test_uniform_discr_fromdiscr_per_axis(odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs

    discr = odl.uniform_discr([0, -1], [1, 1], [10, 5], impl=impl, device=device)
    # csides = [0.1, 0.4]

    new_min_pt = [-2, None]
    new_max_pt = [4, 2]
    new_shape = (None, 20)
    new_csides = [None, None]

    true_new_min_pt = [-2, -6]
    true_new_max_pt = [4, 2]
    true_new_shape = (10, 20)
    true_new_csides = [0.6, 0.4]

    new_discr = odl.uniform_discr_fromdiscr(
        discr, min_pt=new_min_pt, max_pt=new_max_pt,
        shape=new_shape, cell_sides=new_csides)

    assert all_almost_equal(new_discr.min_pt, true_new_min_pt)
    assert all_almost_equal(new_discr.max_pt, true_new_max_pt)
    assert all_equal(new_discr.shape, true_new_shape)
    assert all_almost_equal(new_discr.cell_sides, true_new_csides)

    new_min_pt = None
    new_max_pt = [None, 2]
    new_shape = (5, None)
    new_csides = [None, 0.2]

    true_new_min_pt = [0, 1]
    true_new_max_pt = [1, 2]
    true_new_shape = (5, 5)
    true_new_csides = [0.2, 0.2]

    new_discr = odl.uniform_discr_fromdiscr(
        discr, min_pt=new_min_pt, max_pt=new_max_pt,
        shape=new_shape, cell_sides=new_csides)

    assert all_almost_equal(new_discr.min_pt, true_new_min_pt)
    assert all_almost_equal(new_discr.max_pt, true_new_max_pt)
    assert all_equal(new_discr.shape, true_new_shape)
    assert all_almost_equal(new_discr.cell_sides, true_new_csides)


if __name__ == '__main__':
    odl.util.test_file(__file__)
