# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division
import numpy as np
import pytest
from pkg_resources import parse_version

import odl
from odl.discr.lp_discr import DiscreteLp, DiscreteLpElement
from odl.space.base_tensors import TensorSpace
from odl.space.npy_tensors import NumpyTensor
from odl.space.weighting import ConstWeighting
from odl.util import array_module, as_numpy
from odl.util.testutils import (
    all_equal, all_almost_equal, noise_elements, simple_fixture, xfail_if)


USE_ARRAY_UFUNCS_INTERFACE = (parse_version(np.__version__) >=
                              parse_version('1.13'))

# --- Pytest fixtures --- #


exponent = simple_fixture('exponent', [2.0, 1.0, float('inf'), 0.5, 1.5])
power = simple_fixture('power', [1.0, 2.0, 0.5, -0.5, -1.0, -2.0])
shape = simple_fixture('shape', [(2, 3, 4), (3, 4), (2,), (1,), (1, 1, 1)])
power = simple_fixture('power', [1.0, 2.0, 0.5, -0.5, -1.0, -2.0])


# --- DiscreteLp --- #


def test_discretelp_init():
    """Test initialization and basic properties of DiscreteLp."""
    # Real space
    fspace = odl.FunctionSpace(odl.IntervalProd([0, 0], [1, 1]))
    part = odl.uniform_partition_fromintv(fspace.domain, (2, 4))
    tspace = odl.rn(part.shape)

    discr = DiscreteLp(fspace, part, tspace)
    assert discr.fspace == fspace
    assert discr.tspace == tspace
    assert discr.partition == part
    assert discr.interp == 'nearest'
    assert discr.interp_byaxis == ('nearest', 'nearest')
    assert discr.exponent == tspace.exponent
    assert discr.axis_labels == ('$x$', '$y$')
    assert discr.is_real

    discr = DiscreteLp(fspace, part, tspace, interp='linear')
    assert discr.interp == 'linear'
    assert discr.interp_byaxis == ('linear', 'linear')

    discr = DiscreteLp(fspace, part, tspace, interp=['nearest', 'linear'])
    assert discr.interp == ('nearest', 'linear')
    assert discr.interp_byaxis == ('nearest', 'linear')

    # Complex space
    fspace_c = odl.FunctionSpace(odl.IntervalProd([0, 0], [1, 1]),
                                 out_dtype=complex)
    tspace_c = odl.cn(part.shape)
    discr = DiscreteLp(fspace_c, part, tspace_c)
    assert discr.is_complex

    # Make sure repr shows something
    assert repr(discr)

    # Error scenarios
    with pytest.raises(ValueError):
        DiscreteLp(fspace, part, tspace_c)  # mixes real & complex

    with pytest.raises(ValueError):
        DiscreteLp(fspace_c, part, tspace)  # mixes complex & real

    part_1d = odl.uniform_partition(0, 1, 2)
    with pytest.raises(ValueError):
        DiscreteLp(fspace, part_1d, tspace)  # wrong dimensionality

    part_diffshp = odl.uniform_partition_fromintv(fspace.domain, (3, 4))
    with pytest.raises(ValueError):
        DiscreteLp(fspace, part_diffshp, tspace)  # shape mismatch


def test_empty():
    """Check if empty spaces behave as expected and all methods work."""
    discr = odl.uniform_discr([], [], ())

    assert discr.interp == 'nearest'
    assert discr.axis_labels == ()
    assert discr.tangent_bundle == odl.ProductSpace(field=odl.RealNumbers())
    assert discr.complex_space == odl.uniform_discr([], [], (), dtype=complex)
    hash(discr)
    assert repr(discr) != ''

    elem = discr.element(1.0)
    assert np.array_equal(elem.asarray(), 1.0)
    assert np.array_equal(elem.real, 1.0)
    assert np.array_equal(elem.imag, 0.0)
    assert np.array_equal(elem.conj(), 1.0)


# --- uniform_discr --- #


def test_uniform_discr_dtypes(odl_tspace_impl):
    """Test basic properties of spaces created by uniform_discr."""
    impl = odl_tspace_impl
    real_float_dtypes = [np.float32, np.float64]
    nonfloat_dtypes = [np.int8, np.int16, np.int32, np.int64,
                       np.uint8, np.uint16, np.uint32, np.uint64]
    complex_float_dtypes = [np.complex64, np.complex128]

    for dtype in real_float_dtypes:
        try:
            discr = odl.uniform_discr(0, 1, 10, impl=impl, dtype=dtype)
        except TypeError:
            continue
        else:
            assert isinstance(discr.tspace, TensorSpace)
            assert discr.tspace.impl == impl
            assert discr.is_real

    for dtype in nonfloat_dtypes:
        try:
            discr = odl.uniform_discr(0, 1, 10, impl=impl, dtype=dtype)
        except TypeError:
            continue
        else:
            assert isinstance(discr.tspace, TensorSpace)
            assert discr.tspace.impl == impl
            assert discr.tspace.element().space.dtype == dtype

    for dtype in complex_float_dtypes:
        try:
            discr = odl.uniform_discr(0, 1, 10, impl=impl, dtype=dtype)
        except TypeError:
            continue
        else:
            assert isinstance(discr.tspace, TensorSpace)
            assert discr.tspace.impl == impl
            assert discr.is_complex
            assert discr.tspace.element().space.dtype == dtype


def test_uniform_discr_init_real(odl_tspace_impl):
    """Test initialization and basic properties with uniform_discr, real."""
    impl = odl_tspace_impl

    # 1D
    discr = odl.uniform_discr(0, 1, 10, impl=impl)
    assert isinstance(discr, DiscreteLp)
    assert isinstance(discr.tspace, TensorSpace)
    assert discr.impl == impl
    assert discr.is_real
    assert discr.tspace.exponent == 2.0
    assert discr.dtype == discr.tspace.default_dtype(odl.RealNumbers())
    assert discr.is_real
    assert not discr.is_complex
    assert all_equal(discr.min_pt, [0])
    assert all_equal(discr.max_pt, [1])
    assert discr.shape == (10,)
    assert repr(discr)

    discr = odl.uniform_discr(0, 1, 10, impl=impl, exponent=1.0)
    assert discr.exponent == 1.0

    discr = odl.uniform_discr(0, 1, 10, impl=impl, interp='linear')
    assert discr.interp == 'linear'

    # 2D
    discr = odl.uniform_discr([0, 0], [1, 1], (5, 5))
    assert all_equal(discr.min_pt, np.array([0, 0]))
    assert all_equal(discr.max_pt, np.array([1, 1]))
    assert discr.shape == (5, 5)

    # nd
    discr = odl.uniform_discr([0] * 10, [1] * 10, (5,) * 10)
    assert all_equal(discr.min_pt, np.zeros(10))
    assert all_equal(discr.max_pt, np.ones(10))
    assert discr.shape == (5,) * 10


def test_uniform_discr_init_complex(odl_tspace_impl):
    """Test initialization and basic properties with uniform_discr, complex."""
    impl = odl_tspace_impl
    if impl != 'numpy':
        pytest.xfail(reason='complex dtypes not supported')

    discr = odl.uniform_discr(0, 1, 10, dtype='complex', impl=impl)
    assert discr.is_complex
    assert discr.dtype == discr.tspace.default_dtype(odl.ComplexNumbers())


# --- DiscreteLp methods --- #


def test_discretelp_element(tspace_impl):
    """Test creation and membership of DiscreteLp elements."""
    # Creation from scratch
    # 1D
    discr = odl.uniform_discr(0, 1, 3, impl=tspace_impl)
    weight = 1.0 if exponent == float('inf') else discr.cell_volume
    tspace = odl.rn(3, weighting=weight, impl=tspace_impl)
    elem = discr.element()
    assert elem in discr
    assert elem.tensor in tspace

    # 2D
    discr = odl.uniform_discr([0, 0], [1, 1], (3, 3), impl=tspace_impl)
    weight = 1.0 if exponent == float('inf') else discr.cell_volume
    tspace = odl.rn((3, 3), weighting=weight, impl=tspace_impl)
    elem = discr.element()
    assert elem in discr
    assert elem.tensor in tspace


def test_discretelp_element_from_array(tspace_impl):
    """Test creation of DiscreteLp elements from arrays."""
    # 1D
    discr = odl.uniform_discr(0, 1, 3, impl=tspace_impl)
    elem = discr.element([1, 2, 3])
    assert np.array_equal(elem.tensor, [1, 2, 3])
    assert isinstance(elem, DiscreteLpElement)
    assert all_equal(elem.tensor, [1, 2, 3])


def test_element_from_array_2d(odl_tspace_impl, odl_elem_order):
    """Test element in 2d with different orderings."""
    impl = odl_tspace_impl
    order = odl_elem_order
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 2])
    elem = discr.element([[1, 2],
                          [3, 4]], order=order)

    assert isinstance(elem, DiscreteLpElement)
    assert isinstance(elem.tensor, NumpyTensor)
    assert all_equal(elem, [[1, 2],
                            [3, 4]])

    if order is None:
        assert elem.tensor.data.flags[discr.default_order + '_CONTIGUOUS']
    else:
        assert elem.tensor.data.flags[order + '_CONTIGUOUS']

    with pytest.raises(ValueError):
        discr.element([1, 2, 3])  # wrong size & shape
    with pytest.raises(ValueError):
        discr.element([1, 2, 3, 4])  # wrong shape
    with pytest.raises(ValueError):
        discr.element([[1],
                       [2],
                       [3],
                       [4]])  # wrong shape


def test_element_from_function_1d(tspace_impl):
    """Test creation of DiscreteLp elements from functions in 1 dimension."""
    space = odl.uniform_discr(-1, 1, 4, impl=tspace_impl)
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
    true_elem = [1.0 for x in points]
    assert all_equal(elem_lam, true_elem)

    # Non vectorized
    elem_lam = space.element(lambda x: x[0], vectorized=False)
    assert all_equal(elem_lam, points)


def test_element_from_function_2d(tspace_impl):
    """Test creation of DiscreteLp elements from functions in 2 dimensions."""
    space = odl.uniform_discr([-1, -1], [1, 1], (2, 3), impl=tspace_impl)
    points = space.points()

    # Without parameter
    def f(x):
        return x[0] ** 2 + np.maximum(x[1], 0)

    elem_f = space.element(f)
    true_elem = np.reshape([x[0] ** 2 + max(x[1], 0) for x in points],
                           space.shape)
    assert all_equal(elem_f, true_elem)

    # With parameter
    def f(x, **kwargs):
        c = kwargs.pop('c', 0)
        return x[0] ** 2 + np.maximum(x[1], c)

    elem_f_default = space.element(f)
    true_elem = np.reshape([x[0] ** 2 + max(x[1], 0) for x in points],
                           space.shape)
    assert all_equal(elem_f_default, true_elem)

    elem_f_2 = space.element(f, c=1)
    true_elem = np.reshape([x[0] ** 2 + max(x[1], 1) for x in points],
                           space.shape)
    assert all_equal(elem_f_2, true_elem)

    # Using a lambda
    elem_lam = space.element(lambda x: x[0] - x[1])
    true_elem = np.reshape([x[0] - x[1] for x in points],
                           space.shape)
    assert all_equal(elem_lam, true_elem)

    # Using broadcasting
    elem_lam = space.element(lambda x: x[0])
    true_elem = np.reshape([x[0] for x in points],
                           space.shape)
    assert all_equal(elem_lam, true_elem)

    elem_lam = space.element(lambda x: x[1])
    true_elem = np.reshape([x[1] for x in points],
                           space.shape)
    assert all_equal(elem_lam, true_elem)

    # Broadcast from constant function
    elem_lam = space.element(lambda x: 1.0)
    true_elem = np.reshape([1.0 for x in points],
                           space.shape)
    assert all_equal(elem_lam, true_elem)

    # Non vectorized
    elem_lam = space.element(lambda x: x[0] + x[1], vectorized=False)
    true_elem = np.reshape([x[0] + x[1] for x in points],
                           space.shape)
    assert all_equal(elem_lam, true_elem)


def test_discretelp_zero_one(tspace_impl):
    """Test the zero and one element creators of DiscreteLp."""
    discr = odl.uniform_discr(0, 1, 3, impl=tspace_impl)

    zero = discr.zero()
    assert zero in discr
    assert np.array_equal(zero, [0, 0, 0])

    one = discr.one()
    assert one in discr
    assert np.array_equal(one, [1, 1, 1])


def test_equals_space(exponent, odl_tspace_impl):
    impl = odl_tspace_impl
    x1 = odl.uniform_discr(0, 1, 3, exponent=exponent, impl=impl)
    x2 = odl.uniform_discr(0, 1, 3, exponent=exponent, impl=impl)
    y = odl.uniform_discr(0, 1, 4, exponent=exponent, impl=impl)

    assert x1 is x1
    assert x1 is not x2
    assert x1 is not y
    assert x1 == x1
    assert x1 == x2
    assert x1 != y
    assert hash(x1) == hash(x2)
    assert hash(x1) != hash(y)


def test_equals_vec(exponent, odl_tspace_impl):
    impl = odl_tspace_impl
    discr = odl.uniform_discr(0, 1, 3, exponent=exponent, impl=impl)
    discr2 = odl.uniform_discr(0, 1, 4, exponent=exponent, impl=impl)
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


def test_operators(odl_tspace_impl):
    impl = odl_tspace_impl
    # Test of all operator overloads against the corresponding NumPy
    # implementation
    discr = odl.uniform_discr(0, 1, 10, impl=impl)

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


def test_interp():
    discr = odl.uniform_discr(0, 1, 3, interp='nearest')
    assert isinstance(discr.interpolation, odl.NearestInterpolation)

    discr = odl.uniform_discr(0, 1, 3, interp='linear')
    assert isinstance(discr.interpolation, odl.LinearInterpolation)

    discr = odl.uniform_discr([0, 0], [1, 1], (3, 3),
                              interp=['nearest', 'linear'])
    assert isinstance(discr.interpolation, odl.PerAxisInterpolation)

    with pytest.raises(ValueError):
        # Too many entries in interp
        discr = odl.uniform_discr(0, 1, 3, interp=['nearest', 'linear'])

    with pytest.raises(ValueError):
        # Too few entries in interp
        discr = odl.uniform_discr([0] * 3, [1] * 3, (3,) * 3,
                                  interp=['nearest', 'linear'])


def test_getitem():
    discr = odl.uniform_discr(0, 1, 3)
    elem = discr.element([1, 2, 3])

    assert all_equal(elem, [1, 2, 3])


def test_getslice():
    discr = odl.uniform_discr(0, 1, 3)
    elem = discr.element([1, 2, 3])

    assert isinstance(elem[:], NumpyTensor)
    assert all_equal(elem[:], [1, 2, 3])

    discr = odl.uniform_discr(0, 1, 3, dtype='complex')
    elem = discr.element([1 + 2j, 2 - 2j, 3])

    assert isinstance(elem[:], NumpyTensor)
    assert all_equal(elem[:], [1 + 2j, 2 - 2j, 3])


def test_setitem():
    discr = odl.uniform_discr(0, 1, 3)
    elem = discr.element([1, 2, 3])
    elem[0] = 4
    elem[1] = 5
    elem[2] = 6

    assert all_equal(elem, [4, 5, 6])


def test_setitem_nd():

    # 1D
    discr = odl.uniform_discr(0, 1, 3)
    elem = discr.element([1, 2, 3])

    elem[:] = [4, 5, 6]
    assert all_equal(elem, [4, 5, 6])

    elem[:] = np.array([3, 2, 1])
    assert all_equal(elem, [3, 2, 1])

    elem[:] = 0
    assert all_equal(elem, [0, 0, 0])

    elem[:] = [1]
    assert all_equal(elem, [1, 1, 1])

    with pytest.raises(ValueError):
        elem[:] = [0, 0]  # bad shape

    with pytest.raises(ValueError):
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

    arr = np.arange(6, 12).reshape([3, 2])
    elem[:] = arr
    assert all_equal(elem, arr)

    elem[:] = 0
    assert all_equal(elem, np.zeros(elem.shape))

    elem[:] = [1]
    assert all_equal(elem, np.ones(elem.shape))

    elem[:] = [0, 0]  # broadcasting assignment
    assert all_equal(elem, np.zeros(elem.shape))

    with pytest.raises(ValueError):
        elem[:] = [0, 0, 0]  # bad shape

    with pytest.raises(ValueError):
        elem[:] = np.arange(6)  # bad shape (6,)

    with pytest.raises(ValueError):
        elem[:] = np.ones((2, 3))[..., np.newaxis]  # bad shape (2, 3, 1)

    with pytest.raises(ValueError):
        arr = np.arange(6, 12).reshape([3, 2])
        elem[:] = arr.T  # bad shape (2, 3)

    # nD
    shape = (3,) * 3 + (4,) * 3
    discr = odl.uniform_discr([0] * 6, [1] * 6, shape)
    size = np.prod(shape)
    elem = discr.element(np.zeros(shape))

    arr = np.arange(size).reshape(shape)

    elem[:] = arr
    assert all_equal(elem, arr)

    elem[:] = 0
    assert all_equal(elem, np.zeros(elem.shape))

    elem[:] = [1]
    assert all_equal(elem, np.ones(elem.shape))

    with pytest.raises(ValueError):
        # Reversed shape -> bad
        elem[:] = np.arange(size).reshape((4,) * 3 + (3,) * 3)


def test_setslice():
    discr = odl.uniform_discr(0, 1, 3)
    elem = discr.element([1, 2, 3])

    elem[:] = [4, 5, 6]
    assert all_equal(elem, [4, 5, 6])


def test_asarray_2d(odl_elem_order):
    """Test the asarray method."""
    order = odl_elem_order
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 2])
    elem = discr.element([[1, 2],
                          [3, 4]], order=order)

    arr = elem.asarray()
    assert all_equal(arr, [[1, 2],
                           [3, 4]])
    if order is None:
        assert arr.flags[discr.default_order + '_CONTIGUOUS']
    else:
        assert arr.flags[order + '_CONTIGUOUS']

    # test out parameter
    out_c = np.empty([2, 2], order='C')
    result_c = elem.asarray(out=out_c)
    assert result_c is out_c
    assert all_equal(out_c, [[1, 2],
                             [3, 4]])
    out_f = np.empty([2, 2], order='F')
    result_f = elem.asarray(out=out_f)
    assert result_f is out_f
    assert all_equal(out_f, [[1, 2],
                             [3, 4]])

    # Try wrong shape
    out_wrong_shape = np.empty([2, 3])
    with pytest.raises(ValueError):
        elem.asarray(out=out_wrong_shape)


def test_transpose():
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 2])
    x = discr.element([[1, 2], [3, 4]])
    y = discr.element([[5, 6], [7, 8]])

    assert isinstance(x.T, odl.Operator)
    assert x.T.is_linear

    assert x.T(y) == x.inner(y)
    assert x.T.T == x
    assert all_equal(x.T.adjoint(1.0), x)


def test_cell_sides():
    # Non-degenerated case, should be same as cell size
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 2])
    elem = discr.element()

    assert all_equal(discr.cell_sides, [0.5] * 2)
    assert all_equal(elem.cell_sides, [0.5] * 2)

    # Degenerated case, uses interval size in 1-point dimensions
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 1])
    elem = discr.element()

    assert all_equal(discr.cell_sides, [0.5, 1])
    assert all_equal(elem.cell_sides, [0.5, 1])


def test_cell_volume():
    # Non-degenerated case
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 2])
    elem = discr.element()

    assert discr.cell_volume == 0.25
    assert elem.cell_volume == 0.25

    # Degenerated case, uses interval size in 1-point dimensions
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 1])
    elem = discr.element()

    assert discr.cell_volume == 0.5
    assert elem.cell_volume == 0.5


def test_astype(tspace_impl):

    rdiscr = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype='float64',
                               impl=tspace_impl)
    cdiscr = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype='complex128',
                               impl=tspace_impl)
    rdiscr_s = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype='float32',
                                 impl=tspace_impl)
    cdiscr_s = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype='complex64',
                                 impl=tspace_impl)

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
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype=bool)
    as_float = discr.astype(float)
    assert as_float.dtype == float
    assert not as_float.is_weighted
    as_complex = discr.astype(complex)
    assert as_complex.dtype == complex
    assert not as_complex.is_weighted


def testodl_ufuncs(odl_tspace_impl, odl_ufunc):
    """Test ufuncs in ``x.ufuncs`` against direct Numpy ufuncs."""
    impl = odl_tspace_impl
    space = odl.uniform_discr([0, 0], [1, 1], (2, 3), impl=impl)
    name = odl_ufunc

    # Get the ufunc from numpy as reference, plus some additional info
    ufunc_npy = getattr(np, name)
    nin = ufunc_npy.nin
    nout = ufunc_npy.nout

    # Disable Numpy warnings for the time being
    npy_err_orig = np.geterr()
    np.seterr(all='ignore')

    def _check_result_type(result, expected_type):
        if nout == 1:
            assert isinstance(result, space.element_type)
        elif nout > 1:
            for i in range(nout):
                assert isinstance(result[i], space.element_type)
        else:
            assert False

    def _check_result_is_out(result, out_seq):
        if nout == 1:
            assert result is out_seq[0]
        elif nout > 1:
            for i in range(nout):
                assert result[i] is out_seq[i]
        else:
            assert False

    # See https://github.com/cupy/cupy/issues/794
    cupy_ufuncs_broken_complex = [
        'expm1', 'floor_divide', 'fmin', 'fmax',
        'greater', 'greater_equal', 'less', 'less_equal', 'log1p', 'log2',
        'logical_and', 'logical_or', 'logical_not', 'logical_xor', 'minimum',
        'maximum', 'rint', 'sign', 'square']
    if (space.impl == 'cupy' and
            space.dtype.kind == 'c' and
            ufunc in cupy_ufuncs_broken_complex):
        pytest.xfail('ufunc {} broken for complex input in cupy'.format(ufunc))

    # Create some data
    arrays, elements = noise_elements(space, nin + nout)
    # Arrays of the space's own data storage type
    in_arrays_own = arrays[:nin]
    in_arrays_npy = [as_numpy(arr) for arr in arrays[:nin]]
    data_elem = elements[0]

    # Get function to call, using both interfaces:
    # - vec.ufunc(*other_args)
    # - np.ufunc(vec, *other_args)
    ufunc_method = getattr(data_elem.ufuncs, name)
    in_elems_method = elements[1:nin]
    in_elems_npy = elements[:nin]

    # If Numpy fails, mark the test as xfail (same below)
    try:
        result_npy = ufunc_npy(*in_arrays_npy)
    except TypeError:
        pytest.xfail('numpy ufunc not valid for inputs')

    # Out-of-place -- in = elements -- ufunc = method or numpy
    result = ufunc_method(*in_elems_method)
    assert all_almost_equal(result_npy, result)
    _check_result_type(result, space.element_type)

    # Get element(s) in the right space for in-place later
    if nout == 1:
        out_elems = [result.space.element()]
    elif nout == 2:
        out_elems = [res.space.element() for res in result]
    else:
        assert False

    result = ufunc_npy(*in_elems_npy)
    assert all_almost_equal(result_npy, result)
    _check_result_type(result, space.element_type)

    # Out-of-place -- in = numpy or own arrays -- ufunc = method
    result = ufunc_method(*in_arrays_npy[1:])
    assert all_almost_equal(result_npy, result)
    _check_result_type(result, space.element_type)

    result = ufunc_method(*in_arrays_own[1:])
    assert all_almost_equal(result_npy, result)
    _check_result_type(result, space.element_type)

    # In-place -- in = elements -- out = elements -- ufunc = method or numpy
    if nout == 1:
        kwargs_out = {'out': out_elems[0]}
    elif nout == 2:
        kwargs_out = {'out1': out_elems[0], 'out2': out_elems[1]}

    result = ufunc_method(*in_elems_method, **kwargs_out)
    assert all_almost_equal(result_npy, result)
    _check_result_is_out(result, out_elems[:nout])
    if USE_ARRAY_UFUNCS_INTERFACE:
        # Custom objects not allowed as `out` for numpy < 1.13

        if nout == 1:
            kwargs_out = {'out': out_elems[0]}
        elif nout == 2:
            kwargs_out = {'out': (out_elems[0], out_elems[1])}

        result = ufunc_npy(*in_elems_npy, **kwargs_out)
        assert all_almost_equal(result_npy, result)
        _check_result_is_out(result, out_elems[:nout])

    # In-place -- in = elements -- out = numpy or own arrays -- ufunc = numpy
    # This case is only supported with the new interface
    if USE_ARRAY_UFUNCS_INTERFACE:
        # Fresh arrays for output
        out_arrays_npy = [np.empty_like(as_numpy(arr))
                          for arr in arrays[nin:]]
        out_arrays_own = [array_module(space.impl).empty_like(arr)
                          for arr in arrays[nin:]]
        if nout == 1:
            kwargs_npy = {'out': out_arrays_npy[0]}
            kwargs_own = {'out': out_arrays_own[0]}
        elif nout > 1:
            kwargs_npy = {'out': out_arrays_npy[:nout]}
            kwargs_own = {'out': out_arrays_own[:nout]}

        try:
            result_out_npy = ufunc_npy(*in_elems_npy, **kwargs_npy)
        except TypeError:
            pytest.xfail('numpy ufunc not valid for inputs')

        result_out_own = ufunc_npy(*in_elems_npy, **kwargs_own)
        assert all_almost_equal(result_out_npy, result_npy)
        assert all_almost_equal(result_out_own, result_npy)
        _check_result_is_out(result_out_npy, out_arrays_npy)
        _check_result_is_out(result_out_own, out_arrays_own)

    if USE_ARRAY_UFUNCS_INTERFACE:
        # Check `ufunc.at`
        indices = [[0, 0, 1],
                   [0, 1, 2]]

        mod_array = in_arrays_npy[0].copy()
        mod_elem = in_elems_npy[0].copy()
        if nin == 1:
            try:
                result_npy = ufunc_npy.at(mod_array, indices)
            except TypeError:
                pytest.xfail('numpy ufunc.at not valid for inputs')

            result = ufunc_npy.at(mod_elem, indices)

        elif nin == 2:
            other_array = in_arrays_npy[1][indices]
            other_elem = in_elems_npy[1][indices]
            try:
                result_npy = ufunc_npy.at(mod_array, indices, other_array)
            except TypeError:
                pytest.xfail('numpy ufunc.at not valid for inputs')

            result = ufunc_npy.at(mod_elem, indices, other_elem)

        assert all_almost_equal(result, result_npy)

    # Check `ufunc.reduce`
    if nin == 2 and nout == 1 and USE_ARRAY_UFUNCS_INTERFACE:
        in_array = in_arrays_npy[0]
        in_elem = in_elems_npy[0]

        # We only test along one axis since some binary ufuncs are not
        # re-orderable, in which case Numpy raises a ValueError

        try:
            result_npy = ufunc_npy.reduce(in_array)
        except TypeError:
            pytest.xfail('numpy ufunc.reduce not valid for inputs')

        # Out-of-place -- in = element
        result = ufunc_npy.reduce(in_elem)
        assert all_almost_equal(result, result_npy)
        # keepdims not allowed
        with pytest.raises(ValueError):
            ufunc_npy.reduce(in_elem, keepdims=True)

        # Using a specific dtype
        try:
            result_npy = ufunc_npy.reduce(in_array, dtype=complex)
        except TypeError:
            pytest.xfail('numpy ufunc.reduce not valid for complex dtype')

        with xfail_if(space.impl == 'cupy',
                      reason='complex reduce is broken in cupy'):
            result = ufunc_npy.reduce(in_elem, dtype=complex)
            assert result.dtype == result_npy.dtype
            assert all_almost_equal(result, result_npy)

    # Other ufunc method use the same interface, to we don't perform
    # extra tests for them.

    # Reset Numpy err handling
    np.seterr(**npy_err_orig)


def test_ufunc_corner_cases(odl_tspace_impl):
    """Check if some corner cases are handled correctly."""
    impl = odl_tspace_impl
    space = odl.uniform_discr([0, 0], [1, 1], (2, 3), impl=impl)
    x = space.element([[-1, 0, 1],
                       [1, 2, 3]])
    space_no_w = odl.uniform_discr([0, 0], [1, 1], (2, 3), impl=impl,
                                   weighting=1.0)

    # --- Ufuncs with nin = 1, nout = 1 --- #

    with pytest.raises(ValueError):
        # Too many arguments
        x.__array_ufunc__(np.sin, '__call__', x, np.ones((2, 3)))

    # Check that `out=(None,)` is the same as not providing `out`
    res = x.__array_ufunc__(np.sin, '__call__', x, out=(None,))
    assert all_almost_equal(res, np.sin(x.asarray()))
    # Check that the result space is the same
    assert res.space == space

    # Check usage of `order` argument
    with xfail_if(tspace_impl == 'cupy',
                  reason='cupy does not accept `order` in ufuncs'):
        for order in ('C', 'F'):
            res = x.__array_ufunc__(np.sin, '__call__', x, order=order)
            assert all_almost_equal(res, np.sin(x.asarray()))
            assert res.tensor.data.flags[order + '_CONTIGUOUS']

    # Check usage of `dtype` argument
    res = x.__array_ufunc__(np.sin, '__call__', x, dtype=complex)
    assert all_almost_equal(res, np.sin(x.asarray(), dtype=complex))
    assert res.dtype == complex

    # Check propagation of weightings
    y = space_no_w.one()
    res = y.__array_ufunc__(np.sin, '__call__', y)
    assert res.space.weighting == space_no_w.weighting
    y = space_no_w.one()
    res = y.__array_ufunc__(np.sin, '__call__', y)
    assert res.space.weighting == space_no_w.weighting

    # --- Ufuncs with nin = 2, nout = 1 --- #

    with pytest.raises(ValueError):
        # Too few arguments
        x.__array_ufunc__(np.add, '__call__', x)

    with pytest.raises(ValueError):
        # Too many outputs
        out1, out2 = np.empty_like(x), np.empty_like(x)
        x.__array_ufunc__(np.add, '__call__', x, x, out=(out1, out2))

    # Check that npy_array += odl_vector works
    arr = np.ones((2, 3))
    arr += x
    assert all_almost_equal(arr, x.asarray() + 1)
    # For Numpy >= 1.13, this will be equivalent
    arr = np.ones((2, 3))
    res = x.__array_ufunc__(np.add, '__call__', arr, x, out=(arr,))
    assert all_almost_equal(arr, x.asarray() + 1)
    assert res is arr

    # --- `accumulate` --- #

    res = x.__array_ufunc__(np.add, 'accumulate', x)
    assert all_almost_equal(res, np.add.accumulate(x.asarray()))
    assert res.space == space
    arr = np.empty_like(x)
    res = x.__array_ufunc__(np.add, 'accumulate', x, out=(arr,))
    assert all_almost_equal(arr, np.add.accumulate(x.asarray()))
    assert res is arr

    # `accumulate` with other dtype
    res = x.__array_ufunc__(np.add, 'accumulate', x, dtype='float32')
    assert res.dtype == 'float32'

    # Error scenarios
    with pytest.raises(ValueError):
        # Too many `out` arguments
        out1, out2 = np.empty_like(x), np.empty_like(x)
        x.__array_ufunc__(np.add, 'accumulate', x, out=(out1, out2))

    # --- `reduce` --- #

    res = x.__array_ufunc__(np.add, 'reduce', x)
    assert all_almost_equal(res, np.add.reduce(x.asarray()))

    with pytest.raises(ValueError):
        x.__array_ufunc__(np.add, 'reduce', x, keepdims=True)

    # With `out` argument and `axis`
    out_ax0 = np.empty(3)
    res = x.__array_ufunc__(np.add, 'reduce', x, axis=0, out=(out_ax0,))
    assert all_almost_equal(out_ax0, np.add.reduce(x.asarray(), axis=0))
    assert res is out_ax0
    out_ax1 = odl.rn(2).element()
    res = x.__array_ufunc__(np.add, 'reduce', x, axis=1, out=(out_ax1,))
    assert all_almost_equal(out_ax1, np.add.reduce(x.asarray(), axis=1))
    assert res is out_ax1

    # Addition is reorderable, so we can give multiple axes
    res = x.__array_ufunc__(np.add, 'reduce', x, axis=(0, 1))
    assert res == pytest.approx(np.add.reduce(x.asarray(), axis=(0, 1)))

    # Constant weighting should be preserved (recomputed from cell
    # volume)
    y = space.one()
    res = y.__array_ufunc__(np.add, 'reduce', y, axis=0)
    assert res.space.weighting.const == pytest.approx(space.cell_sides[1])

    # Check that `exponent` is propagated
    space_1 = odl.uniform_discr([0, 0], [1, 1], (2, 3), impl=impl,
                                exponent=1)
    z = space_1.one()
    res = z.__array_ufunc__(np.add, 'reduce', z, axis=0)
    assert res.space.exponent == 1

    # --- `outer` --- #

    # Check that weightings are propagated correctly
    x = y = space.one()
    res = x.__array_ufunc__(np.add, 'outer', x, y)
    assert isinstance(res.space.weighting, ConstWeighting)
    assert res.space.weighting.const == pytest.approx(x.space.weighting.const *
                                                      y.space.weighting.const)

    x = space.one()
    y = space_no_w.one()
    res = x.__array_ufunc__(np.add, 'outer', x, y)
    assert isinstance(res.space.weighting, ConstWeighting)
    assert res.space.weighting.const == pytest.approx(x.space.weighting.const)

    x = y = space_no_w.one()
    res = x.__array_ufunc__(np.add, 'outer', x, y)
    assert not res.space.is_weighted


def test_real_imag(odl_tspace_impl, odl_elem_order):
    """Check if real and imaginary parts can be read and written to."""
    impl = odl_tspace_impl
    order = odl_elem_order
    tspace_cls = odl.space.entry_points.tensor_space_impl(impl)
    for dtype in filter(odl.util.is_complex_floating_dtype,
                        tspace_cls.available_dtypes()):
        cdiscr = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype=dtype,
                                   impl=impl)
        rdiscr = cdiscr.real_space

        # Get real and imag
        x = cdiscr.element([[1 - 1j, 2 - 2j],
                            [3 - 3j, 4 - 4j]], order=order)
        assert x.real in rdiscr
        assert all_equal(x.real, [[1, 2],
                                  [3, 4]])
        assert x.imag in rdiscr
        assert all_equal(x.imag, [[-1, -2],
                                  [-3, -4]])

        # Set with different data types and shapes
        for assigntype in (lambda x: x, tuple, rdiscr.element):

            # Using setters
            x = cdiscr.zero()
            x.real = assigntype([[2, 3],
                                 [4, 5]])
            assert all_equal(x.real, [[2, 3],
                                      [4, 5]])

            x = cdiscr.zero()
            x.imag = assigntype([[4, 5],
                                 [6, 7]])
            assert all_equal(x.imag, [[4, 5],
                                      [6, 7]])

            # With [:] assignment
            x = cdiscr.zero()
            with xfail_if(tspace_impl == 'cupy',
                          reason='cupy x.real[:] does not mutate x'):
                x.real[:] = assigntype([[2, 3],
                                        [4, 5]])
                assert all_equal(x.real, [[2, 3],
                                          [4, 5]])

            x = cdiscr.zero()
            with xfail_if(tspace_impl == 'cupy',
                          reason='cupy x.imag[:] does not mutate x'):
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
    with pytest.raises(ValueError):
        x.real = [4, 5, 6, 7]
    with pytest.raises(ValueError):
        x.imag = [4, 5, 6, 7]


def test_reduction(odl_tspace_impl, odl_reduction):
    impl = odl_tspace_impl
    name = odl_reduction
    space = odl.uniform_discr([0, 0], [1, 1], [2, 2], impl=impl)

    reduction = getattr(np, name)

    # Create some data
    x_arr, x = noise_elements(space, 1)
    assert (reduction(as_numpy(x_arr)) ==
            pytest.approx(getattr(x.ufuncs, name)()))


def test_power(odl_tspace_impl, power):
    impl = odl_tspace_impl
    space = odl.uniform_discr([0, 0], [1, 1], [2, 2], impl=impl)

    x_arr, x = noise_elements(space, 1)
    x_pos_arr = array_module(tspace_impl).abs(x_arr)
    x_neg_arr = -x_pos_arr
    x_pos = x.ufuncs.absolute()
    x_neg = -x_pos

    if int(power) != power:
        # Make input positive to get real result
        for y in [x_pos_arr, x_neg_arr, x_pos, x_neg]:
            y += 0.1

    with np.errstate(invalid='ignore'):
        true_pos_pow = array_module(tspace_impl).power(x_pos_arr, power)
        true_neg_pow = array_module(tspace_impl).power(x_neg_arr, power)

    if int(power) != power and impl == 'cuda':
        with pytest.raises(ValueError):
            x_pos ** power
        with pytest.raises(ValueError):
            x_pos **= power
    else:
        with np.errstate(invalid='ignore'):
            assert all_almost_equal(x_pos ** power, true_pos_pow)
            assert all_almost_equal(x_neg ** power, true_neg_pow)

            x_pos **= power
            x_neg **= power
            assert all_almost_equal(x_pos, true_pos_pow)
            assert all_almost_equal(x_neg, true_neg_pow)


def test_inner_nonuniform():
    """Check if inner products are correct in non-uniform discretizations."""
    fspace = odl.FunctionSpace(odl.IntervalProd(0, 5))
    part = odl.nonuniform_partition([0, 2, 3, 5], min_pt=0, max_pt=5)
    weights = part.cell_sizes_vecs[0]
    tspace = odl.rn(part.size, weighting=weights)
    discr = odl.DiscreteLp(fspace, part, tspace)

    one = discr.one()
    linear = discr.element(lambda x: x)

    # Exact inner product is the integral from 0 to 5 of x, which is 5**2 / 2
    exact_inner = 5 ** 2 / 2.0
    inner = one.inner(linear)
    assert inner == pytest.approx(exact_inner)


def test_norm_nonuniform():
    """Check if norms are correct in non-uniform discretizations."""
    fspace = odl.FunctionSpace(odl.IntervalProd(0, 5))
    part = odl.nonuniform_partition([0, 2, 3, 5], min_pt=0, max_pt=5)
    weights = part.cell_sizes_vecs[0]
    tspace = odl.rn(part.size, weighting=weights)
    discr = odl.DiscreteLp(fspace, part, tspace)

    sqrt = discr.element(lambda x: np.sqrt(x))

    # Exact norm is the square root of the integral from 0 to 5 of x,
    # which is sqrt(5**2 / 2)
    exact_norm = np.sqrt(5 ** 2 / 2.0)
    norm = sqrt.norm()
    assert norm == pytest.approx(exact_norm)


def test_norm_interval(exponent):
    # Test the function f(x) = x^2 on the interval (0, 1). Its
    # L^p-norm is (1 + 2*p)^(-1/p) for finite p and 1 for p=inf
    p = exponent
    fspace = odl.FunctionSpace(odl.IntervalProd(0, 1))
    lpdiscr = odl.uniform_discr_fromspace(fspace, 10, exponent=p)

    testfunc = fspace.element(lambda x: x ** 2)
    discr_testfunc = lpdiscr.element(testfunc)

    if p == float('inf'):
        assert discr_testfunc.norm() <= 1  # Max at boundary not hit
    else:
        true_norm = (1 + 2 * p) ** (-1 / p)
        assert discr_testfunc.norm() == pytest.approx(true_norm, rel=1e-2)


def test_norm_rectangle(exponent):
    # Test the function f(x) = x_0^2 * x_1^3 on (0, 1) x (-1, 1). Its
    # L^p-norm is ((1 + 2*p) * (1 + 3 * p) / 2)^(-1/p) for finite p
    # and 1 for p=inf
    p = exponent
    fspace = odl.FunctionSpace(odl.IntervalProd([0, -1], [1, 1]))
    lpdiscr = odl.uniform_discr_fromspace(fspace, (20, 30), exponent=p)

    testfunc = fspace.element(lambda x: x[0] ** 2 * x[1] ** 3)
    discr_testfunc = lpdiscr.element(testfunc)

    if p == float('inf'):
        assert discr_testfunc.norm() <= 1  # Max at boundary not hit
    else:
        true_norm = ((1 + 2 * p) * (1 + 3 * p) / 2) ** (-1 / p)
        assert discr_testfunc.norm() == pytest.approx(true_norm, rel=1e-2)


def test_norm_rectangle_boundary(odl_tspace_impl, exponent):
    # Check the constant function 1 in different situations regarding the
    # placement of the outermost grid points.
    impl = odl_tspace_impl

    dtype = 'float32'
    rect = odl.IntervalProd([-1, -2], [1, 2])
    fspace = odl.FunctionSpace(rect, out_dtype=dtype)

    # Standard case
    discr = odl.uniform_discr_fromspace(fspace, (4, 8), impl=impl,
                                        exponent=exponent)
    if exponent == float('inf'):
        assert discr.one().norm() == 1
    else:
        assert (discr.one().norm() ==
                pytest.approx(rect.volume ** (1 / exponent)))

    # Nodes on the boundary (everywhere)
    discr = odl.uniform_discr_fromspace(
        fspace, (4, 8), exponent=exponent, impl=impl, nodes_on_bdry=True)

    if exponent == float('inf'):
        assert discr.one().norm() == 1
    else:
        assert (discr.one().norm() ==
                pytest.approx(rect.volume ** (1 / exponent)))

    # Nodes on the boundary (selective)
    discr = odl.uniform_discr_fromspace(
        fspace, (4, 8), exponent=exponent,
        impl=impl, nodes_on_bdry=((False, True), False))

    if exponent == float('inf'):
        assert discr.one().norm() == 1
    else:
        assert (discr.one().norm() ==
                pytest.approx(rect.volume ** (1 / exponent)))

    discr = odl.uniform_discr_fromspace(
        fspace, (4, 8), exponent=exponent,
        impl=impl, nodes_on_bdry=(False, (True, False)))

    if exponent == float('inf'):
        assert discr.one().norm() == 1
    else:
        assert (discr.one().norm() ==
                pytest.approx(rect.volume ** (1 / exponent)))

    # Completely arbitrary boundary
    grid = odl.uniform_grid([0, 0], [1, 1], (4, 4))
    part = odl.RectPartition(rect, grid)
    weight = 1.0 if exponent == float('inf') else part.cell_volume
    tspace = odl.rn(part.shape, dtype=dtype, impl=impl,
                    exponent=exponent, weighting=weight)
    discr = DiscreteLp(fspace, part, tspace)

    if exponent == float('inf'):
        assert discr.one().norm() == 1
    else:
        assert (discr.one().norm() ==
                pytest.approx(rect.volume ** (1 / exponent)))


def test_uniform_discr_fromdiscr_one_attr():
    # Change 1 attribute

    discr = odl.uniform_discr([0, -1], [1, 1], [10, 5])
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


def test_uniform_discr_fromdiscr_two_attrs():
    # Change 2 attributes -> resize and translate

    discr = odl.uniform_discr([0, -1], [1, 1], [10, 5])
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


def test_uniform_discr_fromdiscr_per_axis():

    discr = odl.uniform_discr([0, -1], [1, 1], [10, 5])
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
