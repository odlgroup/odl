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


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External module imports
import pytest
import numpy as np

# Internal
import odl
from odl.discr.lp_discr import DiscreteLp, DiscreteLpElement
from odl.space.base_tensors import TensorSpace
from odl.space.npy_tensors import NumpyTensor
from odl.util.testutils import (almost_equal, all_equal, all_almost_equal,
                                noise_elements, simple_fixture)


# --- Pytest fixtures --- #


exponent = simple_fixture('exponent', [2.0, 1.0, float('inf'), 0.5, 1.5])
shape = simple_fixture('shape', [(2, 3, 4), (3, 4), (2,), (1,), (1, 1, 1)])


# --- DiscreteLp --- #


def test_discretelp_init():
    """Test initialization and basic properties of DiscreteLp."""
    # Real space
    fspace = odl.FunctionSpace(odl.IntervalProd([0, 0], [1, 1]))
    part = odl.uniform_partition_fromintv(fspace.domain, (2, 4))
    dspace = odl.rn(part.shape)

    discr = DiscreteLp(fspace, part, dspace)
    assert discr.uspace == fspace
    assert discr.dspace == dspace
    assert discr.partition == part
    assert discr.interp == 'nearest'
    assert discr.interp_by_axis == ('nearest', 'nearest')
    assert discr.exponent == dspace.exponent
    assert discr.axis_labels == ('$x$', '$y$')
    assert discr.is_real_space

    discr = DiscreteLp(fspace, part, dspace, interp='linear')
    assert discr.interp == 'linear'
    assert discr.interp_by_axis == ('linear', 'linear')

    discr = DiscreteLp(fspace, part, dspace, interp=['nearest', 'linear'])
    assert discr.interp == ('nearest', 'linear')
    assert discr.interp_by_axis == ('nearest', 'linear')

    # Complex space
    fspace_c = odl.FunctionSpace(odl.IntervalProd([0, 0], [1, 1]),
                                 field=odl.ComplexNumbers())
    dspace_c = odl.cn(part.shape)
    discr = DiscreteLp(fspace_c, part, dspace_c)
    assert discr.is_complex_space

    # Make sure repr shows something
    assert repr(discr)

    # Error scenarios
    with pytest.raises(ValueError):
        DiscreteLp(fspace, part, dspace_c)  # mixes real & complex

    with pytest.raises(ValueError):
        DiscreteLp(fspace_c, part, dspace)  # mixes complex & real

    part_1d = odl.uniform_partition(0, 1, 2)
    with pytest.raises(ValueError):
        DiscreteLp(fspace, part_1d, dspace)  # wrong dimensionality

    part_diffshp = odl.uniform_partition_fromintv(fspace.domain, (3, 4))
    with pytest.raises(ValueError):
        DiscreteLp(fspace, part_diffshp, dspace)  # shape mismatch


# --- uniform_discr --- #


def test_uniform_discr_init_real(tspace_impl):
    """Test initialization and basic properties with uniform_discr, real."""
    # 1D
    discr = odl.uniform_discr(0, 1, 10, impl=tspace_impl)
    assert isinstance(discr, DiscreteLp)
    assert isinstance(discr.dspace, TensorSpace)
    assert discr.impl == tspace_impl
    assert discr.is_real_space
    assert discr.dspace.exponent == 2.0
    assert discr.dtype == discr.dspace.default_dtype(odl.RealNumbers())
    assert all_equal(discr.min_pt, np.array([0]))
    assert all_equal(discr.max_pt, np.array([1]))
    assert discr.shape == (10,)
    assert repr(discr)

    discr = odl.uniform_discr(0, 1, 10, impl=tspace_impl, exponent=1.0)
    assert discr.exponent == 1.0

    discr = odl.uniform_discr(0, 1, 10, impl=tspace_impl, interp='linear')
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


def test_uniform_discr_init_complex(tspace_impl):
    """Test initialization and basic properties with uniform_discr, complex."""
    if tspace_impl != 'numpy':
        pytest.xfail(reason='complex dtypes not supported')

    discr = odl.uniform_discr(0, 1, 10, dtype='complex', impl=tspace_impl)
    assert discr.is_complex_space
    assert discr.dtype == discr.dspace.default_dtype(odl.ComplexNumbers())


# --- DiscreteLp methods --- #


def test_discretelp_element():
    """Test creation and membership of DiscreteLp elements."""
    # Creation from scratch
    # 1D
    discr = odl.uniform_discr(0, 1, 3)
    weight = 1.0 if exponent == float('inf') else discr.cell_volume
    dspace = odl.rn(3, weighting=weight)
    elem = discr.element()
    assert elem in discr
    assert elem.tensor in dspace

    # 2D
    discr = odl.uniform_discr([0, 0], [1, 1], (3, 3))
    weight = 1.0 if exponent == float('inf') else discr.cell_volume
    dspace = odl.rn((3, 3), weighting=weight)
    elem = discr.element()
    assert elem in discr
    assert elem.tensor in dspace


def test_discretelp_element_from_array():
    """Test creation of DiscreteLp elements from arrays."""
    # 1D
    discr = odl.uniform_discr(0, 1, 3)
    elem = discr.element([1, 2, 3])
    assert np.array_equal(elem.tensor, [1, 2, 3])

    assert isinstance(elem, DiscreteLpElement)
    assert isinstance(elem.tensor, NumpyTensor)
    assert all_equal(elem.tensor, [1, 2, 3])


def test_element_from_array_2d():
    # assert orderings work properly with 2d
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 2], impl='numpy', order='C')
    elem = discr.element([[1, 2],
                         [3, 4]])

    assert isinstance(elem, DiscreteLpElement)
    assert isinstance(elem.tensor, NumpyTensor)

    # Check ordering
    assert all_equal(elem.tensor, [1, 2, 3, 4])

    # Linear creation works as well
    linear_elem = discr.element([1, 2, 3, 4])
    assert all_equal(linear_elem.tensor, [1, 2, 3, 4])

    # Fortran order
    discr = odl.uniform_discr([0, 0], [1, 1], (2, 2), impl='numpy', order='F')
    elem = discr.element([[1, 2],
                          [3, 4]])
    assert np.array_equal(elem.tensor, [[1, 2],
                                        [3, 4]])

    with pytest.raises(ValueError):
        discr.element([1, 2, 3])  # wrong shape


def test_discretelp_element_order():
    """Test axis ordering when creating element from flat array."""
    # 'C' ordering
    discr = odl.uniform_discr([0, 0], [1, 1], (2, 2), order='C')
    elem = discr.element([1, 2, 3, 4])
    assert np.array_equal(elem.tensor, [[1, 2],
                                        [3, 4]])

    # 'F' ordering
    discr = odl.uniform_discr([0, 0], [1, 1], (2, 2), order='F')
    assert discr.order == 'F'
    elem = discr.element([1, 2, 3, 4])
    assert np.array_equal(elem.tensor, [[1, 3],
                                        [2, 4]])


def test_discretelp_element_bcast():
    """Test broadcasting of input to the element method."""
    discr = odl.uniform_discr([0, 0], [1, 1], (2, 2), order='C')
    bcast_elem = discr.element([2])  # broadcast along both axes
    assert np.array_equal(bcast_elem, [[2, 2],
                                       [2, 2]])
    bcast_elem = discr.element([[1, 2]])  # along axis 0
    assert np.array_equal(bcast_elem, [[1, 2],
                                       [1, 2]])
    bcast_elem = discr.element([[1], [2]])  # along axis 1
    assert np.array_equal(bcast_elem, [[1, 1],
                                       [2, 2]])


def test_element_from_function_1d():
    """Test creation of DiscreteLp elements from functions in 1 dimension."""
    space = odl.uniform_discr(-1, 1, 4)
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


def test_element_from_function_2d():
    """Test creation of DiscreteLp elements from functions in 2 dimensions."""
    space = odl.uniform_discr([-1, -1], [1, 1], (2, 3))
    points = space.points()

    # Without parameter
    def f(x):
        return x[0] ** 2 + np.maximum(x[1], 0)

    elem_f = space.element(f)
    true_elem = [x[0] ** 2 + max(x[1], 0) for x in points]
    assert all_equal(elem_f, true_elem)

    # With parameter
    def f(x, **kwargs):
        c = kwargs.pop('c', 0)
        return x[0] ** 2 + np.maximum(x[1], c)

    elem_f_default = space.element(f)
    true_elem = [x[0] ** 2 + max(x[1], 0) for x in points]
    assert all_equal(elem_f_default, true_elem)

    elem_f_2 = space.element(f, c=1)
    true_elem = [x[0] ** 2 + max(x[1], 1) for x in points]
    assert all_equal(elem_f_2, true_elem)

    # Using a lambda
    elem_lam = space.element(lambda x: x[0] - x[1])
    true_elem = [x[0] - x[1] for x in points]
    assert all_equal(elem_lam, true_elem)

    # Using broadcasting
    elem_lam = space.element(lambda x: x[0])
    true_elem = [x[0] for x in points]
    assert all_equal(elem_lam, true_elem)

    elem_lam = space.element(lambda x: x[1])
    true_elem = [x[1] for x in points]
    assert all_equal(elem_lam, true_elem)

    # Broadcast from constant function
    elem_lam = space.element(lambda x: 1.0)
    true_elem = [1.0 for x in points]
    assert all_equal(elem_lam, true_elem)

    # Non vectorized
    elem_lam = space.element(lambda x: x[0] + x[1], vectorized=False)
    true_elem = [x[0] + x[1] for x in points]
    assert all_equal(elem_lam, true_elem)


def test_discretelp_zero_one():
    """Test the zero and one element creators of DiscreteLp."""
    discr = odl.uniform_discr(0, 1, 3)

    zero = discr.zero()
    assert zero in discr
    assert np.array_equal(zero, [0, 0, 0])

    one = discr.one()
    assert one in discr
    assert np.array_equal(one, [1, 1, 1])


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


def test_operators(tspace_impl):
    # Test of all operator overloads against the corresponding NumPy
    # implementation
    discr = odl.uniform_discr(0, 1, 10, impl=tspace_impl)

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

    assert isinstance(elem[:], odl.NumpyTensor)
    assert all_equal(elem[:], [1, 2, 3])

    discr = odl.uniform_discr(0, 1, 3, dtype='complex')
    elem = discr.element([1 + 2j, 2 - 2j, 3])

    assert isinstance(elem[:], odl.NumpyTensor)
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
    assert all_equal(elem, [-1, -2, -3, -4, -5, -6])

    arr = np.arange(6, 12).reshape([3, 2])
    elem[:] = arr
    assert all_equal(elem, np.arange(6, 12))

    elem[:] = 0
    assert all_equal(elem, [0] * 6)

    elem[:] = [1]
    assert all_equal(elem, [1] * 6)

    with pytest.raises(ValueError):
        elem[:] = [0, 0]  # bad shape

    with pytest.raises(ValueError):
        elem[:] = [0, 0, 0]  # bad shape

    with pytest.raises(ValueError):
        elem[:] = np.arange(6)[:, np.newaxis]  # bad shape (6, 1)

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
    assert all_equal(elem, np.arange(size))

    elem[:] = 0
    assert all_equal(elem, np.zeros(size))

    elem[:] = [1]
    assert all_equal(elem, np.ones(size))

    with pytest.raises(ValueError):
        # Reversed shape -> bad
        elem[:] = np.arange(size).reshape((4,) * 3 + (3,) * 3)


def test_setslice():
    discr = odl.uniform_discr(0, 1, 3)
    elem = discr.element([1, 2, 3])

    elem[:] = [4, 5, 6]
    assert all_equal(elem, [4, 5, 6])


def test_asarray_2d():
    discr_F = odl.uniform_discr([0, 0], [1, 1], [2, 2], order='F')
    elem_F = discr_F.element([[1, 2],
                             [3, 4]])

    # Verify that returned array equals input data
    assert all_equal(elem_F.asarray(), [[1, 2],
                                        [3, 4]])
    # Check order of out array
    assert elem_F.asarray().flags['F_CONTIGUOUS']

    # test out parameter
    out_F = np.asfortranarray(np.empty([2, 2]))
    result_F = elem_F.asarray(out=out_F)
    assert result_F is out_F
    assert all_equal(out_F, [[1, 2],
                             [3, 4]])

    # Try discontinuous
    out_F_wrong = np.asfortranarray(np.empty([2, 2]))[::2, :]
    with pytest.raises(ValueError):
        result_F = elem_F.asarray(out=out_F_wrong)

    # Try wrong shape
    out_F_wrong = np.asfortranarray(np.empty([2, 3]))
    with pytest.raises(ValueError):
        result_F = elem_F.asarray(out=out_F_wrong)

    # Try wrong order
    out_F_wrong = np.empty([2, 2])
    with pytest.raises(ValueError):
        elem_F.asarray(out=out_F_wrong)

    # Also check with C ordering
    discr_C = odl.uniform_discr([0, 0], [1, 1], (2, 2), order='C')
    elem_C = discr_C.element([[1, 2],
                             [3, 4]])

    # Verify that returned array equals input data
    assert all_equal(elem_C.asarray(), [[1, 2],
                                        [3, 4]])

    # Check order of out array
    assert elem_C.asarray().flags['C_CONTIGUOUS']

    # test out parameter
    out_C = np.empty([2, 2])
    result_C = elem_C.asarray(out=out_C)
    assert result_C is out_C
    assert all_equal(out_C, [[1, 2],
                             [3, 4]])

    # Try discontinuous
    out_C_wrong = np.empty([4, 2])[::2, :]
    with pytest.raises(ValueError):
        result_C = elem_C.asarray(out=out_C_wrong)

    # Try wrong shape
    out_C_wrong = np.empty([2, 3])
    with pytest.raises(ValueError):
        result_C = elem_C.asarray(out=out_C_wrong)

    # Try wrong order
    out_C_wrong = np.asfortranarray(np.empty([2, 2]))
    with pytest.raises(ValueError):
        elem_C.asarray(out=out_C_wrong)


def test_transpose():
    discr = odl.uniform_discr([0, 0], [1, 1], [2, 2], order='F')
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


def test_astype():

    rdiscr = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype='float64')
    cdiscr = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype='complex128')
    rdiscr_s = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype='float32')
    cdiscr_s = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype='complex64')

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


def test_ufunc(tspace_impl, ufunc):
    space = odl.uniform_discr([0, 0], [1, 1], (2, 2), impl=tspace_impl)
    name, n_args, n_out, _ = ufunc
    if (np.issubsctype(space.dtype, np.floating) and
            name in ['bitwise_and',
                     'bitwise_or',
                     'bitwise_xor',
                     'invert',
                     'left_shift',
                     'right_shift']):
        # Skip integer only methods if floating point type
        return

    # Get the ufunc from numpy as reference
    ufunc = getattr(np, name)

    # Create some data
    arrays, vectors = noise_elements(space, n_args + n_out)
    in_arrays = arrays[:n_args]
    out_arrays = arrays[n_args:]
    data_vector = vectors[0]
    in_vectors = vectors[1:n_args]
    out_vectors = vectors[n_args:]

    # Verify type
    assert isinstance(data_vector.ufuncs,
                      odl.util.ufuncs.DiscreteLpUfuncs)

    # Out-of-place:
    np_result = ufunc(*in_arrays)
    vec_fun = getattr(data_vector.ufuncs, name)
    odl_result = vec_fun(*in_vectors)
    assert all_almost_equal(np_result, odl_result)

    # Test type of output
    if n_out == 1:
        assert isinstance(odl_result, space.element_type)
    elif n_out > 1:
        for i in range(n_out):
            assert isinstance(odl_result[i], space.element_type)

    # In-place:
    np_result = ufunc(*(in_arrays + out_arrays))
    vec_fun = getattr(data_vector.ufuncs, name)
    odl_result = vec_fun(*(in_vectors + out_vectors))
    assert all_almost_equal(np_result, odl_result)

    # Test in-place actually holds:
    if n_out == 1:
        assert odl_result is out_vectors[0]
    elif n_out > 1:
        for i in range(n_out):
            assert odl_result[i] is out_vectors[i]

    # Test out-of-place with np data
    np_result = ufunc(*in_arrays)
    vec_fun = getattr(data_vector.ufuncs, name)
    odl_result = vec_fun(*in_arrays[1:])
    assert all_almost_equal(np_result, odl_result)

    # Test type of output
    if n_out == 1:
        assert isinstance(odl_result, space.element_type)
    elif n_out > 1:
        for i in range(n_out):
            assert isinstance(odl_result[i], space.element_type)


def test_real_imag():

    # Get real and imag
    cdiscr = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype=complex)
    rdiscr = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype=float)

    x = cdiscr.element([[1 - 1j, 2 - 2j], [3 - 3j, 4 - 4j]])
    assert x.real in rdiscr
    assert all_equal(x.real, [1, 2, 3, 4])
    assert x.imag in rdiscr
    assert all_equal(x.imag, [-1, -2, -3, -4])

    # Set with different data types and shapes
    newreal = rdiscr.element([[2, 3], [4, 5]])
    x.real = newreal
    assert all_equal(x.real, [2, 3, 4, 5])
    newreal = [[3, 4], [5, 6]]
    x.real = newreal
    assert all_equal(x.real, [3, 4, 5, 6])
    newreal = [4, 5, 6, 7]
    x.real = newreal
    assert all_equal(x.real, [4, 5, 6, 7])
    newreal = 0
    x.real = newreal
    assert all_equal(x.real, [0, 0, 0, 0])

    newimag = rdiscr.element([-2, -3, -4, -5])
    x.imag = newimag
    assert all_equal(x.imag, [-2, -3, -4, -5])
    newimag = [[-3, -4], [-5, -6]]
    x.imag = newimag
    assert all_equal(x.imag, [-3, -4, -5, -6])
    newimag = [-4, -5, -6, -7]
    x.imag = newimag
    assert all_equal(x.imag, [-4, -5, -6, -7])
    newimag = -1
    x.imag = newimag
    assert all_equal(x.imag, [-1, -1, -1, -1])

    # 'F' ordering
    cdiscr = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype=complex,
                               order='F')

    x = cdiscr.element()
    newreal = [[3, 4], [5, 6]]
    x.real = newreal
    assert all_equal(x.real, [3, 5, 4, 6])  # flattened in 'F' order
    newreal = [4, 5, 6, 7]
    x.real = newreal
    assert all_equal(x.real, [4, 5, 6, 7])


def test_reduction(tspace_impl, reduction):
    space = odl.uniform_discr([0, 0], [1, 1], [2, 2], impl=tspace_impl)

    name, _ = reduction

    ufunc = getattr(np, name)

    # Create some data
    x_arr, x = noise_elements(space, 1)
    assert almost_equal(ufunc(x_arr), getattr(x.ufuncs, name)())


powers = [1.0, 2.0, 0.5, -0.5, -1.0, -2.0]
power_ids = [' power = {} '.format(p) for p in powers]


@pytest.fixture(scope='module', ids=power_ids, params=powers)
def power(request):
    return request.param


def test_power(tspace_impl, power):
    space = odl.uniform_discr([0, 0], [1, 1], [2, 2], impl=tspace_impl)

    x_arr, x = noise_elements(space, 1)
    x_pos_arr = np.abs(x_arr)
    x_neg_arr = -x_pos_arr
    x_pos = np.abs(x)
    x_neg = -x_pos

    if int(power) != power:
        # Make input positive to get real result
        for y in [x_pos_arr, x_neg_arr, x_pos, x_neg]:
            y += 0.1

    true_pos_pow = np.power(x_pos_arr, power)
    true_neg_pow = np.power(x_neg_arr, power)

    if int(power) != power and tspace_impl == 'cuda':
        with pytest.raises(ValueError):
            x_pos ** power
        with pytest.raises(ValueError):
            x_pos **= power
    else:
        assert all_almost_equal(x_pos ** power, true_pos_pow)
        assert all_almost_equal(x_neg ** power, true_neg_pow)

        x_pos **= power
        x_neg **= power
        assert all_almost_equal(x_pos, true_pos_pow)
        assert all_almost_equal(x_neg, true_neg_pow)


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
        assert almost_equal(discr_testfunc.norm(), true_norm, places=2)


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
        assert almost_equal(discr_testfunc.norm(), true_norm, places=2)


def test_norm_rectangle_boundary(tspace_impl, exponent):
    # Check the constant function 1 in different situations regarding the
    # placement of the outermost grid points.

    if exponent == float('inf'):
        pytest.xfail('inf-norm not implemented in CUDA')

    dtype = 'float32'
    rect = odl.IntervalProd([-1, -2], [1, 2])
    fspace = odl.FunctionSpace(rect, out_dtype=dtype)

    # Standard case
    discr = odl.uniform_discr_fromspace(fspace, (4, 8),
                                        impl=tspace_impl, exponent=exponent)
    if exponent == float('inf'):
        assert discr.one().norm() == 1
    else:
        assert almost_equal(discr.one().norm(),
                            (rect.volume) ** (1 / exponent))

    # Nodes on the boundary (everywhere)
    discr = odl.uniform_discr_fromspace(
        fspace, (4, 8), exponent=exponent,
        impl=tspace_impl, nodes_on_bdry=True)

    if exponent == float('inf'):
        assert discr.one().norm() == 1
    else:
        assert almost_equal(discr.one().norm(),
                            (rect.volume) ** (1 / exponent))

    # Nodes on the boundary (selective)
    discr = odl.uniform_discr_fromspace(
        fspace, (4, 8), exponent=exponent,
        impl=tspace_impl, nodes_on_bdry=((False, True), False))

    if exponent == float('inf'):
        assert discr.one().norm() == 1
    else:
        assert almost_equal(discr.one().norm(),
                            (rect.volume) ** (1 / exponent))

    discr = odl.uniform_discr_fromspace(
        fspace, (4, 8), exponent=exponent,
        impl=tspace_impl, nodes_on_bdry=(False, (True, False)))

    if exponent == float('inf'):
        assert discr.one().norm() == 1
    else:
        assert almost_equal(discr.one().norm(),
                            (rect.volume) ** (1 / exponent))

    # Completely arbitrary boundary
    grid = odl.uniform_grid([0, 0], [1, 1], (4, 4))
    part = odl.RectPartition(rect, grid)
    weight = 1.0 if exponent == float('inf') else part.cell_volume
    dspace = odl.rn(part.shape, dtype=dtype, impl=tspace_impl,
                    exponent=exponent, weighting=weight)
    discr = DiscreteLp(fspace, part, dspace)

    if exponent == float('inf'):
        assert discr.one().norm() == 1
    else:
        assert almost_equal(discr.one().norm(),
                            (rect.volume) ** (1 / exponent))


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
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
