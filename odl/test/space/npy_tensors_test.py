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
import operator
import scipy as sp
import sys

import odl
from odl.set.space import LinearSpaceTypeError
from odl.space.npy_tensors import (
    NumpyTensor, NumpyTensorSet, NumpyTensorSpace,
    NumpyTensorSpaceConstWeighting, NumpyTensorSpaceArrayWeighting,
    NumpyTensorSpaceNoWeighting, NumpyTensorSpaceCustomInner,
    NumpyTensorSpaceCustomNorm, NumpyTensorSpaceCustomDist,
    npy_weighted_inner, npy_weighted_norm, npy_weighted_dist)
from odl.util.testutils import (
    all_almost_equal, all_equal, simple_fixture,
    noise_array, noise_element, noise_elements)
from odl.util.ufuncs import UFUNCS, REDUCTIONS


# --- Test helpers --- #

# Check for python3
PYTHON2 = sys.version_info < (3, 0)


# Helpers to generate data
def _pos_array(tspace):
    """Create an array with positive real entries in ``tspace``."""
    return np.abs(noise_array(tspace)) + 0.1


# --- Pytest fixtures --- #

exponent = simple_fixture(
    name='exponent',
    params=[2.0, 1.0, float('inf'), 0.5, 1.5])


s_indices_params = [
    0, [1], (1,), (0, 1), (2, 3),
    (0, slice(None)), (slice(None, None, 2), slice(None))
    ]
setitem_indices = simple_fixture(name='indices', params=s_indices_params)

g_indices_params = s_indices_params + [[[0, 2, 1, 1], [0, 1, 1, 3]],
                                       (Ellipsis, None)]
getitem_indices = simple_fixture(name='indices', params=g_indices_params)


weight_params = [None, 0.5, _pos_array(odl.tensor_space((3, 4))),
                 _pos_array(odl.tensor_space((3, 4))).ravel()]
weight_ids = [' weight = None ', ' weight = 0.5 ', ' weight = <array>',
              ' weight = <array>']


@pytest.fixture(scope='module', params=weight_params, ids=weight_ids)
def weight(request):
    return request.param


@pytest.fixture(scope='module')
def tspace(floating_dtype):
    return odl.tensor_space(shape=(3, 4), dtype=floating_dtype)


@pytest.fixture(scope='module')
def tset(scalar_dtype):
    return odl.tensor_set(shape=(3, 4), dtype=scalar_dtype)


# --- Space classes --- #


def test_init_tset():
    """Test the different initialization patterns and options."""
    # Basic class constructors
    NumpyTensorSet((3, 4), dtype='S1')
    NumpyTensorSet((3, 4), dtype=int)
    NumpyTensorSet((3, 4), dtype=float)
    NumpyTensorSet((3, 4), dtype=complex)
    NumpyTensorSet((3, 4), dtype=complex, order='F')

    # Alternative constructors
    odl.tensor_set((3, 4), dtype=int)
    odl.tensor_set((3, 4), dtype='S1', order='F')

    # Check error scenarios
    with pytest.raises(TypeError):
        NumpyTensorSet(3, dtype=int)  # bad shape


def test_init_tspace():
    """Test the different initialization patterns and options."""
    # Basic class constructor
    NumpyTensorSpace((3, 4))
    NumpyTensorSpace((3, 4), dtype=int)
    NumpyTensorSpace((3, 4), dtype=float)
    NumpyTensorSpace((3, 4), dtype=complex)
    NumpyTensorSpace((3, 4), dtype=complex, order='F')
    NumpyTensorSpace((3, 4), dtype=complex, exponent=1.0)
    NumpyTensorSpace((3, 4), dtype=complex, exponent=float('inf'))

    # Alternative constructor
    odl.tensor_space((3, 4))
    odl.tensor_space((3, 4), dtype=int)
    odl.tensor_space((3, 4), order='F')
    odl.tensor_space((3, 4), exponent=1.0)

    # Only works with scalar data types
    with pytest.raises(ValueError):
        NumpyTensorSpace((3, 4), dtype='S1')

    # Constructors for real spaces
    odl.rn((3, 4))
    odl.rn((3, 4), dtype='float32')
    odl.rn(3)
    odl.rn(3, dtype='float32')

    # Works only for real data types
    with pytest.raises(ValueError):
        odl.rn((3, 4), complex)
    with pytest.raises(ValueError):
        odl.rn(3, int)
    with pytest.raises(ValueError):
        odl.rn(3, 'S1')

    # Constructors for complex spaces
    odl.cn((3, 4))
    odl.cn((3, 4), dtype='complex64')
    odl.cn(3)
    odl.cn(3, dtype='complex64')

    # Works only for complex data types
    with pytest.raises(ValueError):
        odl.cn((3, 4), float)
    with pytest.raises(ValueError):
        odl.cn(3, 'S1')

    # Backported int from future fails (not recognized by numpy.dtype())
    # (Python 2 only)
    from builtins import int as future_int
    if PYTHON2:
        with pytest.raises(ValueError):
            NumpyTensorSpace((3, 4), future_int)

    # Init with weights or custom space functions
    weight_const = 1.5
    weight_arr = _pos_array(odl.rn((3, 4), float))

    odl.rn((3, 4), weighting=weight_const)
    odl.rn((3, 4), weighting=weight_arr)


def test_init_tspace_weighting(weight, exponent):
    """Test if weightings during init give the correct weighting classes."""

    space = odl.tensor_space((3, 4), weighting=weight, exponent=exponent)
    if isinstance(weight, np.ndarray):
        weighting = NumpyTensorSpaceArrayWeighting(weight, exponent=exponent)
    elif weight is None:
        weighting = NumpyTensorSpaceNoWeighting(exponent=exponent)
    else:
        weighting = NumpyTensorSpaceConstWeighting(weight, exponent=exponent)

    assert space.weighting == weighting

    # Using the class directly
    space = odl.tensor_space((3, 4), weighting=weighting, exponent=exponent)
    assert space.weighting is weighting

    # Errors for bad input
    with pytest.raises(ValueError):
        odl.tensor_space((3, 4), weighting=np.ones([2, 4]))  # bad size

    with pytest.raises(ValueError):
        odl.tensor_space((3, 4), weighting=1j * np.ones([2, 4]))  # bad dtype

    with pytest.raises(TypeError):
        odl.tensor_space((3, 4), weighting=1j)  # raised by float() conversion


def test_properties():
    """Test that the space and element properties are as expected."""
    tspace = odl.tensor_space((3, 4), dtype=complex, exponent=1, weighting=2)
    x = tspace.element()
    assert x.space is tspace
    assert x.ndim == tspace.ndim == 2
    assert x.dtype == tspace.dtype == np.dtype(complex)
    assert x.size == tspace.size == 12
    assert x.shape == tspace.shape == (3, 4)
    assert x.itemsize == tspace.dtype.itemsize
    assert x.nbytes == x.itemsize * x.size


def test_tspace_astype():
    """Test creation of real/complex space counterparts."""
    real = odl.rn((3, 4), weighting=1.5)
    cplx = odl.cn((3, 4), weighting=1.5)
    real_s = odl.rn((3, 4), weighting=1.5, dtype='float32')
    cplx_s = odl.cn((3, 4), weighting=1.5, dtype='complex64')

    # Real
    assert real.astype('float32') == real_s
    assert real.astype('float64') is real
    assert real.real_space is real
    assert real.astype('complex64') == cplx_s
    assert real.astype('complex128') == cplx
    assert real.complex_space == cplx

    # Complex
    assert cplx.astype('complex64') == cplx_s
    assert cplx.astype('complex128') is cplx
    assert cplx.real_space == real
    assert cplx.astype('float32') == real_s
    assert cplx.astype('float64') == real
    assert cplx.complex_space is cplx


def _test_lincomb(tspace, a, b):
    """Validate lincomb against direct result using arrays."""

    # Unaliased arguments
    [xarr, yarr, zarr], [x, y, z] = noise_elements(tspace, 3)
    zarr[:] = a * xarr + b * yarr
    tspace.lincomb(a, x, b, y, out=z)
    assert all_almost_equal([x, y, z], [xarr, yarr, zarr])

    # First argument aliased with output
    [xarr, yarr, zarr], [x, y, z] = noise_elements(tspace, 3)
    zarr[:] = a * zarr + b * yarr
    tspace.lincomb(a, z, b, y, out=z)
    assert all_almost_equal([x, y, z], [xarr, yarr, zarr])

    # Second argument aliased with output
    [xarr, yarr, zarr], [x, y, z] = noise_elements(tspace, 3)
    zarr[:] = a * xarr + b * zarr
    tspace.lincomb(a, x, b, z, out=z)
    assert all_almost_equal([x, y, z], [xarr, yarr, zarr])

    # Both arguments aliased with each other
    [xarr, yarr, zarr], [x, y, z] = noise_elements(tspace, 3)
    zarr[:] = a * xarr + b * xarr
    tspace.lincomb(a, x, b, x, out=z)
    assert all_almost_equal([x, y, z], [xarr, yarr, zarr])

    # All aliased
    [xarr, yarr, zarr], [x, y, z] = noise_elements(tspace, 3)
    zarr[:] = a * zarr + b * zarr
    tspace.lincomb(a, z, b, z, out=z)
    assert all_almost_equal([x, y, z], [xarr, yarr, zarr])


def test_lincomb(tspace):
    """Validate lincomb against direct result using arrays and some scalars."""
    scalar_values = [0, 1, -1, 3.41]
    for a in scalar_values:
        for b in scalar_values:
            _test_lincomb(tspace, a, b)


def test_lincomb_raise(tspace):
    """Test if lincomb raises correctly for bad input."""
    other_tspace = odl.rn((4, 3))

    other_x = other_tspace.zero()
    x, y, z = tspace.zero(), tspace.zero(), tspace.zero()

    with pytest.raises(LinearSpaceTypeError):
        tspace.lincomb(1, other_x, 1, y, z)

    with pytest.raises(LinearSpaceTypeError):
        tspace.lincomb(1, y, 1, other_x, z)

    with pytest.raises(LinearSpaceTypeError):
        tspace.lincomb(1, y, 1, z, other_x)

    with pytest.raises(LinearSpaceTypeError):
        tspace.lincomb([], x, 1, y, z)

    with pytest.raises(LinearSpaceTypeError):
        tspace.lincomb(1, x, [], y, z)


def test_multiply(tspace):
    """Test multiply against direct array multiplication."""
    # space method
    [x_arr, y_arr, out_arr], [x, y, out] = noise_elements(tspace, 3)
    out_arr = x_arr * y_arr

    tspace.multiply(x, y, out)
    assert all_almost_equal([x_arr, y_arr, out_arr], [x, y, out])

    # member method
    [x_arr, y_arr, out_arr], [x, y, out] = noise_elements(tspace, 3)
    out_arr = x_arr * y_arr

    x.multiply(y, out=out)
    assert all_almost_equal([x_arr, y_arr, out_arr], [x, y, out])


def test_multiply_exceptions(tspace):
    """Test if multiply raises correctly for bad input."""
    other_tspace = odl.rn((4, 3))

    other_x = other_tspace.zero()
    x, y = tspace.zero(), tspace.zero()

    with pytest.raises(LinearSpaceTypeError):
        tspace.multiply(other_x, x, y)

    with pytest.raises(LinearSpaceTypeError):
        tspace.multiply(x, other_x, y)

    with pytest.raises(LinearSpaceTypeError):
        tspace.multiply(x, y, other_x)


def test_power(tspace):
    """Test ``**`` against direct array exponentiation."""
    [x_arr, y_arr], [x, y] = noise_elements(tspace, n=2)
    y_pos = tspace.element(np.abs(y) + 0.1)
    y_pos_arr = np.abs(y_arr) + 0.1

    # Testing standard positive integer power out-of-place and in-place
    assert all_almost_equal(x ** 2, x_arr ** 2)
    y **= 2
    y_arr **= 2
    assert all_almost_equal(y, y_arr)

    # Real number and negative integer power
    assert all_almost_equal(y_pos ** 1.3, y_pos_arr ** 1.3)
    assert all_almost_equal(y_pos ** (-3), y_pos_arr ** (-3))
    y_pos **= 2.5
    y_pos_arr **= 2.5
    assert all_almost_equal(y_pos, y_pos_arr)

    # Array raised to the power of another array, entry-wise
    assert all_almost_equal(y_pos ** x, y_pos_arr ** x_arr)
    y_pos **= x.real
    y_pos_arr **= x_arr.real
    assert all_almost_equal(y_pos, y_pos_arr)


def test_unary_ops(tspace):
    """Verify that the unary operators (`+x` and `-x`) work as expected."""
    for op in [operator.pos, operator.neg]:
        x_arr, x = noise_elements(tspace)

        y_arr = op(x_arr)
        y = op(x)

        assert all_almost_equal([x, y], [x_arr, y_arr])


def test_scalar_operator(tspace, arithmetic_op):
    """Verify binary operations with scalars.

    Verifies that the statement y = op(x, scalar) gives equivalent results
    to NumPy.
    """
    if arithmetic_op in (operator.truediv, operator.itruediv):
        places = int(-np.log10(np.finfo(tspace.dtype).resolution) // 2)
    else:
        places = int(-np.log10(np.finfo(tspace.dtype).resolution))

    for scalar in [-31.2, -1, 0, 1, 2.13]:
        x_arr, x = noise_elements(tspace)

        # Left op
        if scalar == 0 and arithmetic_op in [operator.truediv,
                                             operator.itruediv]:
            # Check for correct zero division behaviour
            with pytest.raises(ZeroDivisionError):
                y = arithmetic_op(x, scalar)
        else:
            y_arr = arithmetic_op(x_arr, scalar)
            y = arithmetic_op(x, scalar)

            assert all_almost_equal([x, y], [x_arr, y_arr], places=places)

        # right op
        x_arr, x = noise_elements(tspace)

        y_arr = arithmetic_op(scalar, x_arr)
        y = arithmetic_op(scalar, x)

        assert all_almost_equal([x, y], [x_arr, y_arr], places=places)


def test_binary_operator(tspace, arithmetic_op):
    """Verify binary operations with tensors.

    Verifies that the statement z = op(x, y) gives equivalent results
    to NumPy.
    """
    if arithmetic_op in (operator.truediv, operator.itruediv):
        places = int(-np.log10(np.finfo(tspace.dtype).resolution) // 2)
    else:
        places = int(-np.log10(np.finfo(tspace.dtype).resolution))

    [x_arr, y_arr], [x, y] = noise_elements(tspace, 2)

    # non-aliased left
    z_arr = arithmetic_op(x_arr, y_arr)
    z = arithmetic_op(x, y)

    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr], places=places)

    # non-aliased right
    z_arr = arithmetic_op(y_arr, x_arr)
    z = arithmetic_op(y, x)

    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr], places=places)

    # aliased operation
    z_arr = arithmetic_op(x_arr, x_arr)
    z = arithmetic_op(x, x)

    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr], places=places)


def test_assign(tspace):
    """Test the assign method using ``==`` comparison."""
    x = noise_element(tspace)
    y = noise_element(tspace)

    y.assign(x)

    assert y == x
    assert y is not x

    # test alignment
    x *= 2
    assert y != x


def test_inner(tspace):
    """Test the inner method against numpy.vdot."""
    xd = noise_element(tspace)
    yd = noise_element(tspace)

    correct_inner = np.vdot(yd, xd)
    assert tspace.inner(xd, yd) == pytest.approx(correct_inner)
    assert xd.inner(yd) == pytest.approx(correct_inner)


def test_inner_exceptions(tspace):
    """Test if inner raises correctly for bad input."""
    other_tspace = odl.rn((4, 3))
    other_x = other_tspace.zero()
    x = tspace.zero()

    with pytest.raises(LinearSpaceTypeError):
        tspace.inner(other_x, x)

    with pytest.raises(LinearSpaceTypeError):
        tspace.inner(x, other_x)


def test_norm(tspace):
    """Test the norm method against numpy.linalg.norm."""
    xarr, x = noise_elements(tspace)

    correct_norm = np.linalg.norm(xarr.ravel())
    assert tspace.norm(x) == pytest.approx(correct_norm)
    assert x.norm() == pytest.approx(correct_norm)


def test_norm_exceptions(tspace):
    """Test if norm raises correctly for bad input."""
    other_tspace = odl.rn((4, 3))
    other_x = other_tspace.zero()

    with pytest.raises(LinearSpaceTypeError):
        tspace.norm(other_x)


def test_pnorm(exponent):
    """Test the norm method with p!=2 against numpy.linalg.norm."""
    for tspace in (odl.rn((3, 4), exponent=exponent),
                   odl.cn((3, 4), exponent=exponent)):
        xarr, x = noise_elements(tspace)
        correct_norm = np.linalg.norm(xarr.ravel(), ord=exponent)

        assert tspace.norm(x) == pytest.approx(correct_norm)
        assert x.norm() == pytest.approx(correct_norm)


def test_dist(tspace):
    """Test the dist method against numpy.linalg.norm of the difference."""
    [xarr, yarr], [x, y] = noise_elements(tspace, n=2)

    correct_dist = np.linalg.norm((xarr - yarr).ravel())
    assert tspace.dist(x, y) == pytest.approx(correct_dist)
    assert x.dist(y) == pytest.approx(correct_dist)


def test_dist_exceptions(tspace):
    """Test if dist raises correctly for bad input."""
    other_tspace = odl.rn((4, 3))
    other_x = other_tspace.zero()
    x = tspace.zero()

    with pytest.raises(LinearSpaceTypeError):
        tspace.dist(other_x, x)

    with pytest.raises(LinearSpaceTypeError):
        tspace.dist(x, other_x)


def test_pdist(exponent):
    """Test the dist method with p!=2 against numpy.linalg.norm of diff."""
    for tspace in (odl.rn((3, 4), exponent=exponent),
                   odl.cn((3, 4), exponent=exponent)):
        [xarr, yarr], [x, y] = noise_elements(tspace, n=2)

        correct_dist = np.linalg.norm((xarr - yarr).ravel(), ord=exponent)
        assert tspace.dist(x, y) == pytest.approx(correct_dist)
        assert x.dist(y) == pytest.approx(correct_dist)


def test_getitem(getitem_indices):
    """Check if getitem produces correct values, shape and other stuff."""
    tspace = odl.tensor_space((3, 4), dtype=complex, exponent=1, weighting=2)
    x_arr, x = noise_elements(tspace)

    x_arr_sliced = x_arr[getitem_indices]
    sliced_shape = x_arr_sliced.shape
    x_sliced = x[getitem_indices]

    if np.isscalar(x_arr_sliced):
        assert x_arr_sliced == x_sliced
    else:
        assert x_sliced.shape == sliced_shape
        assert all_equal(x_sliced, x_arr_sliced)

        # Check that the space properties are preserved
        sliced_spc = x_sliced.space
        assert sliced_spc.shape == sliced_shape
        assert sliced_spc.dtype == tspace.dtype
        assert sliced_spc.exponent == tspace.exponent
        assert sliced_spc.weighting == tspace.weighting

        # Check that we have a view that manipulates the original array
        # (or not, depending on indexing style)
        x_arr_sliced[:] = 0
        x_sliced[:] = 0
        assert all_equal(x_arr, x)


def test_setitem(setitem_indices):
    """Check if setitem produces the same result as NumPy."""
    tspace = odl.tensor_space((3, 4), dtype=complex, exponent=1, weighting=2)
    x_arr, x = noise_elements(tspace)

    x_arr_sliced = x_arr[setitem_indices]
    sliced_shape = x_arr_sliced.shape

    # Setting values with scalars
    x_arr[setitem_indices] = 1j
    x[setitem_indices] = 1j
    assert all_equal(x, x_arr)

    # Setting values with arrays
    rhs_arr = np.ones(sliced_shape)
    x_arr[setitem_indices] = rhs_arr
    x[setitem_indices] = rhs_arr
    assert all_equal(x, x_arr)

    # Using a list of lists
    rhs_list = (-np.ones(sliced_shape)).tolist()
    x_arr[setitem_indices] = rhs_list
    x[setitem_indices] = rhs_list
    assert all_equal(x, x_arr)


def test_transpose():
    """Test the .T property of tensors against plain inner product."""
    tspace = odl.tensor_space((3, 4), dtype=complex, weighting=2)
    x = noise_element(tspace)
    y = noise_element(tspace)

    # Assert linear operator
    assert isinstance(x.T, odl.Operator)
    assert x.T.is_linear

    # Check result
    assert x.T(y) == pytest.approx(y.inner(x))
    assert all_equal(x.T.adjoint(1.0), x)

    # x.T.T returns self
    assert x.T.T == x


def test_multiply_by_scalar(tspace):
    """Verify that mult. with NumPy scalars preserves the element type."""
    x = tspace.zero()
    assert x * 1.0 in tspace
    assert x * np.float32(1.0) in tspace
    assert 1.0 * x in tspace
    assert np.float32(1.0) * x in tspace


def test_member_copy():
    """Test copy method of elements."""
    tspace = odl.tensor_space((3, 4), dtype=complex, exponent=1, weighting=2)
    x = noise_element(tspace)

    y = x.copy()
    assert x == y
    assert y is not x

    # Check that result is not aliased
    x *= 2
    assert x != y


def test_python_copy():
    """Test compatibility with the Python copy module."""
    import copy
    tspace = odl.tensor_space((3, 4), dtype=complex, exponent=1, weighting=2)
    x = noise_element(tspace)

    # Shallow copy
    y = copy.copy(x)
    assert x == y
    assert y is not x

    # Check that result is not aliased
    x *= 2
    assert x != y

    # Deep copy
    z = copy.deepcopy(x)
    assert x == z
    assert z is not x

    # Check that result is not aliased
    x *= 2
    assert x != z


def test_conversion_to_scalar():
    """Test conversion of size-1 vectors/tensors to scalars."""
    # Size 1 real space
    value = 1.5
    element = odl.rn(1).element(value)

    assert int(element) == int(value)
    assert float(element) == float(value)
    assert complex(element) == complex(value)
    if PYTHON2:
        long = eval('long')
        assert long(element) == long(value)

    # Size 1 complex space
    value = 1.5 + 0.5j
    element = odl.cn(1).element(value)
    assert complex(element) == complex(value)

    # Size 1 multi-dimensional space
    value = 2.1
    element = odl.rn((1, 1, 1)).element(value)
    assert float(element) == float(value)

    # Too large space
    element = odl.rn(2).one()

    with pytest.raises(TypeError):
        int(element)
    with pytest.raises(TypeError):
        float(element)
    with pytest.raises(TypeError):
        complex(element)
    if PYTHON2:
        with pytest.raises(TypeError):
            long(element)


def test_numpy_array_interface():
    """Verify that the __array__ interface for NumPy works."""
    tspace = odl.tensor_space((3, 4), dtype='float32', exponent=1, weighting=2)
    x = tspace.zero()
    arr = x.__array__()

    assert isinstance(arr, np.ndarray)
    assert np.array_equal(arr, np.zeros(x.shape))


def test_array_wrap_method():
    """Verify that the __array_wrap__ method for NumPy works."""
    tspace = odl.tensor_space((3, 4), dtype='float32', exponent=1, weighting=2)
    x_arr, x = noise_elements(tspace)
    y_arr = np.sin(x_arr)
    y = np.sin(x)  # Should yield again an ODL tensor

    assert all_equal(y, y_arr)
    assert y in tspace


def test_conj(tspace):
    """Test complex conjugation of tensors."""
    xarr, x = noise_elements(tspace)

    xconj = x.conj()
    assert all_equal(xconj, xarr.conj())

    y = tspace.element()
    xconj = x.conj(out=y)
    assert xconj is y
    assert all_equal(y, xarr.conj())


# --- MatVecOperator --- #

@pytest.mark.xfail(reason='No MatVecOperator currently')
def test_matvec_init(tspace):

    # Square matrices, sparse and dense
    sparse_mat = _sparse_matrix(tspace)
    dense_mat = _dense_matrix(tspace)

    MatVecOperator(sparse_mat, tspace, tspace)
    MatVecOperator(dense_mat, tspace, tspace)

    # Test defaults
    op_float = MatVecOperator([[1.0, 2],
                               [-1, 0.5]])

    assert isinstance(op_float.domain, NumpyTensorSpace)
    assert op_float.domain.is_rn
    assert isinstance(op_float.range, NumpyTensorSpace)
    assert op_float.domain.is_rn

    op_complex = MatVecOperator([[1.0, 2 + 1j],
                                 [-1 - 1j, 0.5]])

    assert isinstance(op_complex.domain, NumpyTensorSpace)
    assert op_complex.domain.is_cn
    assert isinstance(op_complex.range, NumpyTensorSpace)
    assert op_complex.domain.is_cn

    op_int = MatVecOperator([[1, 2],
                             [-1, 0]])

    assert isinstance(op_int.domain, NumpyTensorSpace)
    assert op_int.domain.dtype == int
    assert isinstance(op_int.range, NumpyTensorSpace)
    assert op_int.domain.dtype == int

    # Rectangular
    rect_mat = 2 * np.eye(2, 3)
    r2 = odl.rn(2)
    r3 = odl.rn(3)

    MatVecOperator(rect_mat, r3, r2)

    with pytest.raises(ValueError):
        MatVecOperator(rect_mat, r2, r2)

    with pytest.raises(ValueError):
        MatVecOperator(rect_mat, r3, r3)

    with pytest.raises(ValueError):
        MatVecOperator(rect_mat, r2, r3)

    # Rn to Cn okay
    MatVecOperator(rect_mat, r3, odl.cn(2))

    # Cn to Rn not okay (no safe cast)
    with pytest.raises(TypeError):
        MatVecOperator(rect_mat, odl.cn(3), r2)

    # Complex matrix between real spaces not okay
    rect_complex_mat = rect_mat + 1j
    with pytest.raises(TypeError):
        MatVecOperator(rect_complex_mat, r3, r2)

    # Init with array-like structure (including numpy.matrix)
    MatVecOperator(rect_mat.tolist(), r3, r2)
    MatVecOperator(np.asmatrix(rect_mat), r3, r2)


@pytest.mark.xfail(reason='No MatVecOperator currently')
def test_matvec_simple_properties():
    # Matrix - always ndarray in for dense input, scipy.sparse.spmatrix else
    rect_mat = 2 * np.eye(2, 3)
    r2 = odl.rn(2)
    r3 = odl.rn(3)

    op = MatVecOperator(rect_mat, r3, r2)
    assert isinstance(op.matrix, np.ndarray)

    op = MatVecOperator(np.asmatrix(rect_mat), r3, r2)
    assert isinstance(op.matrix, np.ndarray)

    op = MatVecOperator(rect_mat.tolist(), r3, r2)
    assert isinstance(op.matrix, np.ndarray)
    assert not op.matrix_issparse

    sparse_mat = _sparse_matrix(odl.rn(5))
    op = MatVecOperator(sparse_mat, odl.rn(5), odl.rn(5))
    assert isinstance(op.matrix, sp.sparse.spmatrix)
    assert op.matrix_issparse


@pytest.mark.xfail(reason='No MatVecOperator currently')
def test_matvec_adjoint(tspace):
    # Square cases
    sparse_mat = _sparse_matrix(tspace)
    dense_mat = _dense_matrix(tspace)

    op_sparse = MatVecOperator(sparse_mat, tspace, tspace)
    op_dense = MatVecOperator(dense_mat, tspace, tspace)

    # Just test if it runs, nothing interesting to test here
    op_sparse.adjoint
    op_dense.adjoint

    # Rectangular case
    rect_mat = 2 * np.eye(2, 3)
    r2, r3 = odl.rn(2), odl.rn(3)
    c2 = odl.cn(2)

    op = MatVecOperator(rect_mat, r3, r2)
    op_adj = op.adjoint
    assert op_adj.domain == op.range
    assert op_adj.range == op.domain
    assert np.array_equal(op_adj.matrix, op.matrix.conj().T)
    assert np.array_equal(op_adj.adjoint.matrix, op.matrix)

    # The operator Rn -> Cn has no adjoint
    op_noadj = MatVecOperator(rect_mat, r3, c2)
    with pytest.raises(NotImplementedError):
        op_noadj.adjoint


@pytest.mark.xfail(reason='No MatVecOperator currently')
def test_matvec_inverse(tspace):
    # Sparse case
    sparse_mat = _sparse_matrix(tspace)
    op_sparse = MatVecOperator(sparse_mat, tspace, tspace)

    op_sparse_inv = op_sparse.inverse
    assert op_sparse_inv.domain == op_sparse.range
    assert op_sparse_inv.range == op_sparse.domain
    assert all_almost_equal(op_sparse_inv.matrix,
                            np.linalg.inv(op_sparse.matrix.todense()))
    assert all_almost_equal(op_sparse_inv.inverse.matrix,
                            op_sparse.matrix.todense())

    # Test application
    x = noise_element(tspace)
    assert all_almost_equal(x, op_sparse.inverse(op_sparse(x)))

    # Dense case
    dense_mat = _dense_matrix(tspace)
    op_dense = MatVecOperator(dense_mat, tspace, tspace)
    op_dense_inv = op_dense.inverse
    assert op_dense_inv.domain == op_dense.range
    assert op_dense_inv.range == op_dense.domain
    assert all_almost_equal(op_dense_inv.matrix,
                            np.linalg.inv(op_dense.matrix))
    assert all_almost_equal(op_dense_inv.inverse.matrix,
                            op_dense.matrix)

    # Test application
    x = noise_element(tspace)
    assert all_almost_equal(x, op_dense.inverse(op_dense(x)))


@pytest.mark.xfail(reason='No MatVecOperator currently')
def test_matvec_call(tspace):
    # Square cases
    sparse_mat = _sparse_matrix(tspace)
    dense_mat = _dense_matrix(tspace)
    xarr, x = noise_elements(tspace)

    op_sparse = MatVecOperator(sparse_mat, tspace, tspace)
    op_dense = MatVecOperator(dense_mat, tspace, tspace)

    yarr_sparse = sparse_mat.dot(xarr)
    yarr_dense = dense_mat.dot(xarr)

    # Out-of-place
    y = op_sparse(x)
    assert all_almost_equal(y, yarr_sparse)

    y = op_dense(x)
    assert all_almost_equal(y, yarr_dense)

    # In-place
    y = tspace.element()
    op_sparse(x, out=y)
    assert all_almost_equal(y, yarr_sparse)

    y = tspace.element()
    op_dense(x, out=y)
    assert all_almost_equal(y, yarr_dense)

    # Rectangular case
    rect_mat = 2 * np.eye(2, 3)
    r2, r3 = odl.rn(2), odl.rn(3)

    op = MatVecOperator(rect_mat, r3, r2)
    xarr = np.arange(3, dtype=float)
    x = r3.element(xarr)

    yarr = rect_mat.dot(xarr)

    # Out-of-place
    y = op(x)
    assert all_almost_equal(y, yarr)

    # In-place
    y = r2.element()
    op(x, out=y)
    assert all_almost_equal(y, yarr)


# --- Weightings --- #


def test_array_weighting_init(exponent):
    """Test initialization of array weighting."""
    space = odl.rtensors((3, 4))
    weight_arr = _pos_array(space)
    weight_elem = space.element(weight_arr)

    weighting_arr = NumpyTensorSpaceArrayWeighting(weight_arr,
                                                   exponent=exponent)
    weighting_elem = NumpyTensorSpaceArrayWeighting(weight_elem,
                                                    exponent=exponent)

    assert isinstance(weighting_arr.array, np.ndarray)
    assert isinstance(weighting_elem.array, NumpyTensor)


def test_array_weighting_array_is_valid():
    rn = odl.rn(5)
    weight_arr = _pos_array(rn)
    weighting_vec = NumpyTensorSpaceArrayWeighting(weight_arr)

    assert weighting_vec.is_valid()

    # Invalid
    weight_arr[0] = 0
    weighting_vec = NumpyTensorSpaceArrayWeighting(weight_arr)
    assert not weighting_vec.is_valid()


def test_array_weighting_equals():
    rn = odl.rn(5)
    weight_arr = _pos_array(rn)
    weight_elem = rn.element(weight_arr)

    weighting_vec = NumpyTensorSpaceArrayWeighting(weight_arr)
    weighting_vec2 = NumpyTensorSpaceArrayWeighting(weight_arr)
    weighting_elem = NumpyTensorSpaceArrayWeighting(weight_elem)
    weighting_elem2 = NumpyTensorSpaceArrayWeighting(weight_elem)
    weighting_other_vec = NumpyTensorSpaceArrayWeighting(weight_arr - 1)
    weighting_other_exp = NumpyTensorSpaceArrayWeighting(weight_arr - 1,
                                                         exponent=1)

    assert weighting_vec == weighting_vec2
    assert weighting_vec != weighting_elem
    assert weighting_elem == weighting_elem2
    assert weighting_vec != weighting_other_vec
    assert weighting_vec != weighting_other_exp


def test_array_weighting_equiv():
    """Test the equiv method of space weightings."""
    space = odl.rn((3, 4))
    weight_arr = _pos_array(space)
    weight_elem = space.element(weight_arr)
    different_arr = weight_arr + 1

    w_arr = NumpyTensorSpaceArrayWeighting(weight_arr)
    w_elem = NumpyTensorSpaceArrayWeighting(weight_elem)
    w_different_arr = NumpyTensorSpaceArrayWeighting(different_arr)

    # Equal -> True
    assert w_arr.equiv(w_arr)
    assert w_arr.equiv(w_elem)
    # Different array -> False
    assert not w_arr.equiv(w_different_arr)

    # Test shortcuts in the implementation
    const_arr = np.ones(space.shape) * 1.5

    w_const_arr = NumpyTensorSpaceArrayWeighting(const_arr)
    w_const = NumpyTensorSpaceConstWeighting(1.5)
    w_wrong_const = NumpyTensorSpaceConstWeighting(1)
    w_wrong_exp = NumpyTensorSpaceConstWeighting(1.5, exponent=1)

    assert w_const_arr.equiv(w_const)
    assert not w_const_arr.equiv(w_wrong_const)
    assert not w_const_arr.equiv(w_wrong_exp)

    # Bogus input
    assert not w_const_arr.equiv(True)
    assert not w_const_arr.equiv(object)
    assert not w_const_arr.equiv(None)


def test_array_weighting_inner(tspace):
    """Test inner product in a weighted space."""
    rtol = np.sqrt(np.finfo(tspace.dtype).resolution)
    [xarr, yarr], [x, y] = noise_elements(tspace, 2)

    weight_arr = _pos_array(tspace)
    weighting = NumpyTensorSpaceArrayWeighting(weight_arr)

    true_inner = np.vdot(yarr, xarr * weight_arr)
    assert weighting.inner(x, y) == pytest.approx(true_inner)

    # With free function
    inner = npy_weighted_inner(weight_arr)
    assert inner(x, y) == pytest.approx(true_inner, rel=rtol)

    # Exponent != 2 -> no inner product, should raise
    with pytest.raises(NotImplementedError):
        NumpyTensorSpaceArrayWeighting(weight_arr, exponent=1.0).inner(x, y)


def test_array_weighting_norm(tspace, exponent):
    """Test norm in a weighted space."""
    rtol = np.sqrt(np.finfo(tspace.dtype).resolution)
    xarr, x = noise_elements(tspace)

    weight_arr = _pos_array(tspace)
    weighting = NumpyTensorSpaceArrayWeighting(weight_arr, exponent=exponent)

    if exponent == float('inf'):
        true_norm = np.linalg.norm(
            (weight_arr * xarr).ravel(),
            ord=float('inf'))
    else:
        true_norm = np.linalg.norm(
            (weight_arr ** (1 / exponent) * xarr).ravel(),
            ord=exponent)

    assert weighting.norm(x) == pytest.approx(true_norm, rel=rtol)

    # With free function
    pnorm = npy_weighted_norm(weight_arr, exponent=exponent)
    assert pnorm(x) == pytest.approx(true_norm, rel=rtol)


def test_array_weighting_dist(tspace, exponent):
    """Test dist product in a weighted space."""
    rtol = np.sqrt(np.finfo(tspace.dtype).resolution)
    [xarr, yarr], [x, y] = noise_elements(tspace, n=2)

    weight_arr = _pos_array(tspace)
    weighting = NumpyTensorSpaceArrayWeighting(weight_arr, exponent=exponent)

    if exponent == float('inf'):
        true_dist = np.linalg.norm(
            (weight_arr * (xarr - yarr)).ravel(),
            ord=float('inf'))
    else:
        true_dist = np.linalg.norm(
            (weight_arr ** (1 / exponent) * (xarr - yarr)).ravel(),
            ord=exponent)

    assert weighting.dist(x, y) == pytest.approx(true_dist, rel=rtol)

    # With free function
    pdist = npy_weighted_dist(weight_arr, exponent=exponent)
    assert pdist(x, y) == pytest.approx(true_dist, rel=rtol)


def test_array_weighting_dist_using_inner(tspace):
    """Test dist using inner product in a weighted space."""
    rtol = np.sqrt(np.finfo(tspace.dtype).resolution)

    [xarr, yarr], [x, y] = noise_elements(tspace, 2)

    weight_arr = _pos_array(tspace)
    w = NumpyTensorSpaceArrayWeighting(weight_arr)

    true_dist = np.linalg.norm(np.sqrt(weight_arr) * (xarr - yarr))
    assert w.dist(x, y) == pytest.approx(true_dist, rel=rtol)

    # Only possible for exponent == 2
    with pytest.raises(ValueError):
        NumpyTensorSpaceArrayWeighting(weight_arr, exponent=1,
                                       dist_using_inner=True)

    # With free function
    w_dist = npy_weighted_dist(weight_arr, use_inner=True)
    assert w_dist(x, y) == pytest.approx(true_dist, rel=rtol)


def test_const_weighting_init(exponent):
    """Test initialization of constant weightings."""
    constant = 1.5

    # Just test if the code runs
    NumpyTensorSpaceConstWeighting(constant, exponent=exponent)

    with pytest.raises(ValueError):
        NumpyTensorSpaceConstWeighting(0)
    with pytest.raises(ValueError):
        NumpyTensorSpaceConstWeighting(-1)
    with pytest.raises(ValueError):
        NumpyTensorSpaceConstWeighting(float('inf'))


def test_const_weighting_comparison():
    """Test equality to and equivalence with const weightings."""
    constant = 1.5

    w_const = NumpyTensorSpaceConstWeighting(constant)
    w_const2 = NumpyTensorSpaceConstWeighting(constant)
    w_other_const = NumpyTensorSpaceConstWeighting(constant + 1)
    w_other_exp = NumpyTensorSpaceConstWeighting(constant, exponent=1)

    const_arr = constant * np.ones((3, 4))
    w_const_arr = NumpyTensorSpaceArrayWeighting(const_arr)
    other_const_arr = (constant + 1) * np.ones((3, 4))
    w_other_const_arr = NumpyTensorSpaceArrayWeighting(other_const_arr)

    assert w_const == w_const
    assert w_const == w_const2
    assert w_const2 == w_const
    # Different but equivalent
    assert w_const.equiv(w_const_arr)
    assert w_const != w_const_arr

    # Not equivalent
    assert not w_const.equiv(w_other_exp)
    assert w_const != w_other_exp
    assert not w_const.equiv(w_other_const)
    assert w_const != w_other_const
    assert not w_const.equiv(w_other_const_arr)
    assert w_const != w_other_const_arr

    # Bogus input
    assert not w_const.equiv(True)
    assert not w_const.equiv(object)
    assert not w_const.equiv(None)


def test_const_weighting_inner(tspace):
    """Test inner product with const weighting."""
    [xarr, yarr], [x, y] = noise_elements(tspace, 2)

    constant = 1.5
    true_result_const = constant * np.vdot(yarr, xarr)

    w_const = NumpyTensorSpaceConstWeighting(constant)
    assert w_const.inner(x, y) == pytest.approx(true_result_const)

    # Exponent != 2 -> no inner
    w_const = NumpyTensorSpaceConstWeighting(constant, exponent=1)
    with pytest.raises(NotImplementedError):
        w_const.inner(x, y)


def test_const_weighting_norm(tspace, exponent):
    """Test norm with const weighting."""
    xarr, x = noise_elements(tspace)

    constant = 1.5
    if exponent == float('inf'):
        factor = constant
    else:
        factor = constant ** (1 / exponent)
    true_norm = factor * np.linalg.norm(xarr.ravel(), ord=exponent)

    w_const = NumpyTensorSpaceConstWeighting(constant, exponent=exponent)
    assert w_const.norm(x) == pytest.approx(true_norm)

    # With free function
    w_const_norm = npy_weighted_norm(constant, exponent=exponent)
    assert w_const_norm(x) == pytest.approx(true_norm)


def test_const_weighting_dist(tspace, exponent):
    """Test dist with const weighting."""
    [xarr, yarr], [x, y] = noise_elements(tspace, 2)

    constant = 1.5
    if exponent == float('inf'):
        factor = constant
    else:
        factor = constant ** (1 / exponent)
    true_dist = factor * np.linalg.norm((xarr - yarr).ravel(), ord=exponent)

    w_const = NumpyTensorSpaceConstWeighting(constant, exponent=exponent)
    assert w_const.dist(x, y) == pytest.approx(true_dist)

    # With free function
    w_const_dist = npy_weighted_dist(constant, exponent=exponent)
    assert w_const_dist(x, y) == pytest.approx(true_dist)


def test_const_weighting_dist_using_inner(tspace):
    """Test dist with constant weighting using inner product."""
    rtol = np.sqrt(np.finfo(tspace.dtype).resolution)

    [xarr, yarr], [x, y] = noise_elements(tspace, 2)

    constant = 1.5
    w = NumpyTensorSpaceConstWeighting(constant)

    true_dist = np.sqrt(constant) * np.linalg.norm(xarr - yarr)
    assert w.dist(x, y) == pytest.approx(true_dist, rel=rtol)

    # Only possible for exponent=2
    with pytest.raises(ValueError):
        NumpyTensorSpaceConstWeighting(constant, exponent=1,
                                       dist_using_inner=True)

    # With free function
    w_dist = npy_weighted_dist(constant, use_inner=True)
    assert w_dist(x, y) == pytest.approx(true_dist, rel=rtol)


def test_noweight_init():
    """Test initialization of trivial weighting."""
    w = NumpyTensorSpaceNoWeighting()
    w_same1 = NumpyTensorSpaceNoWeighting()
    w_same2 = NumpyTensorSpaceNoWeighting(2)
    w_same3 = NumpyTensorSpaceNoWeighting(2, False)
    w_same4 = NumpyTensorSpaceNoWeighting(2, dist_using_inner=False)
    w_same5 = NumpyTensorSpaceNoWeighting(exponent=2, dist_using_inner=False)
    w_other_exp = NumpyTensorSpaceNoWeighting(exponent=1)
    w_dist_inner = NumpyTensorSpaceNoWeighting(dist_using_inner=True)

    # Singleton pattern
    for same in (w_same1, w_same2, w_same3, w_same4, w_same5):
        assert w is same

    # Proper creation
    assert w is not w_other_exp
    assert w is not w_dist_inner
    assert w != w_other_exp
    assert w != w_dist_inner


def test_custom_inner(tspace):
    """Test weighting with a custom inner product."""
    rtol = np.sqrt(np.finfo(tspace.dtype).resolution)

    [xarr, yarr], [x, y] = noise_elements(tspace, 2)

    def inner(x, y):
        return np.vdot(y, x)

    w = NumpyTensorSpaceCustomInner(inner)
    w_same = NumpyTensorSpaceCustomInner(inner)
    w_other = NumpyTensorSpaceCustomInner(np.dot)
    w_d = NumpyTensorSpaceCustomInner(inner, dist_using_inner=False)

    assert w == w
    assert w == w_same
    assert w != w_other
    assert w != w_d

    true_inner = inner(xarr, yarr)
    assert w.inner(x, y) == pytest.approx(true_inner)

    true_norm = np.linalg.norm(xarr.ravel())
    assert w.norm(x) == pytest.approx(true_norm)

    true_dist = np.linalg.norm((xarr - yarr).ravel())
    # Uses dist_using_inner by default in this case, therefore tolerance
    assert w.dist(x, y) == pytest.approx(true_dist, rel=rtol)
    assert w_d.dist(x, y) == pytest.approx(true_dist)

    with pytest.raises(TypeError):
        NumpyTensorSpaceCustomInner(1)


def test_custom_norm(tspace):
    """Test weighting with a custom norm."""
    [xarr, yarr], [x, y] = noise_elements(tspace, 2)

    norm = np.linalg.norm

    def other_norm(x):
        return np.linalg.norm(x, ord=1)

    w = NumpyTensorSpaceCustomNorm(norm)
    w_same = NumpyTensorSpaceCustomNorm(norm)
    w_other = NumpyTensorSpaceCustomNorm(other_norm)

    assert w == w
    assert w == w_same
    assert w != w_other

    with pytest.raises(NotImplementedError):
        w.inner(x, y)

    true_norm = np.linalg.norm(xarr.ravel())
    assert w.norm(x) == pytest.approx(true_norm)

    true_dist = np.linalg.norm((xarr - yarr).ravel())
    assert w.dist(x, y) == pytest.approx(true_dist)

    with pytest.raises(TypeError):
        NumpyTensorSpaceCustomNorm(1)


def test_custom_dist(tspace):
    """Test weighting with a custom dist."""
    [xarr, yarr], [x, y] = noise_elements(tspace, 2)

    def dist(x, y):
        return np.linalg.norm(x - y)

    def other_dist(x, y):
        return np.linalg.norm(x - y, ord=1)

    w = NumpyTensorSpaceCustomDist(dist)
    w_same = NumpyTensorSpaceCustomDist(dist)
    w_other = NumpyTensorSpaceCustomDist(other_dist)

    assert w == w
    assert w == w_same
    assert w != w_other

    with pytest.raises(NotImplementedError):
        w.inner(x, y)

    with pytest.raises(NotImplementedError):
        w.norm(x)

    true_dist = np.linalg.norm((xarr - yarr).ravel())
    assert w.dist(x, y) == pytest.approx(true_dist)

    with pytest.raises(TypeError):
        NumpyTensorSpaceCustomDist(1)


# --- Ufuncs & Reductions --- #


def test_ufuncs(tspace, ufunc):
    """Test ufuncs in x.ufunc against direct Numpy ufuncs."""
    name, n_args, n_out, _ = ufunc
    if (np.issubsctype(tspace.dtype, np.floating) or
            np.issubsctype(tspace.dtype, np.complexfloating)and
            name in ['bitwise_and',
                     'bitwise_or',
                     'bitwise_xor',
                     'invert',
                     'left_shift',
                     'right_shift']):
        # Skip integer only methods for floating point data types
        return

    if (np.issubsctype(tspace.dtype, np.complexfloating) and
            name in ['remainder',
                     'trunc',
                     'signbit',
                     'invert',
                     'left_shift',
                     'right_shift',
                     'rad2deg',
                     'deg2rad',
                     'copysign',
                     'mod',
                     'modf',
                     'fmod',
                     'logaddexp2',
                     'logaddexp',
                     'hypot',
                     'arctan2',
                     'floor',
                     'ceil']):
        # Skip real-only methods for complex data types
        return

    # Get the ufunc from NumPy as reference
    npufunc = getattr(np, name)

    # Create some data
    arrays, vectors = noise_elements(tspace, n_args + n_out)
    in_arrays = arrays[:n_args]
    out_arrays = arrays[n_args:]
    data_vector = vectors[0]
    in_vectors = vectors[1:n_args]
    out_vectors = vectors[n_args:]

    # Out-of-place:
    np_result = npufunc(*in_arrays)
    vec_fun = getattr(data_vector.ufunc, name)
    odl_result = vec_fun(*in_vectors)
    assert all_almost_equal(np_result, odl_result)

    # Test type of output
    if n_out == 1:
        assert isinstance(odl_result, tspace.element_type)
    elif n_out > 1:
        for i in range(n_out):
            assert isinstance(odl_result[i], tspace.element_type)

    # In-place:
    np_result = npufunc(*(in_arrays + out_arrays))
    vec_fun = getattr(data_vector.ufunc, name)
    odl_result = vec_fun(*(in_vectors + out_vectors))
    assert all_almost_equal(np_result, odl_result)

    # Test in-place actually holds:
    if n_out == 1:
        assert odl_result is out_vectors[0]
    elif n_out > 1:
        for i in range(n_out):
            assert odl_result[i] is out_vectors[i]


def test_reduction(tspace, reduction):
    """Test reductions in x.ufunc against direct Numpy reduction."""
    name, _ = reduction
    npy_reduction = getattr(np, name)

    x_arr, x = noise_elements(tspace, 1)
    x_reduction = getattr(x.ufunc, name)

    # Should be equal theoretically, but summation order, other stuff, ...,
    # hence we use approx

    # Full reduction, produces scalar
    result_npy = npy_reduction(x_arr)
    result = x_reduction()
    assert result == pytest.approx(result_npy)
    result = x_reduction(axis=(0, 1))
    assert result == pytest.approx(result_npy)

    # Reduction along axes, produces element in reduced space
    result_npy = npy_reduction(x_arr, axis=0)
    result = x_reduction(axis=0)
    assert isinstance(result, NumpyTensor)
    assert result.shape == result_npy.shape
    assert result.dtype == x.dtype
    assert np.allclose(result, result_npy)
    # Check reduced space properties
    assert isinstance(result.space, NumpyTensorSpace)
    assert result.space.exponent == x.space.exponent
    assert result.space.weighting == x.space.weighting  # holds true here
    # Evaluate in-place
    out = result.space.element()
    x_reduction(axis=0, out=out)
    assert np.allclose(out, result_npy)

    # Use keepdims parameter
    result_npy = npy_reduction(x_arr, axis=1, keepdims=True)
    result = x_reduction(axis=1, keepdims=True)
    assert result.shape == result_npy.shape
    assert np.allclose(result, result_npy)
    # Evaluate in-place
    out = result.space.element()
    x_reduction(axis=1, keepdims=True, out=out)
    assert np.allclose(out, result_npy)

    # Use dtype parameter
    # These reductions have a `dtype` parameter
    if name in ('cumprod', 'cumsum', 'mean', 'prod', 'std', 'sum',
                'trace', 'var'):
        result_npy = npy_reduction(x_arr, axis=1, dtype='complex64')
        result = x_reduction(axis=1, dtype='complex64')
        assert result.dtype == np.dtype('complex64')
        assert np.allclose(result, result_npy)
        # Evaluate in-place
        out = result.space.element()
        x_reduction(axis=1, dtype='complex64', out=out)
        assert np.allclose(out, result_npy)


def test_reduction_with_weighting():
    """Weightings are tricky to handle, check some cases."""
    # Constant weighting, should propagate
    tspace = odl.rtensors((3, 4), weight=0.5)
    x = tspace.one()
    red = x.ufuncs.sum(axis=0)
    assert red.space.weighting == tspace.weighting

    # Array weighting, should result in no weighting
    weight_arr = np.ones((3, 4)) * 0.5
    tspace = odl.rtensors((3, 4), weight=weight_arr, exponent=1.5)
    x = tspace.one()
    red = x.ufuncs.sum(axis=0)
    assert red.space.weighting == NumpyTensorSpaceNoWeighting(exponent=1.5)


def test_ufunc_reduction_docs_notempty():
    """Check that the generated docstrings are not empty."""
    for _, __, ___, doc in UFUNCS:
        assert doc.splitlines()[0] != ''

    for _, doc in REDUCTIONS:
        assert doc.splitlines()[0] != ''


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
