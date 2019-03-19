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
import pytest

import odl
from odl.discr.discr_space import DiscretizedSpace, DiscretizedSpaceElement
from odl.space.base_tensors import TensorSpace
from odl.util.testutils import all_almost_equal, all_equal, simple_fixture


# --- Pytest fixtures --- #


exponent = simple_fixture('exponent', [2.0, 1.0, float('inf'), 0.5, 1.5])
shape = simple_fixture('shape', [(2, 3, 4), (3, 4), (2,), (1,), (1, 1, 1)])


# --- DiscretizedSpace --- #


def test_discretizedspace_init():
    """Test initialization and basic properties of DiscretizedSpace."""
    # Real space
    part = odl.uniform_partition([0, 0], [1, 1], (2, 4))
    tspace = odl.rn(part.shape)

    discr = DiscretizedSpace(part, tspace)
    assert discr.tspace == tspace
    assert discr.partition == part
    assert discr.exponent == tspace.exponent
    assert discr.axis_labels == ('$x$', '$y$')
    assert discr.is_real

    # Complex space
    tspace_c = odl.cn(part.shape)
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


def test_empty():
    """Check if empty spaces behave as expected and all methods work."""
    discr = odl.uniform_discr([], [], ())

    assert discr.axis_labels == ()
    assert discr.tangent_bundle == odl.ProductSpace(field=odl.RealNumbers())
    assert discr.complex_space == odl.uniform_discr([], [], (), dtype=complex)
    hash(discr)
    assert repr(discr) != ''

    elem = discr.element(1.0)
    assert elem.shape == ()
    assert elem.size == 1
    assert elem == 1.0


# --- uniform_discr --- #


def test_uniform_discr_dtypes(odl_tspace_impl):
    """Check the uniform_discr factory function wrt dtypes."""
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
            assert discr.tspace.element().dtype == dtype

    for dtype in complex_float_dtypes:
        try:
            discr = odl.uniform_discr(0, 1, 10, impl=impl, dtype=dtype)
        except TypeError:
            continue
        else:
            assert isinstance(discr.tspace, TensorSpace)
            assert discr.tspace.impl == impl
            assert discr.is_complex
            assert discr.tspace.element().dtype == dtype


def test_uniform_discr_init_real(odl_tspace_impl):
    """Test initialization and basic properties with uniform_discr, real."""
    impl = odl_tspace_impl

    # 1D
    discr = odl.uniform_discr(0, 1, 10, impl=impl)
    assert isinstance(discr, DiscretizedSpace)
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


# --- DiscretizedSpace methods --- #


def test_discretizedspace_element(odl_elem_order):
    """Test creation and membership of DiscretizedSpace elements."""
    order = odl_elem_order

    # 1D
    space = odl.uniform_discr(0, 1, 3)
    assert space.element() in space
    elem = space.element([1, 2, 3])
    assert np.array_equal(elem, [1, 2, 3])

    other_space = odl.uniform_discr(0, 1, 4)
    assert other_space.element() not in space
    other_space = odl.uniform_discr(0, 1, 3, dtype=complex)
    assert other_space.element() not in space

    # 2D
    space = odl.uniform_discr([0, 0], [1, 1], (2, 2))
    assert space.element() in space

    elem = space.element([[1, 2],
                          [3, 4]], order=order)
    assert np.array_equal(elem, [[1, 2],
                                 [3, 4]])

    if order is None:
        assert elem.flags[space.default_order + '_CONTIGUOUS']
    else:
        assert elem.flags[order + '_CONTIGUOUS']

    with pytest.raises(ValueError):
        space.element([1, 2, 3])  # wrong size & shape
    with pytest.raises(ValueError):
        space.element([1, 2, 3, 4])  # wrong shape
    with pytest.raises(ValueError):
        space.element([[1],
                       [2],
                       [3],
                       [4]])  # wrong shape


def test_element_from_function_1d():
    """Test creation of DiscretizedSpace elements from functions in 1D."""
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
    true_elem = [1.0 for _ in points]
    assert all_equal(elem_lam, true_elem)


def test_element_from_function_2d():
    """Test creation of DiscretizedSpace elements from functions in 2D."""
    space = odl.uniform_discr([-1, -1], [1, 1], (2, 3))
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


def test_discretizedspace_zero_one():
    """Test the zero and one element creators of DiscretizedSpace."""
    discr = odl.uniform_discr(0, 1, 3)

    zero = discr.zero()
    assert zero in discr
    assert np.array_equal(zero, [0, 0, 0])

    one = discr.one()
    assert one in discr
    assert np.array_equal(one, [1, 1, 1])


def test_discretizedspace_equals(exponent, odl_tspace_impl):
    """Check equality testing between spaces."""
    impl = odl_tspace_impl
    space1 = odl.uniform_discr(0, 1, 3, exponent=exponent, impl=impl)
    space2 = odl.uniform_discr(0, 1, 3, exponent=exponent, impl=impl)
    other_space = odl.uniform_discr(0, 1, 4, exponent=exponent, impl=impl)

    assert space1 == space1
    assert space1 == space2
    assert space1 != other_space
    assert hash(space1) == hash(space2)
    assert hash(space1) != hash(other_space)


def test_discretizedspace_cell_sides():
    """Check correctness of cell_sides."""
    # Non-degenerated case, should be same as cell size
    space = odl.uniform_discr([0, 0], [1, 1], [2, 2])
    assert all_equal(space.cell_sides, [0.5] * 2)

    # Degenerated case, uses interval size in 1-point dimensions
    space = odl.uniform_discr([0, 0], [1, 1], [2, 1])
    assert all_equal(space.cell_sides, [0.5, 1])


def test_discretizedspace_cell_volume():
    """Check correctness of cell_volume."""
    # Non-degenerated case
    space = odl.uniform_discr([0, 0], [1, 1], [2, 2])
    assert space.cell_volume == 0.25

    # Degenerated case, uses interval size in 1-point dimensions
    space = odl.uniform_discr([0, 0], [1, 1], [2, 1])
    assert space.cell_volume == 0.5


def test_discretizedspace_astype():
    """Check conversion of spaces using astype()."""
    rspace = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype='float64')
    cspace = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype='complex128')
    rspace_s = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype='float32')
    cspace_s = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype='complex64')

    # Real
    assert rspace.astype('float32') == rspace_s
    assert rspace.astype('float64') is rspace
    assert rspace.real_space is rspace
    assert rspace.astype('complex64') == cspace_s
    assert rspace.astype('complex128') == cspace
    assert rspace.complex_space == cspace

    # Complex
    assert cspace.astype('complex64') == cspace_s
    assert cspace.astype('complex128') is cspace
    assert cspace.complex_space is cspace
    assert cspace.astype('float32') == rspace_s
    assert cspace.astype('float64') == rspace
    assert cspace.real_space == rspace

    # More exotic dtype
    space = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype=bool)
    as_float = space.astype(float)
    assert as_float.dtype == float
    assert not as_float.is_weighted
    as_complex = space.astype(complex)
    assert as_complex.dtype == complex
    assert not as_complex.is_weighted


def test_real_imag(odl_tspace_impl, odl_elem_order):
    """Check if real and imaginary parts can be read and written to."""
    impl = odl_tspace_impl
    order = odl_elem_order
    tspace_cls = odl.space.entry_points.tensor_space_impl(impl)
    for dtype in filter(odl.util.is_complex_floating_dtype,
                        tspace_cls.available_dtypes()):
        cspace = odl.uniform_discr([0, 0], [1, 1], [2, 2], dtype=dtype,
                                   impl=impl)
        rspace = cspace.real_space

        x = cspace.element([[1 - 1j, 2 - 2j],
                            [3 - 3j, 4 - 4j]], order=order)
        assert x.real in rspace
        assert x.imag in rspace


def test_inner_nonuniform():
    """Check if inner products are correct in non-uniform discretizations."""
    part = odl.nonuniform_partition([0, 2, 3, 5], min_pt=0, max_pt=5)
    weights = part.cell_sizes_vecs[0]
    tspace = odl.rn(part.size, weighting=weights)
    space = odl.DiscretizedSpace(part, tspace)

    linear = space.element(lambda x: x)

    # Exact inner product is the integral from 0 to 5 of x, which is 5**2 / 2
    exact_inner = 5 ** 2 / 2.0
    assert space.inner(space.one(), linear) == pytest.approx(exact_inner)


def test_norm_nonuniform():
    """Check if norms are correct in non-uniform discretizations."""
    part = odl.nonuniform_partition([0, 2, 3, 5], min_pt=0, max_pt=5)
    weights = part.cell_sizes_vecs[0]
    tspace = odl.rn(part.size, weighting=weights)
    space = odl.DiscretizedSpace(part, tspace)

    sqrt = space.element(lambda x: np.sqrt(x))

    # Exact norm is the square root of the integral from 0 to 5 of x,
    # which is sqrt(5**2 / 2)
    exact_norm = np.sqrt(5 ** 2 / 2.0)
    assert space.norm(sqrt) == pytest.approx(exact_norm)


def test_norm_interval(exponent):
    """Check norm computation on a 1D interval."""
    # Test the function f(x) = x^2 on the interval (0, 1). Its
    # L^p-norm is (1 + 2*p)^(-1/p) for finite p and 1 for p=inf
    p = exponent
    space = odl.uniform_discr(0, 1, 10, exponent=p)

    func = space.element(lambda x: x ** 2)
    if p == float('inf'):
        assert space.norm(func) <= 1  # Max at boundary not hit
    else:
        true_norm = (1 + 2 * p) ** (-1 / p)
        assert space.norm(func) == pytest.approx(true_norm, rel=1e-2)


def test_norm_rectangle(exponent):
    """Check norm computation on a 2D rectangle."""
    # Test the function f(x) = x_0^2 * x_1^3 on (0, 1) x (-1, 1). Its
    # L^p-norm is ((1 + 2*p) * (1 + 3 * p) / 2)^(-1/p) for finite p
    # and 1 for p=inf
    p = exponent
    space = odl.uniform_discr([0, -1], [1, 1], (20, 30), exponent=p)

    func = space.element(lambda x: x[0] ** 2 * x[1] ** 3)
    if p == float('inf'):
        assert space.norm(func) <= 1  # Max at boundary not hit
    else:
        true_norm = ((1 + 2 * p) * (1 + 3 * p) / 2) ** (-1 / p)
        assert space.norm(func) == pytest.approx(true_norm, rel=1e-2)


def test_norm_rectangle_boundary(odl_tspace_impl, exponent):
    """Check norm computation with different boundary settings.

    This test uses the constant function ``f(x) = 1`` to check whether the
    boundary correction for the norm reproduces the volume of the spatial
    domain as the correct norm of ``f``.
    """
    impl = odl_tspace_impl
    dtype = 'float32'

    # Standard case
    space = odl.uniform_discr(
        [-1, -2], [1, 2], (4, 8), dtype=dtype, impl=impl, exponent=exponent
    )
    if exponent == float('inf'):
        assert space.norm(space.one()) == 1
    else:
        assert (
            space.norm(space.one())
            == pytest.approx(space.domain.volume ** (1 / exponent))
        )

    # Nodes on the boundary (everywhere)
    space = odl.uniform_discr(
        [-1, -2], [1, 2], (4, 8), dtype=dtype, impl=impl, exponent=exponent,
        nodes_on_bdry=True
    )
    if exponent == float('inf'):
        assert space.norm(space.one()) == 1
    else:
        assert (
            space.norm(space.one())
            == pytest.approx(space.domain.volume ** (1 / exponent))
        )

    # Nodes on the boundary (selective)
    space = odl.uniform_discr(
        [-1, -2], [1, 2], (4, 8), dtype=dtype, impl=impl, exponent=exponent,
        nodes_on_bdry=((False, True), False)
    )
    if exponent == float('inf'):
        assert space.norm(space.one()) == 1
    else:
        assert (
            space.norm(space.one())
            == pytest.approx(space.domain.volume ** (1 / exponent))
        )

    space = odl.uniform_discr(
        [-1, -2], [1, 2], (4, 8), dtype=dtype, impl=impl, exponent=exponent,
        nodes_on_bdry=(False, (True, False))
    )
    if exponent == float('inf'):
        assert space.norm(space.one()) == 1
    else:
        assert (
            space.norm(space.one())
            == pytest.approx(space.domain.volume ** (1 / exponent))
        )

    # Completely arbitrary boundary
    part = odl.RectPartition(
        odl.IntervalProd([-1, -2], [1, 2]),
        odl.uniform_grid([0, 0], [1, 1], (4, 4))
    )
    weight = 1.0 if exponent == float('inf') else part.cell_volume
    tspace = odl.rn(part.shape, dtype=dtype, impl=impl,
                    exponent=exponent, weighting=weight)
    space = DiscretizedSpace(part, tspace)

    if exponent == float('inf'):
        assert space.norm(space.one()) == 1
    else:
        assert (
            space.norm(space.one())
            == pytest.approx(space.domain.volume ** (1 / exponent))
        )


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
