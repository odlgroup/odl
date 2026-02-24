# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for `discr_ops`."""

from __future__ import division

import numpy as np
import pytest

import odl
from odl.core.discr.discr_ops import _SUPPORTED_RESIZE_PAD_MODES
from odl.core.util.testutils import dtype_tol, noise_element, all_equal

from odl.core.util.dtype_utils import AVAILABLE_DTYPES, SCALAR_DTYPES, FLOAT_DTYPES, REAL_DTYPES
# --- pytest fixtures --- #


paddings = list(_SUPPORTED_RESIZE_PAD_MODES)
paddings.remove('constant')
paddings.extend([('constant', 0), ('constant', 1)])
padding_ids = [" pad_mode='{}'-{} ".format(*p)
               if isinstance(p, tuple)
               else " pad_mode='{}' ".format(p)
               for p in paddings]


@pytest.fixture(scope="module", ids=padding_ids, params=paddings)
def padding(request):
    if isinstance(request.param, tuple):
        pad_mode, pad_const = request.param
    else:
        pad_mode = request.param
        pad_const = 0

    return pad_mode, pad_const


# --- ResizingOperator tests --- #


def test_resizing_op_init(odl_impl_device_pairs, padding):
    # Test if the different init patterns run
    impl, device = odl_impl_device_pairs
    pad_mode, pad_const = padding

    space = odl.uniform_discr(
        [0, -1], [1, 1], (10, 5), impl=impl, device=device
        )
    res_space = odl.uniform_discr(
        [0, -3], [2, 3], (20, 15), impl=impl, device=device
    )

    odl.ResizingOperator(space, res_space)
    odl.ResizingOperator(space, ran_shp=(20, 15))
    odl.ResizingOperator(space, ran_shp=(20, 15), offset=(0, 5))
    odl.ResizingOperator(space, ran_shp=(20, 15), pad_mode=pad_mode)
    odl.ResizingOperator(space, ran_shp=(20, 15), pad_mode=pad_mode,
                         pad_const=pad_const)
    odl.ResizingOperator(space, ran_shp=(20, 15),
                         discr_kwargs={'nodes_on_bdry': True})


def test_resizing_op_raise(odl_impl_device_pairs):
    """Validate error checking in ResizingOperator."""
    # Domain not a uniformly discretized Lp
    with pytest.raises(TypeError):
        odl.ResizingOperator(odl.rn(5), ran_shp=(10,))

    grid = odl.RectGrid([0, 2, 3])
    part = odl.RectPartition(odl.IntervalProd(0, 3), grid)

    impl, device = odl_impl_device_pairs

    tspace = odl.rn(3, impl=impl, device=device)
    space = odl.DiscretizedSpace(part, tspace)
    with pytest.raises(ValueError):
        odl.ResizingOperator(space, ran_shp=(10,))

    # Different cell sides in domain and range
    space = odl.uniform_discr(0, 1, 10, impl=impl, device=device)
    res_space = odl.uniform_discr(0, 1, 15, impl=impl, device=device)
    with pytest.raises(ValueError):
        odl.ResizingOperator(space, res_space)

    # Non-integer multiple of cell sides used as shift (grid of the
    # resized space shifted)
    space = odl.uniform_discr(0, 1, 5, impl=impl, device=device)
    res_space = odl.uniform_discr(-0.5, 1.5, 10, impl=impl, device=device)
    with pytest.raises(ValueError):
        odl.ResizingOperator(space, res_space)

    # Need either range or ran_shp
    with pytest.raises(ValueError):
        odl.ResizingOperator(space)

    # Offset cannot be combined with range
    space = odl.uniform_discr([0, -1], [1, 1], (10, 5), impl=impl, device=device)
    res_space = odl.uniform_discr([0, -3], [2, 3], (20, 15), impl=impl, device=device)
    with pytest.raises(ValueError):
        odl.ResizingOperator(space, res_space, offset=(0, 0))

    # Bad pad_mode
    with pytest.raises(ValueError):
        odl.ResizingOperator(space, res_space, pad_mode='something')


def test_resizing_op_properties(odl_impl_device_pairs, padding):

    impl, device = odl_impl_device_pairs

    pad_mode, pad_const = padding

    for dtype in SCALAR_DTYPES:
        # Explicit range
        space = odl.uniform_discr([0, -1], [1, 1], (10, 5), dtype=dtype, impl=impl, device=device)
        res_space = odl.uniform_discr([0, -3], [2, 3], (20, 15), dtype=dtype, impl=impl, device=device)
        res_op = odl.ResizingOperator(space, res_space, pad_mode=pad_mode,
                                      pad_const=pad_const)

        assert res_op.domain == space
        assert res_op.range == res_space
        assert res_op.offset == (0, 5)
        assert res_op.pad_mode == pad_mode
        assert res_op.pad_const == pad_const
        if pad_mode == 'constant' and pad_const != 0:
            assert not res_op.is_linear
        else:
            assert res_op.is_linear

        # Implicit range via ran_shp and offset
        res_op = odl.ResizingOperator(space, ran_shp=(20, 15), offset=[0, 5],
                                      pad_mode=pad_mode, pad_const=pad_const)
        assert np.allclose(res_op.range.min_pt, res_space.min_pt)
        assert np.allclose(res_op.range.max_pt, res_space.max_pt)
        assert np.allclose(res_op.range.cell_sides, res_space.cell_sides)
        assert res_op.range.dtype == res_space.dtype
        assert res_op.offset == (0, 5)
        assert res_op.pad_mode == pad_mode
        assert res_op.pad_const == pad_const
        if pad_mode == 'constant' and pad_const != 0:
            assert not res_op.is_linear
        else:
            assert res_op.is_linear


def test_resizing_op_call(odl_impl_device_pairs):

    impl, device = odl_impl_device_pairs
    
    for dtype in AVAILABLE_DTYPES:
        # Minimal test since this operator only wraps resize_array
        space = odl.uniform_discr(
            [0, -1], [1, 1], (4, 5), dtype=dtype, impl=impl, device=device
        )
        res_space = odl.uniform_discr(
            [0, -0.6], [2, 0.2], (8, 2), dtype=dtype, impl=impl, device=device
        )
        res_op = odl.ResizingOperator(space, res_space)
        out = res_op(space.one())
        true_res = np.zeros((8, 2), dtype=dtype)
        true_res[:4, :] = 1
        assert all_equal(out, true_res)

        if res_space.operation_paradigms.in_place.is_supported:
            out = res_space.element()
            res_op(space.one(), out=out)
            assert all_equal(out, true_res)

        # Test also mapping to default impl for other 'impl'
        # if impl != 'numpy':
        #     space = odl.uniform_discr(
        #         [0, -1], [1, 1], (4, 5), dtype=dtype, impl=impl
        #     )
        #     res_space = odl.uniform_discr(
        #         [0, -0.6], [2, 0.2], (8, 2), dtype=dtype
        #     )
        #     res_op = odl.ResizingOperator(space, res_space)
        #     out = res_op(space.one())
        #     true_res = np.zeros((8, 2), dtype=dtype)
        #     true_res[:4, :] = 1
        #     assert all_equal(out, true_res)

        #     out = res_space.element()
        #     res_op(space.one(), out=out)
        #     assert all_equal(out, true_res)


def test_resizing_op_deriv(padding, odl_impl_device_pairs):

    impl, device = odl_impl_device_pairs

    pad_mode, pad_const = padding
    space = odl.uniform_discr(
        [0, -1], [1, 1], (4, 5), impl=impl, device=device
        )
    res_space = odl.uniform_discr(
        [0, -0.6], [2, 0.2], (8, 2), impl=impl, device=device
        )
    res_op = odl.ResizingOperator(space, res_space, pad_mode=pad_mode,
                                  pad_const=pad_const)
    res_op_deriv = res_op.derivative(space.one())

    if pad_mode == 'constant' and pad_const != 0:
        # Only non-trivial case is constant padding with const != 0
        assert res_op_deriv.pad_mode == 'constant'
        assert res_op_deriv.pad_const == 0.0
    else:
        assert res_op_deriv is res_op


def test_resizing_op_inverse(padding, odl_impl_device_pairs):

    impl, device = odl_impl_device_pairs
    pad_mode, pad_const = padding

    for dtype in SCALAR_DTYPES:

        if pad_mode == 'order1' and (
                np.issubdtype(dtype, np.unsignedinteger)
                or np.issubdtype(dtype, np.timedelta64()) ):
            # Extrapolating a trend might lead to negative values, which  
            # will raise an error for unsigned integers. For timedeltas, 
            # it would involve a multiplication of two times which was 
            # allowed by numpy 1 but is not allowed in numpy 2.
            continue

        space = odl.uniform_discr([0, -1], [1, 1], (4, 5), dtype=dtype,
                                  impl=impl, device=device)
        res_space = odl.uniform_discr([0, -1.4], [1.5, 1.4], (6, 7),
                                      dtype=dtype, impl=impl, device=device)
        res_op = odl.ResizingOperator(space, res_space, pad_mode=pad_mode,
                                      pad_const=pad_const)

        # Only left inverse if the operator extends in all axes
        x = noise_element(space)
        assert res_op.inverse(res_op(x)) == x


def test_resizing_op_adjoint(padding, odl_impl_device_pairs):

    impl, device = odl_impl_device_pairs
    pad_mode, pad_const = padding
    for dtype in FLOAT_DTYPES:
        space = odl.uniform_discr([0, -1], [1, 1], (4, 5), dtype=dtype,
                                  impl=impl, device=device)
        res_space = odl.uniform_discr([0, -1.4], [1.5, 1.4], (6, 7),
                                      dtype=dtype, impl=impl, device=device)
        res_op = odl.ResizingOperator(space, res_space, pad_mode=pad_mode,
                                      pad_const=pad_const)

        if pad_const != 0.0:
            with pytest.raises(NotImplementedError):
                res_op.adjoint
            return

        elem = noise_element(space)
        res_elem = noise_element(res_space)
        inner1 = res_op(elem).inner(res_elem)
        inner2 = elem.inner(res_op.adjoint(res_elem))
        assert inner1 == pytest.approx(
            inner2,
            abs = 1e-2 * space.size * dtype_tol(dtype) * elem.norm() * res_elem.norm())


def test_resizing_op_mixed_uni_nonuni(odl_impl_device_pairs):
    """Check if resizing along uniform axes in mixed discretizations works."""

    impl, device = odl_impl_device_pairs

    nonuni_part = odl.nonuniform_partition([0, 1, 4])
    uni_part = odl.uniform_partition(-1, 1, 4)
    part = uni_part.append(nonuni_part, uni_part, nonuni_part)
    tspace = odl.rn(part.shape, impl=impl, device=device)
    space = odl.DiscretizedSpace(part, tspace)

    # Keep non-uniform axes fixed
    res_op = odl.ResizingOperator(space, ran_shp=(6, 3, 6, 3))

    assert res_op.axes == (0, 2)
    assert res_op.offset == (1, 0, 1, 0)

    # Evaluation test with a simpler case
    part = uni_part.append(nonuni_part)
    tspace = odl.rn(part.shape, impl=impl, device=device)
    space = odl.DiscretizedSpace(part, tspace)
    res_op = odl.ResizingOperator(space, ran_shp=(6, 3))
    result = res_op(space.one())
    true_result = [[0, 0, 0],
                   [1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1],
                   [0, 0, 0]]
    assert all_equal(result, true_result)

    # Test adjoint
    elem = noise_element(space)
    res_elem = noise_element(res_op.range)
    inner1 = res_op(elem).inner(res_elem)
    inner2 = elem.inner(res_op.adjoint(res_elem))
    assert inner1 == pytest.approx(inner2)


if __name__ == '__main__':
    odl.core.util.test_file(__file__)
