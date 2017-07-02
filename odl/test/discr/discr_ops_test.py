# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for `discr_ops`."""

from __future__ import division
import pytest
import numpy as np

import odl
from odl.discr.discr_ops import _SUPPORTED_RESIZE_PAD_MODES
from odl.space.entry_points import TENSOR_SPACE_IMPLS
from odl.util import is_numeric_dtype, is_real_floating_dtype
from odl.util.testutils import almost_equal, noise_element, dtype_places


# --- pytest fixtures --- #


paddings = list(_SUPPORTED_RESIZE_PAD_MODES)
paddings.remove('constant')
paddings.extend([('constant', 0), ('constant', 1)])
padding_ids = [" pad_mode = '{}' {} ".format(*p)
               if isinstance(p, tuple)
               else " pad_mode = '{}' ".format(p)
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


def test_resizing_op_init(tspace_impl, padding):

    # Test if the different init patterns run

    pad_mode, pad_const = padding

    space = odl.uniform_discr([0, -1], [1, 1], (10, 5), impl=tspace_impl)
    res_space = odl.uniform_discr([0, -3], [2, 3], (20, 15), impl=tspace_impl)

    odl.ResizingOperator(space, res_space)
    odl.ResizingOperator(space, ran_shp=(20, 15))
    odl.ResizingOperator(space, ran_shp=(20, 15), offset=(0, 5))
    odl.ResizingOperator(space, ran_shp=(20, 15), pad_mode=pad_mode)
    odl.ResizingOperator(space, ran_shp=(20, 15), pad_mode=pad_mode,
                         pad_const=pad_const)
    odl.ResizingOperator(space, ran_shp=(20, 15),
                         discr_kwargs={'nodes_on_bdry': True})


def test_resizing_op_raise():
    # domain not a uniformely discretized Lp
    with pytest.raises(TypeError):
        odl.ResizingOperator(odl.rn(5), ran_shp=(10,))

    grid = odl.RectGrid([0, 2, 3])
    part = odl.RectPartition(odl.IntervalProd(0, 3), grid)
    fspace = odl.FunctionSpace(odl.IntervalProd(0, 3))
    dspace = odl.rn(3)
    space = odl.DiscreteLp(fspace, part, dspace)
    with pytest.raises(NotImplementedError):
        odl.ResizingOperator(space, ran_shp=(10,))

    # different cell sides in domain and range
    space = odl.uniform_discr(0, 1, 10)
    res_space = odl.uniform_discr(0, 1, 15)
    with pytest.raises(ValueError):
        odl.ResizingOperator(space, res_space)

    # non-integer multiple of cell sides used as shift (grid of the
    # resized space shifted)
    space = odl.uniform_discr(0, 1, 5)
    res_space = odl.uniform_discr(-0.5, 1.5, 10)
    with pytest.raises(ValueError):
        odl.ResizingOperator(space, res_space)

    # need either range or ran_shp
    with pytest.raises(ValueError):
        odl.ResizingOperator(space)

    # offset cannot be combined with range
    space = odl.uniform_discr([0, -1], [1, 1], (10, 5))
    res_space = odl.uniform_discr([0, -3], [2, 3], (20, 15))
    with pytest.raises(ValueError):
        odl.ResizingOperator(space, res_space, offset=(0, 0))

    # bad pad_mode
    with pytest.raises(ValueError):
        odl.ResizingOperator(space, res_space, pad_mode='something')


def test_resizing_op_properties(tspace_impl, padding):

    dtypes = [dt for dt in TENSOR_SPACE_IMPLS[tspace_impl].available_dtypes()
              if is_numeric_dtype(dt)]

    pad_mode, pad_const = padding

    for dtype in dtypes:
        # Explicit range
        space = odl.uniform_discr([0, -1], [1, 1], (10, 5), dtype=dtype)
        res_space = odl.uniform_discr([0, -3], [2, 3], (20, 15), dtype=dtype)
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


def test_resizing_op_call(tspace_impl):

    dtypes = [dt for dt in TENSOR_SPACE_IMPLS[tspace_impl].available_dtypes()
              if is_numeric_dtype(dt)]

    for dtype in dtypes:
        # Minimal test since this operator only wraps resize_array
        space = odl.uniform_discr([0, -1], [1, 1], (4, 5), impl=tspace_impl)
        res_space = odl.uniform_discr([0, -0.6], [2, 0.2], (8, 2),
                                      impl=tspace_impl)
        res_op = odl.ResizingOperator(space, res_space)
        out = res_op(space.one())
        true_res = np.zeros((8, 2))
        true_res[:4, :] = 1
        assert np.array_equal(out, true_res)

        out = res_space.element()
        res_op(space.one(), out=out)
        assert np.array_equal(out, true_res)

        # Test also mapping to default impl for other 'tspace_impl'
        if tspace_impl != 'numpy':
            space = odl.uniform_discr([0, -1], [1, 1], (4, 5),
                                      impl=tspace_impl)
            res_space = odl.uniform_discr([0, -0.6], [2, 0.2], (8, 2))
            res_op = odl.ResizingOperator(space, res_space)
            out = res_op(space.one())
            true_res = np.zeros((8, 2))
            true_res[:4, :] = 1
            assert np.array_equal(out, true_res)

            out = res_space.element()
            res_op(space.one(), out=out)
            assert np.array_equal(out, true_res)


def test_resizing_op_deriv(padding):
    pad_mode, pad_const = padding
    space = odl.uniform_discr([0, -1], [1, 1], (4, 5))
    res_space = odl.uniform_discr([0, -0.6], [2, 0.2], (8, 2))
    res_op = odl.ResizingOperator(space, res_space, pad_mode=pad_mode,
                                  pad_const=pad_const)
    res_op_deriv = res_op.derivative(space.one())

    if pad_mode == 'constant' and pad_const != 0:
        # Only non-trivial case is constant padding with const != 0
        assert res_op_deriv.pad_mode == 'constant'
        assert res_op_deriv.pad_const == 0.0
    else:
        assert res_op_deriv is res_op


def test_resizing_op_inverse(padding, tspace_impl):

    pad_mode, pad_const = padding
    dtypes = [dt for dt in TENSOR_SPACE_IMPLS[tspace_impl].available_dtypes()
              if is_numeric_dtype(dt)]

    for dtype in dtypes:
        space = odl.uniform_discr([0, -1], [1, 1], (4, 5), dtype=dtype,
                                  impl=tspace_impl)
        res_space = odl.uniform_discr([0, -1.4], [1.5, 1.4], (6, 7),
                                      dtype=dtype, impl=tspace_impl)
        res_op = odl.ResizingOperator(space, res_space, pad_mode=pad_mode,
                                      pad_const=pad_const)

        # Only left inverse if the operator extentds in all axes
        x = noise_element(space)
        assert res_op.inverse(res_op(x)) == x


def test_resizing_op_adjoint(padding, tspace_impl):

    pad_mode, pad_const = padding
    dtypes = [dt for dt in TENSOR_SPACE_IMPLS[tspace_impl].available_dtypes()
              if is_real_floating_dtype(dt)]

    for dtype in dtypes:
        space = odl.uniform_discr([0, -1], [1, 1], (4, 5), dtype=dtype,
                                  impl=tspace_impl)
        res_space = odl.uniform_discr([0, -1.4], [1.5, 1.4], (6, 7),
                                      dtype=dtype, impl=tspace_impl)
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
        assert almost_equal(inner1, inner2, places=dtype_places(dtype))


def test_resizing_op_mixed_uni_nonuni():
    """Check if resizing along uniform axes in mixed discretizations works."""
    nonuni_part = odl.nonuniform_partition([0, 1, 4])
    uni_part = odl.uniform_partition(-1, 1, 4)
    part = uni_part.append(nonuni_part, uni_part, nonuni_part)
    fspace = odl.FunctionSpace(odl.IntervalProd(part.min_pt, part.max_pt))
    dspace = odl.rn(part.size)
    space = odl.DiscreteLp(fspace, part, dspace)

    # Keep non-uniform axes fixed
    res_op = odl.ResizingOperator(space, ran_shp=(6, 3, 6, 3))

    assert res_op.axes == (0, 2)
    assert res_op.offset == (1, 0, 1, 0)

    # Evaluation test with a simpler case
    part = uni_part.append(nonuni_part)
    fspace = odl.FunctionSpace(odl.IntervalProd(part.min_pt, part.max_pt))
    dspace = odl.rn(part.size)
    space = odl.DiscreteLp(fspace, part, dspace)
    res_op = odl.ResizingOperator(space, ran_shp=(6, 3))
    result = res_op(space.one())
    true_result = [[0, 0, 0],
                   [1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1],
                   [0, 0, 0]]
    assert np.array_equal(result, true_result)

    # Test adjoint
    elem = noise_element(space)
    res_elem = noise_element(res_op.range)
    inner1 = res_op(elem).inner(res_elem)
    inner2 = elem.inner(res_op.adjoint(res_elem))
    assert almost_equal(inner1, inner2)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
