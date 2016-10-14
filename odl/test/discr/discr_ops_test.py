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

"""Unit tests for `discr_ops`."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import pytest
import numpy as np

import odl
from odl.discr.discr_ops import _SUPPORTED_RESIZE_PAD_MODES
from odl.util.testutils import almost_equal, noise_element, dtype_places
from odl.util.utility import is_scalar_dtype, is_real_floating_dtype


# --- ResizingOperator --- #


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


def test_resizing_op_init(fn_impl, padding):

    # Test if the different init patterns run

    pad_mode, pad_const = padding

    space = odl.uniform_discr([0, -1], [1, 1], (10, 5), impl=fn_impl)
    res_space = odl.uniform_discr([0, -3], [2, 3], (20, 15), impl=fn_impl)

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

    grid = odl.TensorGrid([0, 2, 3])
    part = odl.RectPartition(odl.IntervalProd(0, 3), grid)
    fspace = odl.FunctionSpace(odl.IntervalProd(0, 3))
    dspace = odl.rn(3)
    space = odl.DiscreteLp(fspace, part, dspace)
    with pytest.raises(ValueError):
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


def test_resizing_op_properties(fn_impl, padding):

    dtypes = [dt for dt in odl.FN_IMPLS[fn_impl].available_dtypes()
              if is_scalar_dtype(dt)]

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


def test_resizing_op_call(fn_impl):

    dtypes = [dt for dt in odl.FN_IMPLS[fn_impl].available_dtypes()
              if is_scalar_dtype(dt)]

    for dtype in dtypes:
        # Minimal test since this operator only wraps resize_array
        space = odl.uniform_discr([0, -1], [1, 1], (4, 5), impl=fn_impl)
        res_space = odl.uniform_discr([0, -0.6], [2, 0.2], (8, 2),
                                      impl=fn_impl)
        res_op = odl.ResizingOperator(space, res_space)
        out = res_op(space.one())
        true_res = np.zeros((8, 2))
        true_res[:4, :] = 1
        assert np.array_equal(out, true_res)

        out = res_space.element()
        res_op(space.one(), out=out)
        assert np.array_equal(out, true_res)

        # Test also mapping to default impl for other 'fn_impl'
        if fn_impl != 'numpy':
            space = odl.uniform_discr([0, -1], [1, 1], (4, 5), impl=fn_impl)
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


def test_resizing_op_inverse(padding, fn_impl):

    pad_mode, pad_const = padding
    dtypes = [dt for dt in odl.FN_IMPLS[fn_impl].available_dtypes()
              if is_scalar_dtype(dt)]

    for dtype in dtypes:
        space = odl.uniform_discr([0, -1], [1, 1], (4, 5), dtype=dtype,
                                  impl=fn_impl)
        res_space = odl.uniform_discr([0, -1.4], [1.5, 1.4], (6, 7),
                                      dtype=dtype, impl=fn_impl)
        res_op = odl.ResizingOperator(space, res_space, pad_mode=pad_mode,
                                      pad_const=pad_const)

        # Only left inverse if the operator extentds in all axes
        x = noise_element(space)
        assert res_op.inverse(res_op(x)) == x


def test_resizing_op_adjoint(padding, fn_impl):

    pad_mode, pad_const = padding
    dtypes = [dt for dt in odl.FN_IMPLS[fn_impl].available_dtypes()
              if is_real_floating_dtype(dt)]

    for dtype in dtypes:
        space = odl.uniform_discr([0, -1], [1, 1], (4, 5), dtype=dtype,
                                  impl=fn_impl)
        res_space = odl.uniform_discr([0, -1.4], [1.5, 1.4], (6, 7),
                                      dtype=dtype, impl=fn_impl)
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


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
