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
from odl.discr.discr_ops import _SUPPORTED_PAD_MODES


# --- ResizingOperator --- #


def test_resizing_op_init():

    # Test if the different init patterns run

    space = odl.uniform_discr([0, -1], [1, 1], (10, 5))
    res_space = odl.uniform_discr([0, -3], [2, 3], (20, 15))

    odl.ResizingOperator(space, res_space)
    odl.ResizingOperator(space, ran_shp=(20, 15))
    odl.ResizingOperator(space, ran_shp=(20, 15), num_left=(0, 5))
    odl.ResizingOperator(space, ran_shp=(20, 15), pad_mode='symmetric')
    odl.ResizingOperator(space, ran_shp=(20, 15), pad_const=1.0)
    odl.ResizingOperator(space, ran_shp=(20, 15), pad_const=1.0,
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

    # num_left cannot be combined with range
    space = odl.uniform_discr([0, -1], [1, 1], (10, 5))
    res_space = odl.uniform_discr([0, -3], [2, 3], (20, 15))
    with pytest.raises(ValueError):
        odl.ResizingOperator(space, res_space, num_left=(0, 0))

    # bad pad_mode
    with pytest.raises(ValueError):
        odl.ResizingOperator(space, res_space, pad_mode='something')


def test_resizing_op_properties():

    # Explicit range, rest default
    space = odl.uniform_discr([0, -1], [1, 1], (10, 5))
    res_space = odl.uniform_discr([0, -3], [2, 3], (20, 15))
    res_op = odl.ResizingOperator(space, res_space)

    assert res_op.domain == space
    assert res_op.range == res_space
    assert res_op.num_left == (0, 5)
    assert res_op.pad_mode == 'constant'
    assert res_op.pad_const == 0.0
    assert res_op.is_linear

    # Implicit range via ran_shp and num_left
    res_op = odl.ResizingOperator(space, ran_shp=(20, 15), num_left=[0, 5])
    assert res_op.range == res_space
    assert res_op.num_left == (0, 5)

    # Different padding mode
    res_op = odl.ResizingOperator(space, res_space, pad_const=1)
    assert res_op.pad_const == 1.0
    assert not res_op.is_linear

    res_op = odl.ResizingOperator(space, res_space, pad_mode='symmetric')
    assert res_op.pad_mode == 'symmetric'
    assert res_op.is_linear


def test_resizing_op_call(fn_impl):

    # Minimal test since this operator only wraps resize_array
    space = odl.uniform_discr([0, -1], [1, 1], (4, 5))
    res_space = odl.uniform_discr([0, -0.6], [2, 0.2], (8, 2))
    res_op = odl.ResizingOperator(space, res_space)
    out = res_op(space.one())
    true_res = np.zeros((8, 2))
    true_res[:4, :] = 1
    assert np.array_equal(out, true_res)

    out = res_space.element()
    res_op(space.one(), out=out)
    assert np.array_equal(out, true_res)


def test_resizing_op_deriv():

    # Only non-trivial case is constant padding with const != 0
    space = odl.uniform_discr([0, -1], [1, 1], (4, 5))
    res_space = odl.uniform_discr([0, -0.6], [2, 0.2], (8, 2))
    res_op = odl.ResizingOperator(space, res_space, pad_const=1.0)
    res_op_deriv = res_op.derivative(space.one())
    assert res_op_deriv.pad_mode == 'constant'
    assert res_op_deriv.pad_const == 0.0


pad_modes = [('constant', 0), ('constant', 1), 'symmetric', 'periodic',
             'order0', 'order1']
pad_mode_ids = [' constant=0 ', ' constant=1 ', ' symmetric ', ' periodic ',
                ' order0 ', ' order1 ']


@pytest.fixture(scope="module", params=pad_modes, ids=pad_mode_ids)
def pad_mode(request):
    return request.param


def test_resizing_op_inverse(pad_mode):

    if isinstance(pad_mode, tuple):
        pad_mode, pad_const = pad_mode
    else:
        pad_const = 0.0

    space = odl.uniform_discr([0, -1], [1, 1], (4, 5))
    res_space = odl.uniform_discr([0, -1.4], [1.5, 1.4], (6, 7))
    res_op = odl.ResizingOperator(space, res_space, pad_mode=pad_mode,
                                  pad_const=pad_const)

    # Only left inverse if the operator extentds in all axes
    x = space.element(np.arange(space.size).reshape(space.shape))
    assert res_op.inverse(res_op(x)) == x


# TODO: adjoint


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
