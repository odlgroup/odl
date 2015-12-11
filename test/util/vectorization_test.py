# Copyright 2014, 2015 The ODL development group
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

# External
import numpy as np
import pytest

# Internal
from odl.discr.grid import sparse_meshgrid
from odl.util.testutils import all_equal
from odl.util.vectorization import (
    is_valid_input_array, is_valid_input_meshgrid,
    meshgrid_input_order, vecs_from_meshgrid,
    out_shape_from_meshgrid, out_shape_from_array,
    vectorize)


def test_is_valid_input_array():

    # 1d
    valid_shapes = [(1, 1), (1, 2), (1, 20), (1,), (20,)]
    invalid_shapes = [(2, 1), (1, 1, 1), ()]

    for shp in valid_shapes:
        arr = np.zeros(shp)
        assert is_valid_input_array(arr, d=1)

    for shp in invalid_shapes:
        arr = np.zeros(shp)
        assert not is_valid_input_array(arr, d=1)

    # 3d
    valid_shapes = [(3, 1), (3, 2), (3, 20)]
    invalid_shapes = [(3,), (20,), (4, 1), (3, 1, 1), ()]

    for shp in valid_shapes:
        arr = np.zeros(shp)
        assert is_valid_input_array(arr, d=3)

    for shp in invalid_shapes:
        arr = np.zeros(shp)
        assert not is_valid_input_array(arr, d=3)

    # Other input
    invalid_input = [1, [[1, 2], [3, 4]], (5,)]
    for inp in invalid_input:
        assert not is_valid_input_array(inp, d=2)


def test_is_valid_input_meshgrid():

    # 1d
    x = np.zeros(2)

    valid_mg = sparse_meshgrid(x)
    assert is_valid_input_meshgrid(valid_mg, d=1)

    invalid_mg = sparse_meshgrid(x, x)
    assert not is_valid_input_meshgrid(invalid_mg, d=1)

    x = np.zeros((2, 2))
    invalid_mg = sparse_meshgrid(x)
    assert not is_valid_input_meshgrid(invalid_mg, d=1)

    # 3d
    x, y, z = np.zeros(2), np.zeros(3), np.zeros(4)

    valid_mg = sparse_meshgrid(x, y, z)
    assert is_valid_input_meshgrid(valid_mg, d=3)

    valid_mg = sparse_meshgrid(x, y, z, order='F')
    assert is_valid_input_meshgrid(valid_mg, d=3)

    invalid_mg = sparse_meshgrid(x, x, y, z)
    assert not is_valid_input_meshgrid(invalid_mg, d=3)

    x = np.zeros((3, 3))
    invalid_mg = sparse_meshgrid(x)
    assert not is_valid_input_meshgrid(invalid_mg, d=3)

    # Other input
    invalid_input = [1, [1, 2], ([1, 2], [3, 4]), (5,), np.zeros((2, 2))]
    for inp in invalid_input:
        assert not is_valid_input_meshgrid(inp, d=2)


def test_meshgrid_input_order():

    # 1d
    x = np.zeros(2)

    mg = sparse_meshgrid(x, order='C')
    assert meshgrid_input_order(mg) == 'C'

    # Both 'C' and 'F' contiguous, defaults to 'C'
    mg = sparse_meshgrid(x, order='F')
    assert meshgrid_input_order(mg) == 'C'

    # 3d
    x, y, z = np.zeros(2), np.zeros(3), np.zeros(4)

    mg = sparse_meshgrid(x, y, z, order='C')
    assert meshgrid_input_order(mg) == 'C'

    mg = sparse_meshgrid(x, y, z, order='F')
    assert meshgrid_input_order(mg) == 'F'

    # 3d, fleshed out meshgrids
    x, y, z = np.zeros(2), np.zeros(3), np.zeros(4)
    mg = np.meshgrid(x, y, z, sparse=False, indexing='ij', copy=True)
    assert meshgrid_input_order(mg) == 'C'

    mg = np.meshgrid(x, y, z, sparse=False, indexing='ij', copy=True)
    mg = [np.asfortranarray(arr) for arr in mg]
    assert meshgrid_input_order(mg) == 'F'

    # non-contiguous --> error
    mg = np.meshgrid(x, y, z, sparse=False, indexing='ij', copy=False)
    with pytest.raises(ValueError):
        meshgrid_input_order(mg)

    # messed up tuple
    mg = list(sparse_meshgrid(x, y, z, order='C'))
    mg[1] = mg[0]
    with pytest.raises(ValueError):
        meshgrid_input_order(mg)


def test_vecs_from_meshgrid():

    # 1d
    x = np.zeros(2)

    mg = sparse_meshgrid(x, order='C')
    vx = vecs_from_meshgrid(mg, order='C')[0]
    assert x.shape == vx.shape
    assert all_equal(x, vx)

    mg = sparse_meshgrid(x, order='F')
    vx = vecs_from_meshgrid(mg, order='F')[0]
    assert x.shape == vx.shape
    assert all_equal(x, vx)

    # 3d
    x, y, z = np.zeros(2), np.zeros(3), np.zeros(4)

    mg = sparse_meshgrid(x, y, z, order='C')
    vx, vy, vz = vecs_from_meshgrid(mg, order='C')
    assert x.shape == vx.shape
    assert all_equal(x, vx)
    assert y.shape == vy.shape
    assert all_equal(y, vy)
    assert z.shape == vz.shape
    assert all_equal(z, vz)

    mg = sparse_meshgrid(x, y, z, order='F')
    vx, vy, vz = vecs_from_meshgrid(mg, order='F')
    assert x.shape == vx.shape
    assert all_equal(x, vx)
    assert y.shape == vy.shape
    assert all_equal(y, vy)
    assert z.shape == vz.shape
    assert all_equal(z, vz)

    # 3d, fleshed out meshgrids
    x, y, z = np.zeros(2), np.zeros(3), np.zeros(4)
    mg = np.meshgrid(x, y, z, sparse=False, indexing='ij', copy=True)
    vx, vy, vz = vecs_from_meshgrid(mg, order='C')
    assert x.shape == vx.shape
    assert all_equal(x, vx)
    assert y.shape == vy.shape
    assert all_equal(y, vy)
    assert z.shape == vz.shape
    assert all_equal(z, vz)

    mg = np.meshgrid(x, y, z, sparse=False, indexing='ij', copy=True)
    mg = tuple(reversed([np.asfortranarray(arr) for arr in mg]))
    vx, vy, vz = vecs_from_meshgrid(mg, order='F')
    assert x.shape == vx.shape
    assert all_equal(x, vx)
    assert y.shape == vy.shape
    assert all_equal(y, vy)
    assert z.shape == vz.shape
    assert all_equal(z, vz)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
