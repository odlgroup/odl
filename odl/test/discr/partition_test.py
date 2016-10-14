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

# ODL imports
import odl
from odl.util.testutils import all_equal, all_almost_equal


# ---- RectPartition ---- #


def test_partition_init():
    vec1 = np.array([2, 4, 5, 7])
    vec2 = np.array([-4, -3, 0, 1, 4])
    min_pt = [2, -5]
    max_pt = [10, 4]

    # Simply test if code runs
    odl.RectPartition(odl.IntervalProd(min_pt, max_pt),
                      odl.TensorGrid(vec1, vec2))
    odl.RectPartition(odl.IntervalProd(min_pt[0], max_pt[0]),
                      odl.TensorGrid(vec1))

    # Degenerate dimensions should work, too
    vec2 = np.array([1.0])
    odl.RectPartition(odl.IntervalProd(min_pt, max_pt),
                      odl.TensorGrid(vec1, vec2))


def test_partition_init_raise():
    # Check different error scenarios
    vec1 = np.array([2, 4, 5, 7])
    vec2 = np.array([-4, -3, 0, 1, 4])
    grid = odl.TensorGrid(vec1, vec2)
    min_pt = [2, -5]
    max_pt = [10, 4]

    min_pt_toolarge = (2, -3.5)
    max_pt_toosmall = (7, 1)
    min_pt_badshape = (-1, 2, 0)
    max_pt_badshape = (2,)

    with pytest.raises(ValueError):
        odl.RectPartition(odl.IntervalProd(min_pt_toolarge, max_pt), grid)

    with pytest.raises(ValueError):
        odl.RectPartition(odl.IntervalProd(min_pt, max_pt_toosmall), grid)

    with pytest.raises(ValueError):
        odl.RectPartition(odl.IntervalProd(min_pt_badshape, max_pt_badshape),
                          grid)

    with pytest.raises(TypeError):
        odl.RectPartition(None, grid)

    with pytest.raises(TypeError):
        odl.RectPartition(odl.IntervalProd(min_pt_toolarge, max_pt), None)


def test_partition_set():
    vec1 = np.array([2, 4, 5, 7])
    vec2 = np.array([-4, -3, 0, 1, 4])
    grid = odl.TensorGrid(vec1, vec2)

    min_pt = [1, -4]
    max_pt = [10, 5]
    intv = odl.IntervalProd(min_pt, max_pt)

    part = odl.RectPartition(intv, grid)
    assert part.set == odl.IntervalProd(min_pt, max_pt)
    assert all_equal(part.min_pt, min_pt)
    assert all_equal(part.min(), min_pt)
    assert all_equal(part.max_pt, max_pt)
    assert all_equal(part.max(), max_pt)


def test_partition_cell_boundary_vecs():
    vec1 = np.array([2, 4, 5, 7])
    vec2 = np.array([-4, -3, 0, 1, 4])
    grid = odl.TensorGrid(vec1, vec2)

    midpts1 = [3, 4.5, 6]
    midpts2 = [-3.5, -1.5, 0.5, 2.5]

    min_pt = [2, -6]
    max_pt = [10, 4]
    intv = odl.IntervalProd(min_pt, max_pt)

    true_bvec1 = [2] + midpts1 + [10]
    true_bvec2 = [-6] + midpts2 + [4]

    part = odl.RectPartition(intv, grid)
    assert all_equal(part.cell_boundary_vecs, (true_bvec1, true_bvec2))


def test_partition_cell_sizes_vecs():
    vec1 = np.array([2, 4, 5, 7])
    vec2 = np.array([-4, -3, 0, 1, 4])
    grid = odl.TensorGrid(vec1, vec2)

    midpts1 = [3, 4.5, 6]
    midpts2 = [-3.5, -1.5, 0.5, 2.5]

    min_pt = [2, -6]
    max_pt = [10, 4]
    intv = odl.IntervalProd(min_pt, max_pt)

    bvec1 = np.array([2] + midpts1 + [10])
    bvec2 = np.array([-6] + midpts2 + [4])
    true_csizes1 = bvec1[1:] - bvec1[:-1]
    true_csizes2 = bvec2[1:] - bvec2[:-1]

    part = odl.RectPartition(intv, grid)
    assert all_equal(part.cell_sizes_vecs, (true_csizes1, true_csizes2))


def test_partition_cell_sides():
    grid = odl.RegularGrid([0, 1], [2, 4], (5, 3))
    intv = odl.IntervalProd([0, 1], [2, 4])
    part = odl.RectPartition(intv, grid)
    true_sides = [0.5, 1.5]
    assert all_equal(part.cell_sides, true_sides)


def test_partition_cell_volume():
    grid = odl.RegularGrid([0, 1], [2, 4], (5, 3))
    intv = odl.IntervalProd([0, 1], [2, 4])
    part = odl.RectPartition(intv, grid)
    true_volume = 0.5 * 1.5
    assert part.cell_volume == true_volume


def test_partition_insert():
    vec11 = [2, 4, 5, 7]
    vec12 = [-4, -3, 0, 1, 4]
    min_pt1 = [1, -4]
    max_pt1 = [7, 5]
    grid1 = odl.TensorGrid(vec11, vec12)
    intv1 = odl.IntervalProd(min_pt1, max_pt1)
    part1 = odl.RectPartition(intv1, grid1)

    vec21 = [-2, 0, 3]
    vec22 = [0]
    min_pt2 = [-2, -2]
    max_pt2 = [4, 0]
    grid2 = odl.TensorGrid(vec21, vec22)
    intv2 = odl.IntervalProd(min_pt2, max_pt2)
    part2 = odl.RectPartition(intv2, grid2)

    part = part1.insert(0, part2)
    assert all_equal(part.min_pt, [-2, -2, 1, -4])
    assert all_equal(part.max_pt, [4, 0, 7, 5])
    assert all_equal(part.grid.min_pt, [-2, 0, 2, -4])
    assert all_equal(part.grid.max_pt, [3, 0, 7, 4])

    part = part1.insert(1, part2)
    assert all_equal(part.min_pt, [1, -2, -2, -4])
    assert all_equal(part.max_pt, [7, 4, 0, 5])
    assert all_equal(part.grid.min_pt, [2, -2, 0, -4])
    assert all_equal(part.grid.max_pt, [7, 3, 0, 4])


def test_partition_getitem():
    vec1 = [2, 4, 5, 7]
    vec2 = [-4, -3, 0, 1, 4]
    vec3 = [-2, 0, 3]
    vec4 = [0]
    vecs = [vec1, vec2, vec3, vec4]
    min_pt = [1, -4, -2, -2]
    max_pt = [7, 5, 4, 0]
    grid = odl.TensorGrid(*vecs)
    intv = odl.IntervalProd(min_pt, max_pt)
    part = odl.RectPartition(intv, grid)

    # Test a couple of slices
    slc = (1, -2, 2, 0)
    slc_vecs = [v[i] for i, v in zip(slc, vecs)]
    slc_part = part[slc]
    assert slc_part.grid == odl.TensorGrid(*slc_vecs)
    slc_min_pt = [3, 0.5, 1.5, -2]
    slc_max_pt = [4.5, 2.5, 4, 0]
    assert slc_part.set == odl.IntervalProd(slc_min_pt, slc_max_pt)

    slc = (slice(None), slice(None, None, 2), slice(None, 2), 0)
    slc_vecs = [v[i] for i, v in zip(slc, vecs)]
    slc_part = part[slc]
    assert slc_part.grid == odl.TensorGrid(*slc_vecs)
    slc_min_pt = [1, -4, -2, -2]
    slc_max_pt = [7, 5, 1.5, 0]
    assert slc_part.set == odl.IntervalProd(slc_min_pt, slc_max_pt)

    # Fewer indices
    assert part[1] == part[1, :, :, :] == part[1, ...]
    assert part[1, 2:] == part[1, 2:, :, :] == part[1, 2:, ...]
    assert part[1, 2:, ::2] == part[1, 2:, ::2, :] == part[1, 2:, ::2, ...]

    # Index list using indices 0 and 2
    lst_min_pt = [1, -4, -2, -2]
    lst_max_pt = [6, 5, 4, 0]
    lst_intv = odl.IntervalProd(lst_min_pt, lst_max_pt)
    lst_vec1 = [2, 5]
    lst_grid = odl.TensorGrid(lst_vec1, vec2, vec3, vec4)
    lst_part = odl.RectPartition(lst_intv, lst_grid)
    assert part[[0, 2]] == lst_part


# ---- Functions ---- #


def test_uniform_partition_fromintv():
    intvp = odl.IntervalProd([0, 0], [1, 2])
    shape = (4, 10)

    # All nodes at the boundary
    part = odl.uniform_partition_fromintv(intvp, shape, nodes_on_bdry=True)
    assert all_equal(part.min_pt, intvp.min_pt)
    assert all_equal(part.max_pt, intvp.max_pt)
    assert all_equal(part.grid.min_pt, intvp.min_pt)
    assert all_equal(part.grid.max_pt, intvp.max_pt)
    for cs in part.cell_sizes_vecs:
        # Check that all cell sizes are equal (except first and last which
        # are halved)
        assert np.allclose(np.diff(cs[1:-1]), 0)
        assert all_almost_equal(cs[0], cs[1] / 2)
        assert all_almost_equal(cs[-1], cs[-2] / 2)

    # All nodes not the boundary
    part = odl.uniform_partition_fromintv(intvp, shape, nodes_on_bdry=False)
    assert all_equal(part.min_pt, intvp.min_pt)
    assert all_equal(part.max_pt, intvp.max_pt)
    for cs in part.cell_sizes_vecs:
        # Check that all cell sizes are equal
        assert np.allclose(np.diff(cs), 0)

    # Only left nodes at the boundary
    part = odl.uniform_partition_fromintv(intvp, shape,
                                          nodes_on_bdry=[[True, False]] * 2)
    assert all_equal(part.min_pt, intvp.min_pt)
    assert all_equal(part.max_pt, intvp.max_pt)
    assert all_equal(part.grid.min_pt, intvp.min_pt)
    for cs in part.cell_sizes_vecs:
        # Check that all cell sizes are equal (except first)
        assert np.allclose(np.diff(cs[1:]), 0)
        assert all_almost_equal(cs[0], cs[1] / 2)

    # Only right nodes at the boundary
    part = odl.uniform_partition_fromintv(intvp, shape,
                                          nodes_on_bdry=[[False, True]] * 2)
    assert all_equal(part.min_pt, intvp.min_pt)
    assert all_equal(part.max_pt, intvp.max_pt)
    assert all_equal(part.grid.max_pt, intvp.max_pt)
    for cs in part.cell_sizes_vecs:
        # Check that all cell sizes are equal (except last)
        assert np.allclose(np.diff(cs[:-1]), 0)
        assert all_almost_equal(cs[-1], cs[-2] / 2)


def test_uniform_partition_fromgrid():
    vec1 = np.array([2, 4, 5, 7])
    vec2 = np.array([-4, -3, 0, 1, 4])
    min_pt = [0, -4]
    max_pt = [7, 8]
    min_pt_calc = [2 - (4 - 2) / 2, -4 - (-3 + 4) / 2]
    max_pt_calc = [7 + (7 - 5) / 2, 4 + (4 - 1) / 2]

    # Default case
    grid = odl.TensorGrid(vec1, vec2)
    part = odl.uniform_partition_fromgrid(grid)
    assert part.set == odl.IntervalProd(min_pt_calc, max_pt_calc)

    # Explicit min_pt / max_pt, full vectors
    part = odl.uniform_partition_fromgrid(grid, min_pt=min_pt)
    assert part.set == odl.IntervalProd(min_pt, max_pt_calc)
    part = odl.uniform_partition_fromgrid(grid, max_pt=max_pt)
    assert part.set == odl.IntervalProd(min_pt_calc, max_pt)

    # min_pt / max_pt as dictionaries
    min_pt_dict = {0: 0.5}
    max_pt_dict = {-1: 8}
    part = odl.uniform_partition_fromgrid(
        grid, min_pt=min_pt_dict, max_pt=max_pt_dict)
    true_min_pt = [0.5, min_pt_calc[1]]
    true_max_pt = [max_pt_calc[0], 8]
    assert part.set == odl.IntervalProd(true_min_pt, true_max_pt)

    # Degenerate dimension, needs both explicit min_pt and max_pt
    grid = odl.TensorGrid(vec1, [1.0])
    with pytest.raises(ValueError):
        odl.uniform_partition_fromgrid(grid)
    with pytest.raises(ValueError):
        odl.uniform_partition_fromgrid(grid, min_pt=min_pt)
    with pytest.raises(ValueError):
        odl.uniform_partition_fromgrid(grid, max_pt=max_pt)


def test_uniform_partition():

    min_pt = [0, 0]
    max_pt = [1, 2]
    shape = (4, 10)
    csides = [0.25, 0.2]

    # Test standard case
    part = odl.uniform_partition(min_pt, max_pt, shape, nodes_on_bdry=True)

    assert all_equal(part.min_pt, min_pt)
    assert all_equal(part.max_pt, max_pt)
    assert all_equal(part.grid.min_pt, min_pt)
    assert all_equal(part.grid.max_pt, max_pt)
    for cs in part.cell_sizes_vecs:
        # Check that all cell sizes are equal (except first and last which
        # are halved)
        assert np.allclose(np.diff(cs[1:-1]), 0)
        assert all_almost_equal(cs[0], cs[1] / 2)
        assert all_almost_equal(cs[-1], cs[-2] / 2)

    assert part[1:, 2:5].is_uniform
    assert part[1:, ::3].is_uniform

    # Test combinations of parameters
    true_part = odl.uniform_partition(min_pt, max_pt, shape,
                                      nodes_on_bdry=False)
    part = odl.uniform_partition(min_pt=min_pt, max_pt=max_pt, shape=shape,
                                 cell_sides=None)
    assert part == true_part
    part = odl.uniform_partition(min_pt=min_pt, max_pt=max_pt, shape=None,
                                 cell_sides=csides)
    assert part == true_part
    part = odl.uniform_partition(min_pt=min_pt, max_pt=None,
                                 shape=shape, cell_sides=csides)
    assert part == true_part
    part = odl.uniform_partition(min_pt=None, max_pt=max_pt, shape=shape,
                                 cell_sides=csides)
    assert part == true_part
    part = odl.uniform_partition(min_pt=min_pt, max_pt=max_pt, shape=shape,
                                 cell_sides=csides)
    assert part == true_part

    # Test parameters per axis
    part = odl.uniform_partition(
        min_pt=[0, None], max_pt=[None, 2], shape=shape, cell_sides=csides)
    assert part == true_part
    part = odl.uniform_partition(
        min_pt=min_pt, max_pt=[None, 2], shape=(4, None), cell_sides=csides)
    assert part == true_part
    part = odl.uniform_partition(
        min_pt=min_pt, max_pt=max_pt, shape=(None, 4), cell_sides=[0.25, None])

    # Test robustness against numerical error
    part = odl.uniform_partition(
        min_pt=min_pt, max_pt=[None, np.sqrt(2) ** 2], shape=shape,
        cell_sides=[0.25, np.log(np.exp(0.2))])
    assert part.approx_equals(true_part, atol=1e-8)

    # Test nodes_on_bdry
    # Here we compute stuff, so we can only expect approximate equality
    csides = [1 / 3., 2 / 9.5]
    true_part = odl.uniform_partition(min_pt, max_pt, shape,
                                      nodes_on_bdry=(True, (False, True)))
    part = odl.uniform_partition(
        min_pt=[0, None], max_pt=[None, 2], shape=shape,
        cell_sides=csides, nodes_on_bdry=(True, (False, True)))
    assert part.approx_equals(true_part, atol=1e-8)
    part = odl.uniform_partition(
        min_pt=min_pt, max_pt=[None, 2], shape=(4, None), cell_sides=csides,
        nodes_on_bdry=(True, (False, True)))
    assert part.approx_equals(true_part, atol=1e-8)
    part = odl.uniform_partition(
        min_pt=min_pt, max_pt=max_pt, shape=(None, 10),
        cell_sides=[1 / 3., None],
        nodes_on_bdry=(True, (False, True)))
    assert part.approx_equals(true_part, atol=1e-8)

    # Test error scenarios

    # Not enough parameters (total / per axis)
    with pytest.raises(ValueError):
        odl.uniform_partition()

    with pytest.raises(ValueError):
        odl.uniform_partition(min_pt, max_pt)

    with pytest.raises(ValueError):
        part = odl.uniform_partition(
            min_pt=[0, None], max_pt=[1, None], shape=shape,
            cell_sides=csides)

    with pytest.raises(ValueError):
        part = odl.uniform_partition(
            min_pt=min_pt, max_pt=[1, None], shape=(4, None),
            cell_sides=csides)

    # Parameters with inconsistent sizes
    with pytest.raises(ValueError):
        part = odl.uniform_partition(
            min_pt=min_pt, max_pt=[1, None, None], shape=shape)

    # Too large rounding error in computing shape
    with pytest.raises(ValueError):
        part = odl.uniform_partition(
            min_pt=min_pt, max_pt=max_pt, cell_sides=[0.25, 0.2001])

    # Inconsistent values
    with pytest.raises(ValueError):
        part = odl.uniform_partition(
            min_pt=min_pt, max_pt=max_pt, shape=shape,
            cell_sides=[0.25, 0.2001])

if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
