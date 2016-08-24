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
    begin = [2, -5]
    end = [10, 4]

    # Simply test if code runs
    odl.RectPartition(odl.Rectangle(begin, end), odl.TensorGrid(vec1, vec2))
    odl.RectPartition(odl.Interval(begin[0], end[0]), odl.TensorGrid(vec1))

    # Degenerate dimensions should work, too
    vec2 = np.array([1.0])
    odl.RectPartition(odl.Rectangle(begin, end), odl.TensorGrid(vec1, vec2))


def test_partition_init_raise():
    # Check different error scenarios
    vec1 = np.array([2, 4, 5, 7])
    vec2 = np.array([-4, -3, 0, 1, 4])
    grid = odl.TensorGrid(vec1, vec2)
    begin = [2, -5]
    end = [10, 4]

    beg_toolarge = (2, -3.5)
    end_toosmall = (7, 1)
    beg_badshape = (-1, 2, 0)
    end_badshape = (2,)

    with pytest.raises(ValueError):
        odl.RectPartition(odl.IntervalProd(beg_toolarge, end), grid)

    with pytest.raises(ValueError):
        odl.RectPartition(odl.IntervalProd(begin, end_toosmall), grid)

    with pytest.raises(ValueError):
        odl.RectPartition(odl.IntervalProd(beg_badshape, end_badshape), grid)

    with pytest.raises(TypeError):
        odl.RectPartition(None, grid)

    with pytest.raises(TypeError):
        odl.RectPartition(odl.IntervalProd(beg_toolarge, end), None)


def test_partition_set():
    vec1 = np.array([2, 4, 5, 7])
    vec2 = np.array([-4, -3, 0, 1, 4])
    grid = odl.TensorGrid(vec1, vec2)

    begin = [1, -4]
    end = [10, 5]
    intv = odl.IntervalProd(begin, end)

    part = odl.RectPartition(intv, grid)
    assert part.set == odl.IntervalProd(begin, end)
    assert all_equal(part.begin, begin)
    assert all_equal(part.min(), begin)
    assert all_equal(part.end, end)
    assert all_equal(part.max(), end)


def test_partition_cell_boundary_vecs():
    vec1 = np.array([2, 4, 5, 7])
    vec2 = np.array([-4, -3, 0, 1, 4])
    grid = odl.TensorGrid(vec1, vec2)

    midpts1 = [3, 4.5, 6]
    midpts2 = [-3.5, -1.5, 0.5, 2.5]

    begin = [2, -6]
    end = [10, 4]
    intv = odl.IntervalProd(begin, end)

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

    begin = [2, -6]
    end = [10, 4]
    intv = odl.IntervalProd(begin, end)

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
    begin1 = [1, -4]
    end1 = [7, 5]
    grid1 = odl.TensorGrid(vec11, vec12)
    intv1 = odl.IntervalProd(begin1, end1)
    part1 = odl.RectPartition(intv1, grid1)

    vec21 = [-2, 0, 3]
    vec22 = [0]
    begin2 = [-2, -2]
    end2 = [4, 0]
    grid2 = odl.TensorGrid(vec21, vec22)
    intv2 = odl.IntervalProd(begin2, end2)
    part2 = odl.RectPartition(intv2, grid2)

    part = part1.insert(0, part2)
    assert all_equal(part.begin, [-2, -2, 1, -4])
    assert all_equal(part.end, [4, 0, 7, 5])
    assert all_equal(part.grid.min_pt, [-2, 0, 2, -4])
    assert all_equal(part.grid.max_pt, [3, 0, 7, 4])

    part = part1.insert(1, part2)
    assert all_equal(part.begin, [1, -2, -2, -4])
    assert all_equal(part.end, [7, 4, 0, 5])
    assert all_equal(part.grid.min_pt, [2, -2, 0, -4])
    assert all_equal(part.grid.max_pt, [7, 3, 0, 4])


def test_partition_getitem():
    vec1 = [2, 4, 5, 7]
    vec2 = [-4, -3, 0, 1, 4]
    vec3 = [-2, 0, 3]
    vec4 = [0]
    vecs = [vec1, vec2, vec3, vec4]
    begin = [1, -4, -2, -2]
    end = [7, 5, 4, 0]
    grid = odl.TensorGrid(*vecs)
    intv = odl.IntervalProd(begin, end)
    part = odl.RectPartition(intv, grid)

    # Test a couple of slices
    slc = (1, -2, 2, 0)
    slc_vecs = [v[i] for i, v in zip(slc, vecs)]
    slc_part = part[slc]
    assert slc_part.grid == odl.TensorGrid(*slc_vecs)
    slc_beg = [3, 0.5, 1.5, -2]
    slc_end = [4.5, 2.5, 4, 0]
    assert slc_part.set == odl.IntervalProd(slc_beg, slc_end)

    slc = (slice(None), slice(None, None, 2), slice(None, 2), 0)
    slc_vecs = [v[i] for i, v in zip(slc, vecs)]
    slc_part = part[slc]
    assert slc_part.grid == odl.TensorGrid(*slc_vecs)
    slc_beg = [1, -4, -2, -2]
    slc_end = [7, 5, 1.5, 0]
    assert slc_part.set == odl.IntervalProd(slc_beg, slc_end)

    # Fewer indices
    assert part[1] == part[1, :, :, :] == part[1, ...]
    assert part[1, 2:] == part[1, 2:, :, :] == part[1, 2:, ...]
    assert part[1, 2:, ::2] == part[1, 2:, ::2, :] == part[1, 2:, ::2, ...]

    # Index list using indices 0 and 2
    lst_beg = [1, -4, -2, -2]
    lst_end = [6, 5, 4, 0]
    lst_intv = odl.IntervalProd(lst_beg, lst_end)
    lst_vec1 = [2, 5]
    lst_grid = odl.TensorGrid(lst_vec1, vec2, vec3, vec4)
    lst_part = odl.RectPartition(lst_intv, lst_grid)
    assert part[[0, 2]] == lst_part


# ---- Functions ---- #


def test_uniform_partition_fromintv():
    intvp = odl.IntervalProd([0, 0], [1, 2])
    nsamp = (4, 10)

    # All nodes at the boundary
    part = odl.uniform_partition_fromintv(intvp, nsamp, nodes_on_bdry=True)
    assert all_equal(part.begin, intvp.begin)
    assert all_equal(part.end, intvp.end)
    assert all_equal(part.grid.min_pt, intvp.begin)
    assert all_equal(part.grid.max_pt, intvp.end)
    for cs in part.cell_sizes_vecs:
        # Check that all cell sizes are equal (except first and last which
        # are halved)
        assert np.allclose(np.diff(cs[1:-1]), 0)
        assert all_almost_equal(cs[0], cs[1] / 2)
        assert all_almost_equal(cs[-1], cs[-2] / 2)

    # All nodes not the boundary
    part = odl.uniform_partition_fromintv(intvp, nsamp, nodes_on_bdry=False)
    assert all_equal(part.begin, intvp.begin)
    assert all_equal(part.end, intvp.end)
    for cs in part.cell_sizes_vecs:
        # Check that all cell sizes are equal
        assert np.allclose(np.diff(cs), 0)

    # Only left nodes at the boundary
    part = odl.uniform_partition_fromintv(intvp, nsamp,
                                          nodes_on_bdry=[[True, False]] * 2)
    assert all_equal(part.begin, intvp.begin)
    assert all_equal(part.end, intvp.end)
    assert all_equal(part.grid.min_pt, intvp.begin)
    for cs in part.cell_sizes_vecs:
        # Check that all cell sizes are equal (except first)
        assert np.allclose(np.diff(cs[1:]), 0)
        assert all_almost_equal(cs[0], cs[1] / 2)

    # Only right nodes at the boundary
    part = odl.uniform_partition_fromintv(intvp, nsamp,
                                          nodes_on_bdry=[[False, True]] * 2)
    assert all_equal(part.begin, intvp.begin)
    assert all_equal(part.end, intvp.end)
    assert all_equal(part.grid.max_pt, intvp.end)
    for cs in part.cell_sizes_vecs:
        # Check that all cell sizes are equal (except last)
        assert np.allclose(np.diff(cs[:-1]), 0)
        assert all_almost_equal(cs[-1], cs[-2] / 2)


def test_uniform_partition_fromgrid():
    vec1 = np.array([2, 4, 5, 7])
    vec2 = np.array([-4, -3, 0, 1, 4])
    begin = [0, -4]
    end = [7, 8]
    beg_calc = [2 - (4 - 2) / 2, -4 - (-3 + 4) / 2]
    end_calc = [7 + (7 - 5) / 2, 4 + (4 - 1) / 2]

    # Default case
    grid = odl.TensorGrid(vec1, vec2)
    part = odl.uniform_partition_fromgrid(grid)
    assert part.set == odl.IntervalProd(beg_calc, end_calc)

    # Explicit begin / end, full vectors
    part = odl.uniform_partition_fromgrid(grid, begin=begin)
    assert part.set == odl.IntervalProd(begin, end_calc)
    part = odl.uniform_partition_fromgrid(grid, end=end)
    assert part.set == odl.IntervalProd(beg_calc, end)

    # begin / end as dictionaries
    beg_dict = {0: 0.5}
    end_dict = {-1: 8}
    part = odl.uniform_partition_fromgrid(grid, begin=beg_dict, end=end_dict)
    true_beg = [0.5, beg_calc[1]]
    true_end = [end_calc[0], 8]
    assert part.set == odl.IntervalProd(true_beg, true_end)

    # Degenerate dimension, needs both explicit begin and end
    grid = odl.TensorGrid(vec1, [1.0])
    with pytest.raises(ValueError):
        odl.uniform_partition_fromgrid(grid)
    with pytest.raises(ValueError):
        odl.uniform_partition_fromgrid(grid, begin=begin)
    with pytest.raises(ValueError):
        odl.uniform_partition_fromgrid(grid, end=end)


def test_uniform_partition():

    begin = [0, 0]
    end = [1, 2]
    nsamp = (4, 10)
    csides = [0.25, 0.2]

    # Test standard case
    part = odl.uniform_partition(begin, end, nsamp, nodes_on_bdry=True)

    assert all_equal(part.begin, begin)
    assert all_equal(part.end, end)
    assert all_equal(part.grid.min_pt, begin)
    assert all_equal(part.grid.max_pt, end)
    for cs in part.cell_sizes_vecs:
        # Check that all cell sizes are equal (except first and last which
        # are halved)
        assert np.allclose(np.diff(cs[1:-1]), 0)
        assert all_almost_equal(cs[0], cs[1] / 2)
        assert all_almost_equal(cs[-1], cs[-2] / 2)

    assert part[1:, 2:5].is_uniform
    assert part[1:, ::3].is_uniform

    # Test combinations of parameters
    true_part = odl.uniform_partition(begin, end, nsamp, nodes_on_bdry=False)
    part = odl.uniform_partition(begin=begin, end=end, num_nodes=nsamp,
                                 cell_sides=None)
    assert part == true_part
    part = odl.uniform_partition(begin=begin, end=end, num_nodes=None,
                                 cell_sides=csides)
    assert part == true_part
    part = odl.uniform_partition(begin=begin, end=None,
                                 num_nodes=nsamp, cell_sides=csides)
    assert part == true_part
    part = odl.uniform_partition(begin=None, end=end, num_nodes=nsamp,
                                 cell_sides=csides)
    assert part == true_part
    part = odl.uniform_partition(begin=begin, end=end, num_nodes=nsamp,
                                 cell_sides=csides)
    assert part == true_part

    # Test parameters per axis
    part = odl.uniform_partition(
        begin=[0, None], end=[None, 2], num_nodes=nsamp, cell_sides=csides)
    assert part == true_part
    part = odl.uniform_partition(
        begin=begin, end=[None, 2], num_nodes=(4, None), cell_sides=csides)
    assert part == true_part
    part = odl.uniform_partition(
        begin=begin, end=end, num_nodes=(None, 4), cell_sides=[0.25, None])

    # Test robustness against numerical error
    part = odl.uniform_partition(
        begin=begin, end=[None, np.sqrt(2) ** 2], num_nodes=nsamp,
        cell_sides=[0.25, np.log(np.exp(0.2))])
    assert part.approx_equals(true_part, atol=1e-8)

    # Test nodes_on_bdry
    # Here we compute stuff, so we can only expect approximate equality
    csides = [1 / 3., 2 / 9.5]
    true_part = odl.uniform_partition(begin, end, nsamp,
                                      nodes_on_bdry=(True, (False, True)))
    part = odl.uniform_partition(
        begin=[0, None], end=[None, 2], num_nodes=nsamp,
        cell_sides=csides, nodes_on_bdry=(True, (False, True)))
    assert part.approx_equals(true_part, atol=1e-8)
    part = odl.uniform_partition(
        begin=begin, end=[None, 2], num_nodes=(4, None), cell_sides=csides,
        nodes_on_bdry=(True, (False, True)))
    assert part.approx_equals(true_part, atol=1e-8)
    part = odl.uniform_partition(
        begin=begin, end=end, num_nodes=(None, 10), cell_sides=[1 / 3., None],
        nodes_on_bdry=(True, (False, True)))
    assert part.approx_equals(true_part, atol=1e-8)

    # Test error scenarios

    # Not enough parameters (total / per axis)
    with pytest.raises(ValueError):
        odl.uniform_partition()

    with pytest.raises(ValueError):
        odl.uniform_partition(begin, end)

    with pytest.raises(ValueError):
        part = odl.uniform_partition(
            begin=[0, None], end=[1, None], num_nodes=nsamp, cell_sides=csides)

    with pytest.raises(ValueError):
        part = odl.uniform_partition(
            begin=begin, end=[1, None], num_nodes=(4, None), cell_sides=csides)

    # Parameters with inconsistent sizes
    with pytest.raises(ValueError):
        part = odl.uniform_partition(
            begin=begin, end=[1, None, None], num_nodes=nsamp)

    # Too large rounding error in computing num_nodes
    with pytest.raises(ValueError):
        part = odl.uniform_partition(
            begin=begin, end=end, cell_sides=[0.25, 0.2001])

    # Inconsistent values
    with pytest.raises(ValueError):
        part = odl.uniform_partition(
            begin=begin, end=end, num_nodes=nsamp, cell_sides=[0.25, 0.2001])

if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
