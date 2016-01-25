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

# External module imports
import pytest
import numpy as np

# ODL imports
import odl
from odl.discr.partition import RectPartition, uniform_partition
from odl.set.domain import IntervalProd
from odl.util.testutils import all_equal, all_almost_equal


# ---- RectPartition ---- #


def test_partition_init():
    vec1 = np.array([2, 4, 5, 7])
    vec2 = np.array([-4, -3, 0, 1, 4])
    begin = [2, -5]
    end = [10, 4]
    grid = odl.TensorGrid(vec1, vec2)

    # Simply test if code runs
    RectPartition(grid)
    RectPartition(grid, begin=begin)
    RectPartition(grid, end=end)
    RectPartition(grid, begin=begin, begin_axes=(1,))
    RectPartition(grid, end=end, end_axes=(-1,))

    # 1d
    grid = odl.TensorGrid(vec1)
    RectPartition(grid)

    # Degenerate dimensions should work with explicit begin / end
    # in corresponding axes
    grid = odl.TensorGrid(vec1, [1.0])
    RectPartition(grid, begin=begin, end=end)
    RectPartition(grid, begin=begin, end=end,
                  begin_axes=(1,), end_axes=(1,))


def test_partition_init_error():
    # Check different error scenarios
    vec1 = np.array([2, 4, 5, 7])
    vec2 = np.array([-4, -3, 0, 1, 4])
    begin = [2, -5]
    end = [10, 4]
    grid = odl.TensorGrid(vec1, vec2)

    beg_toolarge = (2, -3.5)
    end_toosmall = (7, 1)
    beg_badshape = (-1, 2, 0)
    end_badshape = (2,)

    with pytest.raises(ValueError):
        RectPartition(grid, begin=beg_toolarge)

    with pytest.raises(ValueError):
        RectPartition(grid, end=end_toosmall)

    with pytest.raises(ValueError):
        RectPartition(grid, begin=beg_badshape)

    with pytest.raises(ValueError):
        RectPartition(grid, end=end_badshape)

    # Degenerate dimension, needs both explicit begin and end
    grid = odl.TensorGrid(vec1, [1.0])
    with pytest.raises(ValueError):
        RectPartition(grid)
    with pytest.raises(ValueError):
        RectPartition(grid, begin=begin)
    with pytest.raises(ValueError):
        RectPartition(grid, end=end)

    # begin_axes (and end_axes, same function, testing only begin)

    # begin is required
    with pytest.raises(ValueError):
        RectPartition(grid, begin_axes=(0,))

    # OOB indices
    with pytest.raises(IndexError):
        RectPartition(grid, begin=begin, begin_axes=(0, 2))
    with pytest.raises(IndexError):
        RectPartition(grid, begin=begin, begin_axes=(-3,))

    # Duplicate indices
    with pytest.raises(ValueError):
        RectPartition(grid, begin=begin, begin_axes=(0, 0, 1,))


def test_partition_bbox():
    vec1 = np.array([2, 4, 5, 7])
    vec2 = np.array([-4, -3, 0, 1, 4])
    grid = odl.TensorGrid(vec1, vec2)

    begin = [1, -4]
    end = [10, 5]
    minpt_calc = [2 - (4 - 2) / 2, -4 - (-3 + 4) / 2]
    maxpt_calc = [7 + (7 - 5) / 2, 4 + (4 - 1) / 2]

    # Explicit begin / end
    part = RectPartition(grid, begin=begin, end=end)
    assert part.bbox == IntervalProd(begin, end)

    # Implicit begin / end
    part = RectPartition(grid, begin=begin)
    assert part.bbox == IntervalProd(begin, maxpt_calc)

    part = RectPartition(grid, end=end)
    assert part.bbox == IntervalProd(minpt_calc, end)

    part = RectPartition(grid)
    assert part.bbox == IntervalProd(minpt_calc, maxpt_calc)

    # Mixture
    part = RectPartition(grid, begin=begin, begin_axes=(1,))
    minpt_mix = [minpt_calc[0], begin[1]]
    assert part.bbox == IntervalProd(minpt_mix, maxpt_calc)

    part = RectPartition(grid, end=end, end_axes=(-2,))
    maxpt_mix = [end[0], maxpt_calc[1]]
    assert part.bbox == IntervalProd(minpt_calc, maxpt_mix)


def test_partition_cell_boundaries():
    vec1 = np.array([2, 4, 5, 7])
    vec2 = np.array([-4, -3, 0, 1, 4])
    grid = odl.TensorGrid(vec1, vec2)

    midpts1 = [3, 4.5, 6]
    midpts2 = [-3.5, -1.5, 0.5, 2.5]

    # Explicit
    begin = [2, -6]
    end = [10, 4]
    true_bvec1 = [2] + midpts1 + [10]
    true_bvec2 = [-6] + midpts2 + [4]

    part = RectPartition(grid, begin=begin, end=end)
    assert all_equal(part.cell_boundaries(), (true_bvec1, true_bvec2))

    # Implicit
    true_bvec1 = [1] + midpts1 + [8]
    true_bvec2 = [-4.5] + midpts2 + [5.5]

    part = RectPartition(grid)
    assert all_equal(part.cell_boundaries(), (true_bvec1, true_bvec2))


def test_partition_cell_sizes():
    vec1 = np.array([2, 4, 5, 7])
    vec2 = np.array([-4, -3, 0, 1, 4])
    grid = odl.TensorGrid(vec1, vec2)

    midpts1 = [3, 4.5, 6]
    midpts2 = [-3.5, -1.5, 0.5, 2.5]

    # Explicit
    begin = [2, -6]
    end = [10, 4]
    bvec1 = np.array([2] + midpts1 + [10])
    bvec2 = np.array([-6] + midpts2 + [4])
    true_csizes1 = bvec1[1:] - bvec1[:-1]
    true_csizes2 = bvec2[1:] - bvec2[:-1]

    part = RectPartition(grid, begin=begin, end=end)
    assert all_equal(part.cell_sizes(), (true_csizes1, true_csizes2))

    # Implicit
    bvec1 = np.array([1] + midpts1 + [8])
    bvec2 = np.array([-4.5] + midpts2 + [5.5])

    true_csizes1 = bvec1[1:] - bvec1[:-1]
    true_csizes2 = bvec2[1:] - bvec2[:-1]

    part = RectPartition(grid)
    assert all_equal(part.cell_sizes(), (true_csizes1, true_csizes2))


def test_partition_cell_volume():
    grid = odl.RegularGrid([0, 1], [2, 4], (5, 3))
    part = RectPartition(grid)
    true_volume = 0.5 * 1.5
    assert part.cell_volume == true_volume


def test_partition_insert():
    vec11 = [2, 4, 5, 7]
    vec12 = [-4, -3, 0, 1, 4]
    grid1 = odl.TensorGrid(vec11, vec12)
    part1 = RectPartition(grid1)

    vec21 = [-2, 0, 3]
    vec22 = [0]
    grid2 = odl.TensorGrid(vec21, vec22)
    part2 = RectPartition(grid2, begin=[-2, 0], end=[3, 0])

    part = part1.insert(0, part2)
    true_beg = [-2, 0, 1, -4.5]
    true_end = [3, 0, 8, 5.5]
    assert all_equal(part.begin, true_beg)
    assert all_equal(part.end, true_end)

    part = part1.insert(1, part2)
    true_beg = [1, -2, 0, -4.5]
    true_end = [8, 3, 0, 5.5]
    assert all_equal(part.begin, true_beg)
    assert all_equal(part.end, true_end)


# ---- Functions ---- #


def test_uniform_partition():
    intvp = odl.IntervalProd([0, 0], [1, 2])
    nsamp = (4, 10)

    # All nodes at the boundary
    part = uniform_partition(intvp, nsamp, nodes_on_bdry=True)
    assert all_equal(part.begin, intvp.begin)
    assert all_equal(part.end, intvp.end)
    assert all_equal(part.grid.min_pt, intvp.begin)
    assert all_equal(part.grid.max_pt, intvp.end)
    for cs in part.cell_sizes():
        # Check that all cell sizes are equal (except first and last which
        # are halved)
        assert np.allclose(np.diff(cs[1:-1]), 0)
        assert all_almost_equal(cs[0], cs[1] / 2)
        assert all_almost_equal(cs[-1], cs[-2] / 2)

    # All nodes not the boundary
    part = uniform_partition(intvp, nsamp, nodes_on_bdry=False)
    assert all_equal(part.begin, intvp.begin)
    assert all_equal(part.end, intvp.end)
    for cs in part.cell_sizes():
        # Check that all cell sizes are equal
        assert np.allclose(np.diff(cs), 0)

    # Only left nodes at the boundary
    part = uniform_partition(intvp, nsamp, nodes_on_bdry=[[True, False]] * 2)
    assert all_equal(part.begin, intvp.begin)
    assert all_equal(part.end, intvp.end)
    assert all_equal(part.grid.min_pt, intvp.begin)
    for cs in part.cell_sizes():
        # Check that all cell sizes are equal (except first)
        assert np.allclose(np.diff(cs[1:]), 0)
        assert all_almost_equal(cs[0], cs[1] / 2)

    # Only right nodes at the boundary
    part = uniform_partition(intvp, nsamp, nodes_on_bdry=[[False, True]] * 2)
    assert all_equal(part.begin, intvp.begin)
    assert all_equal(part.end, intvp.end)
    assert all_equal(part.grid.max_pt, intvp.end)
    for cs in part.cell_sizes():
        # Check that all cell sizes are equal (except last)
        assert np.allclose(np.diff(cs[:-1]), 0)
        assert all_almost_equal(cs[-1], cs[-2] / 2)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
