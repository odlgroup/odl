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
from builtins import zip

# External module imports
import pytest
import numpy as np

# ODL imports
import odl
from odl.discr.partition import RectPartition
from odl.set.domain import IntervalProd
from odl.util.testutils import all_equal


# TODO: tests for
# - cell boundaries
# - cell sizes


def test_partition_init():
    interv_prod = odl.Rectangle([0, 1], [2, 4])
    grid = odl.uniform_sampling(interv_prod, (5, 15))

    # Simply test if code runs
    RectPartition(grid)
    RectPartition(grid, begin=interv_prod.begin)
    RectPartition(grid, end=interv_prod.end)
    RectPartition(grid, begin=interv_prod.begin, begin_axes=(1,))
    RectPartition(grid, end=interv_prod.end, end_axes=(-1,))

    # Degenerate dimensions should work with explicit begin / end
    # in corresponding axes
    grid = odl.uniform_sampling(interv_prod, (5, 1))
    RectPartition(grid, begin=interv_prod.begin, end=interv_prod.end)
    RectPartition(grid, begin=interv_prod.begin, end=interv_prod.end,
                  begin_axes=(1,), end_axes=(1,))


def test_partition_init_error():
    # Check different error scenarios
    interv_prod = odl.Rectangle([0, 1], [2, 4])
    grid = odl.uniform_sampling(interv_prod, (5, 15))

    beg_toolarge = (0.5, 2)
    end_toosmall = (1, 3.99)
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
    grid = odl.uniform_sampling(interv_prod, (3, 1))
    with pytest.raises(ValueError):
        RectPartition(grid)
    with pytest.raises(ValueError):
        RectPartition(grid, begin=interv_prod.begin)
    with pytest.raises(ValueError):
        RectPartition(grid, end=interv_prod.end)

    # begin_axes (and end_axes, same function, testing only begin)

    # begin is required
    with pytest.raises(ValueError):
        RectPartition(grid, begin_axes=(0,))

    # OOB indices
    with pytest.raises(IndexError):
        RectPartition(grid, begin=interv_prod.begin, begin_axes=(0, 2))
    with pytest.raises(IndexError):
        RectPartition(grid, begin=interv_prod.begin, begin_axes=(-3,))

    # Duplicate indices
    with pytest.raises(ValueError):
        RectPartition(grid, begin=interv_prod.begin, begin_axes=(0, 0, 1,))


def test_partition_bounding_box():
    vec1 = np.array([2, 4, 5, 7])
    vec2 = np.array([-4, -3, 0, 1, 4])
    grid = odl.TensorGrid(vec1, vec2)

    begin = [1, -4]
    end = [10, 5]
    minpt_calc = [2 - (4 - 2) / 2, -4 - (-3 + 4) / 2]
    maxpt_calc = [7 + (7 - 5) / 2, 4 + (4 - 1) / 2]

    # Explicit begin / end
    part = RectPartition(grid, begin=begin, end=end)
    assert part.bounding_box == IntervalProd(begin, end)

    # Implicit begin / end
    part = RectPartition(grid, begin=begin)
    assert part.bounding_box == IntervalProd(begin, maxpt_calc)

    part = RectPartition(grid, end=end)
    assert part.bounding_box == IntervalProd(minpt_calc, end)

    part = RectPartition(grid)
    assert part.bounding_box == IntervalProd(minpt_calc, maxpt_calc)

    # Mixture
    part = RectPartition(grid, begin=begin, begin_axes=(1,))
    minpt_mix = [minpt_calc[0], begin[1]]
    assert part.bounding_box == IntervalProd(minpt_mix, maxpt_calc)

    part = RectPartition(grid, end=end, end_axes=(-2,))
    maxpt_mix = [end[0], maxpt_calc[1]]
    assert part.bounding_box == IntervalProd(minpt_calc, maxpt_mix)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
