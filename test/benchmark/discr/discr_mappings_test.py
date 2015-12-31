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

"""Benchmarks for `discr_mappings`."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import pytest
import numpy as np

# Internal
import odl
from odl.discr.grid import sparse_meshgrid
from odl.discr.discr_mappings import (
    GridCollocation, NearestInterpolation, LinearInterpolation,
    PerAxisInterpolation)


pytestmark = pytest.mark.benchmark


# Initialize some large data

# First, a 100^4 grid with zeros
ndim = 4
ngrid = 100
shape = (ngrid,) * ndim
values = np.zeros(np.prod(shape))

# Next, a function space on a 4d cube
hcube = odl.IntervalProd([0] * ndim, [1] * ndim)
grid = odl.uniform_sampling(hcube, shape, as_midp=True)
space = odl.FunctionSpace(hcube)
dspace32 = odl.Rn(grid.ntotal, dtype='float32')
dspace64 = odl.Rn(grid.ntotal, dtype='float64')

# Finally, create interpolation points (independent from grid), 60^4 points
# in total. Once as point array, once as meshgrid.
npts = 60
points = np.random.rand(ndim, npts ** ndim)
coord_vec = np.random.rand(npts)
coord_vec.sort()
mesh = sparse_meshgrid(*([coord_vec] * ndim))


def func(x):
    return np.sin(x[0] * x[1]) - np.cos(x[2] * x[3])


dspace_params = [dspace32, dspace64]
dspace_ids = ['{!r}'.format(spc) for spc in dspace_params]


@pytest.fixture(scope="module", ids=dspace_ids, params=dspace_params)
def dspace(request):
    return request.param


@pytest.fixture(scope="module", ids=[' Array input ', ' Meshgrid input '],
                params=[points, mesh])
def input_values(request):
    return request.param


# ----- Benchmarks -----


def test_grid_collocation(dspace):
    coll_op = GridCollocation(space, grid, dspace)
    coll_op(func)


def test_nearest_interpolation(dspace, input_values):
    nn_interp_op = NearestInterpolation(space, grid, dspace, variant='left')
    function = nn_interp_op(values)
    function(input_values)


def test_linear_interpolation(dspace, input_values):
    lin_interp_op = LinearInterpolation(space, grid, dspace)
    function = lin_interp_op(values)
    function(input_values)


def test_per_axis_interpolation(dspace, input_values):
    schemes = ['nearest', 'linear', 'linear', 'nearest']
    pa_interp_op = PerAxisInterpolation(space, grid, dspace, schemes=schemes)
    function = pa_interp_op(values)
    function(input_values)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
