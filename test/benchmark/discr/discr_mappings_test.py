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
import pytest
import numpy as np

# Internal
import odl
from odl.discr.grid import sparse_meshgrid
from odl.discr.discr_mappings import (
    GridCollocation, NearestInterpolation, LinearInterpolation,
    PerAxisInterpolation)


pytestmark = pytest.mark.benchmark


def initialize():
    global shape, values, hcube, grid, space, dspace32, dspace64
    global points, mesh
    shape = (100, 100, 100, 100)
    values = np.zeros(np.prod(shape))
    hcube = odl.IntervalProd([0] * 4, [1] * 4)
    grid = odl.uniform_sampling(hcube, shape, as_midp=True)
    space = odl.FunctionSpace(hcube)
    dspace32 = odl.Rn(grid.ntotal, dtype='float32')
    dspace64 = odl.Rn(grid.ntotal, dtype='float64')
    points = np.random.rand(4, 60 ** 4)
    mesh = sparse_meshgrid(*([np.random.rand(60)] * 4))


initialize()


def func(x):
    return np.sin(x[0] * x[1]) - np.cos(x[2] * x[3])


dspace_params = [dspace32, dspace64]
dspace_ids = [' Rn, dtype = {} '.format(spc.dtype) for spc in dspace_params]
dspace_fixture = pytest.fixture(scope="module", ids=dspace_ids,
                                params=dspace_params)


@dspace_fixture
def dspace(request):
    return request.param


input_params = [points, mesh]
input_ids = [' Array input ', ' Meshgrid input ']
input_fixture = pytest.fixture(scope="module", ids=input_ids,
                               params=input_params)


@input_fixture
def input_values(request):
    return request.param


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
