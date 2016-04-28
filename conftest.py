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

"""Test configuration file."""

from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import pytest
import odl
from odl.space.cu_ntuples import CUDA_AVAILABLE
from odl.trafos.wavelet import PYWAVELETS_AVAILABLE

collect_ignore = ['setup.py', 'run_tests.py']

if not CUDA_AVAILABLE:
    collect_ignore.append('odl/space/cu_ntuples.py')
if not PYWAVELETS_AVAILABLE:
    collect_ignore.append('odl/trafos/wavelet.py')


def pytest_addoption(parser):
    parser.addoption('--largescale', action='store_true',
                     help='Run large and slow tests')

    parser.addoption('--benchmark', action='store_true',
                     help='Run benchmarks')


# reusable fixtures
ufunc_params = [ufunc for ufunc in odl.util.ufuncs.UFUNCS]
ufunc_ids = [' ufunc={} '.format(p[0]) for p in ufunc_params]


@pytest.fixture(scope="module", ids=ufunc_ids, params=ufunc_params)
def ufunc(request):
    return request.param


reduction_params = [reduction for reduction in odl.util.ufuncs.REDUCTIONS]
reduction_ids = [' reduction={} '.format(p[0]) for p in reduction_params]


@pytest.fixture(scope="module", ids=reduction_ids, params=reduction_params)
def reduction(request):
    return request.param
