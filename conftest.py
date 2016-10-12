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

import numpy as np
import operator
import pytest

import odl
from odl.trafos.backends import PYFFTW_AVAILABLE, PYWT_AVAILABLE
from odl.util import dtype_repr


collect_ignore = ['setup.py', 'run_tests.py']

if not PYFFTW_AVAILABLE:
    collect_ignore.append('odl/trafos/backends/pyfftw_bindings.py')
if not PYWT_AVAILABLE:
    collect_ignore.append('odl/trafos/backends/pywt_bindings.py')
    # Currently `pywt` is the only implementation
    collect_ignore.append('odl/trafos/wavelet.py')


def pytest_addoption(parser):
    parser.addoption('--largescale', action='store_true',
                     help='Run large and slow tests')

    parser.addoption('--benchmark', action='store_true',
                     help='Run benchmarks')

    parser.addoption('--examples', action='store_true',
                     help='Run examples')


# reusable fixtures
fn_impl_params = odl.FN_IMPLS.keys()
fn_impl_ids = [" impl='{}' ".format(p) for p in fn_impl_params]


@pytest.fixture(scope="module", ids=fn_impl_ids, params=fn_impl_params)
def fn_impl(request):
    return request.param

ntuples_impl_params = odl.NTUPLES_IMPLS.keys()
ntuples_impl_ids = [" impl='{}' ".format(p) for p in ntuples_impl_params]


@pytest.fixture(scope="module", ids=ntuples_impl_ids,
                params=ntuples_impl_params)
def ntuples_impl(request):
    return request.param

ufunc_params = [ufunc for ufunc in odl.util.ufuncs.UFUNCS]
ufunc_ids = [' ufunc={} '.format(p[0]) for p in ufunc_params]


floating_dtype_params = np.sctypes['float'] + np.sctypes['complex']
floating_dtype_ids = [' dtype={} '.format(dtype_repr(dt))
                      for dt in floating_dtype_params]


@pytest.fixture(scope="module", ids=floating_dtype_ids,
                params=floating_dtype_params)
def floating_dtype(request):
    return request.param


scalar_dtype_params = (floating_dtype_params +
                       np.sctypes['int'] +
                       np.sctypes['uint'])
scalar_dtype_ids = [' dtype={} '.format(dtype_repr(dt))
                    for dt in scalar_dtype_params]


@pytest.fixture(scope="module", ids=scalar_dtype_ids,
                params=scalar_dtype_params)
def scalar_dtype(request):
    return request.param


@pytest.fixture(scope="module", ids=ufunc_ids, params=ufunc_params)
def ufunc(request):
    return request.param


reduction_params = [reduction for reduction in odl.util.ufuncs.REDUCTIONS]
reduction_ids = [' reduction={} '.format(p[0]) for p in reduction_params]


@pytest.fixture(scope="module", ids=reduction_ids, params=reduction_params)
def reduction(request):
    return request.param

arithmetic_op_par = [operator.add,
                     operator.truediv,
                     operator.mul,
                     operator.sub,
                     operator.iadd,
                     operator.itruediv,
                     operator.imul,
                     operator.isub]
arithmetic_op_ids = [' + ', ' / ', ' * ', ' - ',
                     ' += ', ' /= ', ' *= ', ' -= ']


@pytest.fixture(ids=arithmetic_op_ids, params=arithmetic_op_par)
def arithmetic_op(request):
    """An arithmetic operator, e.g. +, -, // etc."""
    return request.param
