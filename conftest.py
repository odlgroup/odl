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
import numpy as np
from odl.trafos.wavelet import PYWAVELETS_AVAILABLE


# --- Add numpy and ODL to all doctests ---


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace['np'] = np
    doctest_namespace['odl'] = odl


# --- Files to be ignored by the tests ---


collect_ignore = ['setup.py', 'run_tests.py']

if not PYWAVELETS_AVAILABLE:
    collect_ignore.append('odl/trafos/wavelet.py')


# --- Options flags for the tests ---


def pytest_addoption(parser):
    parser.addoption('--largescale', action='store_true',
                     help='Run large and slow tests')

    parser.addoption('--benchmark', action='store_true',
                     help='Run benchmarks')

    parser.addoption('--examples', action='store_true',
                     help='Run examples')


# --- Reusable fixtures ---

fn_impl_params = odl.FN_IMPLS.keys()
fn_impl_ids = [" impl='{}' ".format(p) for p in fn_impl_params]


@pytest.fixture(scope="module", ids=fn_impl_ids, params=fn_impl_params)
def fn_impl(request):
    """String with a available `FnBase` implementation name."""
    return request.param

ntuples_impl_params = odl.NTUPLES_IMPLS.keys()
ntuples_impl_ids = [" impl='{}' ".format(p) for p in ntuples_impl_params]


@pytest.fixture(scope="module", ids=ntuples_impl_ids,
                params=ntuples_impl_params)
def ntuples_impl(request):
    """String with a available `NtuplesBase` implementation name."""
    return request.param

ufunc_params = [ufunc for ufunc in odl.util.ufuncs.UFUNCS]
ufunc_ids = [' ufunc={} '.format(p[0]) for p in ufunc_params]


@pytest.fixture(scope="module", ids=ufunc_ids, params=ufunc_params)
def ufunc(request):
    """Tuple with information on a ufunc.

    Returns
    -------
    name : `str`
        Name of the ufunc.
    n_in : `int`
        Number of input values of the ufunc.
    n_out : `int`
        Number of output values of the ufunc.
    doc : `str`
        Docstring for the ufunc.
    """
    return request.param


reduction_params = [reduction for reduction in odl.util.ufuncs.REDUCTIONS]
reduction_ids = [' reduction={} '.format(p[0]) for p in reduction_params]


@pytest.fixture(scope="module", ids=reduction_ids, params=reduction_params)
def reduction(request):
    """Tuple with information on a reduction.

    Returns
    -------
    name : `str`
        Name of the reduction.
    doc : `str`
        Docstring for the reduction.
    """
    return request.param
