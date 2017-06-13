# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test configuration file."""

from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import operator
import os

import odl
from odl.trafos.backends import PYFFTW_AVAILABLE, PYWT_AVAILABLE
from odl.util.utility import dtype_repr
from odl.util.testutils import simple_fixture

try:
    from pytest import fixture
except ImportError:
    # Make fixture the identity decorator (default of OptionalArgDecorator)
    from odl.util import OptionalArgDecorator as fixture


# --- Add numpy and ODL to all doctests ---


@fixture(autouse=True)
def add_doctest_np_odl(doctest_namespace):
    doctest_namespace['np'] = np
    doctest_namespace['odl'] = odl


def pytest_addoption(parser):
    parser.addoption('--largescale', action='store_true',
                     help='Run large and slow tests')

    parser.addoption('--benchmark', action='store_true',
                     help='Run benchmarks')

    parser.addoption('--examples', action='store_true',
                     help='Run examples')

    parser.addoption('--doctest-doc', action='store_true',
                     help='Run doctests in the documentation')


# --- Ignored tests due to missing modules ---

this_dir = os.path.dirname(__file__)
odl_root = os.path.abspath(os.path.join(this_dir, os.pardir, os.pardir))
collect_ignore = [os.path.join(odl_root, 'setup.py')]

if not PYFFTW_AVAILABLE:
    collect_ignore.append(
        os.path.join(odl_root, 'odl', 'trafos', 'backends',
                     'pyfftw_bindings.py'))
if not PYWT_AVAILABLE:
    collect_ignore.append(
        os.path.join(odl_root, 'odl', 'trafos', 'backends',
                     'pywt_bindings.py'))
    # Currently `pywt` is the only implementation
    collect_ignore.append(
        os.path.join(odl_root, 'odl', 'trafos', 'wavelet.py'))


def pytest_ignore_collect(path, config):
    return os.path.normcase(str(path)) in collect_ignore


# --- Reusable fixtures ---

fn_impl_params = odl.FN_IMPLS.keys()
fn_impl_ids = [" impl = '{}' ".format(p) for p in fn_impl_params]


@fixture(scope="module", ids=fn_impl_ids, params=fn_impl_params)
def fn_impl(request):
    """String with an available `FnBase` implementation name."""
    return request.param

ntuples_impl_params = odl.NTUPLES_IMPLS.keys()
ntuples_impl_ids = [" impl = '{}' ".format(p) for p in ntuples_impl_params]


@fixture(scope="module", ids=ntuples_impl_ids, params=ntuples_impl_params)
def ntuples_impl(request):
    """String with an available `NtuplesBase` implementation name."""
    return request.param


floating_dtype_params = np.sctypes['float'] + np.sctypes['complex']
floating_dtype_ids = [' dtype = {} '.format(dtype_repr(dt))
                      for dt in floating_dtype_params]


@fixture(scope="module", ids=floating_dtype_ids, params=floating_dtype_params)
def floating_dtype(request):
    """Floating point (real or complex) dtype."""
    return request.param


scalar_dtype_params = (floating_dtype_params +
                       np.sctypes['int'] +
                       np.sctypes['uint'])
scalar_dtype_ids = [' dtype = {} '.format(dtype_repr(dt))
                    for dt in scalar_dtype_params]


@fixture(scope="module", ids=scalar_dtype_ids, params=scalar_dtype_params)
def scalar_dtype(request):
    """Scalar (integers or real or complex) dtype."""
    return request.param


ufunc = simple_fixture('ufunc', [p[0] for p in odl.util.ufuncs.UFUNCS])
reduction = simple_fixture('reduction', ['sum', 'prod', 'min', 'max'])

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


@fixture(ids=arithmetic_op_ids, params=arithmetic_op_par)
def arithmetic_op(request):
    """An arithmetic operator, e.g. +, -, // etc."""
    return request.param
