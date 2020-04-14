# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test configuration file."""

from __future__ import absolute_import, division, print_function

import operator
import os
from os import path

import numpy as np

import odl
from odl.space.entry_points import tensor_space_impl_names
from odl.trafos.backends import PYFFTW_AVAILABLE, PYWT_AVAILABLE
from odl.util.testutils import simple_fixture

try:
    import pytest
    from pytest import fixture
except ImportError:
    pytest = None

    # Identity fixture
    def fixture(*arg, **kw):
        if arg and callable(arg[0]):
            return arg[0]
        return fixture


# --- Add numpy and ODL to all doctests ---


@fixture(autouse=True)
def _add_doctest_np_odl(doctest_namespace):
    doctest_namespace['np'] = np
    doctest_namespace['odl'] = odl


# --- Ignored tests due to missing modules ---


this_dir = path.dirname(__file__)
odl_root = path.abspath(path.join(this_dir, '..', '..'))
collect_ignore = [path.join(odl_root, 'setup.py'),
                  path.join(odl_root, 'odl', 'contrib')]


# Add example directories to `collect_ignore`
def find_example_dirs():
    dirs = []
    for dirpath, dirnames, _ in os.walk(odl_root):
        if 'examples' in dirnames:
            dirs.append(path.join(dirpath, 'examples'))
    return dirs


collect_ignore.extend(find_example_dirs())


if not PYFFTW_AVAILABLE:
    collect_ignore.append(
        path.join(odl_root, 'odl', 'trafos', 'backends', 'pyfftw_bindings.py')
    )
if not PYWT_AVAILABLE:
    collect_ignore.append(
        path.join(odl_root, 'odl', 'trafos', 'backends', 'pywt_bindings.py')
    )
    # Currently `pywt` is the only implementation
    collect_ignore.append(
        path.join(odl_root, 'odl', 'trafos', 'wavelet.py')
    )


# --- Command-line options --- #


def pytest_addoption(parser):
    suite_help = (
        'enable an opt-in test suite NAME. '
        'Available suites: largescale, examples, doc_doctests'
    )
    parser.addoption(
        '-S', '--suite', nargs='*', metavar='NAME', type=str, help=suite_help
    )


def pytest_configure(config):
    # Register an additional marker
    config.addinivalue_line(
        'markers', 'suite(name): mark test to belong to an opt-in suite'
    )


def pytest_runtest_setup(item):
    suites = [mark.args[0] for mark in item.iter_markers(name='suite')]
    if suites:
        if not any(val in suites for val in item.config.getoption('-S')):
            pytest.skip('test not in suites {!r}'.format(suites))


# Remove duplicates
collect_ignore = list(set(collect_ignore))
collect_ignore = [path.normcase(ignored) for ignored in collect_ignore]


# NB: magical `path` param name is needed
def pytest_ignore_collect(path, config):
    normalized = os.path.normcase(str(path))
    return any(normalized.startswith(ignored) for ignored in collect_ignore)


# --- Reusable fixtures --- #

# NOTE: All global fixtures need to be prefixed with `odl_` to make them
# non-conflicting with other packages' fixture names, since these fixtures
# are exposed globally across packages by setuptools.

# Simple ones, use helper
odl_tspace_impl = simple_fixture(name='tspace_impl',
                                 params=tensor_space_impl_names())

floating_dtypes = np.sctypes['float'] + np.sctypes['complex']
floating_dtype_params = [np.dtype(dt) for dt in floating_dtypes]
odl_floating_dtype = simple_fixture(name='dtype',
                                    params=floating_dtype_params,
                                    fmt=' {name} = np.{value.name} ')

scalar_dtypes = floating_dtype_params + np.sctypes['int'] + np.sctypes['uint']
scalar_dtype_params = [np.dtype(dt) for dt in floating_dtypes]
odl_scalar_dtype = simple_fixture(name='dtype',
                                  params=scalar_dtype_params,
                                  fmt=' {name} = np.{value.name} ')

odl_elem_order = simple_fixture(name='order', params=[None, 'C', 'F'])

odl_ufunc = simple_fixture('ufunc', [p[0] for p in odl.util.ufuncs.UFUNCS])
odl_reduction = simple_fixture('reduction', ['sum', 'prod', 'min', 'max'])

# More complicated ones with non-trivial documentation
arithmetic_op_par = [operator.add,
                     operator.truediv,
                     operator.mul,
                     operator.sub,
                     operator.iadd,
                     operator.itruediv,
                     operator.imul,
                     operator.isub]
arithmetic_op_ids = [" op = '{}' ".format(op)
                     for op in ['+', '/', '*', '-', '+=', '/=', '*=', '-=']]


@fixture(ids=arithmetic_op_ids, params=arithmetic_op_par)
def odl_arithmetic_op(request):
    """An arithmetic operator, e.g. +, -, // etc."""
    return request.param
