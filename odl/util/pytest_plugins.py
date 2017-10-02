# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test configuration file."""

from __future__ import print_function, division, absolute_import

import numpy as np
import operator
import os

import odl
from odl.space.entry_points import tensor_space_impl_names
from odl.trafos.backends import PYFFTW_AVAILABLE, PYWT_AVAILABLE
from odl.util.testutils import simple_fixture
from odl.util.utility import dtype_repr

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
collect_ignore = [os.path.join(odl_root, 'setup.py'),
                  os.path.join(odl_root, 'odl', 'contrib')]


# Add example directories to `collect_ignore`
def find_example_dirs():
    dirs = []
    for dirpath, dirnames, _ in os.walk(odl_root):
        if 'examples' in dirnames:
            dirs.append(os.path.join(dirpath, 'examples'))
    return dirs


collect_ignore.extend(find_example_dirs())


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


# Remove duplicates
collect_ignore = list(set(collect_ignore))
collect_ignore = [os.path.normcase(ignored) for ignored in collect_ignore]


def pytest_ignore_collect(path, config):
    normalized = os.path.normcase(str(path))
    return any(normalized.startswith(ignored) for ignored in collect_ignore)


# --- Reusable fixtures ---

tspace_impl = simple_fixture(name='tspace_impl',
                             params=tensor_space_impl_names())


floating_dtype_params = np.sctypes['float'] + np.sctypes['complex']
floating_dtype_ids = [' dtype={} '.format(dtype_repr(dt))
                      for dt in floating_dtype_params]


@fixture(scope="module", ids=floating_dtype_ids, params=floating_dtype_params)
def floating_dtype(request):
    """Floating point (real or complex) dtype."""
    return request.param


scalar_dtype_params = (floating_dtype_params +
                       np.sctypes['int'] +
                       np.sctypes['uint'])
scalar_dtype_ids = [' dtype={} '.format(dtype_repr(dt))
                    for dt in scalar_dtype_params]


@fixture(scope="module", ids=scalar_dtype_ids, params=scalar_dtype_params)
def scalar_dtype(request):
    """Scalar (integers or real or complex) dtype."""
    return request.param


ufunc_params = odl.util.ufuncs.UFUNCS
ufunc_ids = [' ufunc={} '.format(p[0]) for p in ufunc_params]


@fixture(scope="module", ids=ufunc_ids, params=ufunc_params)
def ufunc(request):
    """Tuple with information on a ufunc.

    Returns
    -------
    name : str
        Name of the ufunc.
    n_in : int
        Number of input values of the ufunc.
    n_out : int
        Number of output values of the ufunc.
    doc : str
        Docstring for the ufunc.
    """
    return request.param


reduction_params = odl.util.ufuncs.REDUCTIONS
reduction_ids = [' reduction={} '.format(p[0]) for p in reduction_params]


@fixture(scope="module", ids=reduction_ids, params=reduction_params)
def reduction(request):
    """Tuple with information on a reduction.

    Returns
    -------
    name : str
        Name of the reduction.
    doc : str
        Docstring for the reduction.
    """
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


@fixture(ids=arithmetic_op_ids, params=arithmetic_op_par)
def arithmetic_op(request):
    """An arithmetic operator, e.g. +, -, // etc."""
    return request.param
