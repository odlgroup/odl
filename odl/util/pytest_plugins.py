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
import os

import odl
from odl.trafos.backends import PYFFTW_AVAILABLE, PYWT_AVAILABLE
from odl.util import dtype_repr, OptionalArgDecorator

try:
    from pytest import fixture
except ImportError:
    # Make trivial decorator
    class fixture(OptionalArgDecorator):
        @staticmethod
        def _wrapper(f, *a, **kw):
            return f


# --- Add numpy and ODL to all doctests ---


@fixture(autouse=True)
def add_doctest_np_odl(doctest_namespace):
    doctest_namespace['np'] = np
    doctest_namespace['odl'] = odl


# --- Files to be ignored by the tests ---


this_dir = os.path.dirname(__file__)
odl_root = os.path.abspath(os.path.join(this_dir, os.pardir, os.pardir))
collect_ignore = [os.path.join(odl_root, 'setup.py')]

if not PYFFTW_AVAILABLE:
    collect_ignore.append(
        os.path.join(odl_root, 'odl/trafos/backends/pyfftw_bindings.py'))
if not PYWT_AVAILABLE:
    collect_ignore.append(
        os.path.join(odl_root, 'odl/trafos/backends/pywt_bindings.py'))
    # Currently `pywt` is the only implementation
    collect_ignore.append(
        os.path.join(odl_root, 'odl/trafos/wavelet.py'))


def pytest_addoption(parser):
    parser.addoption('--largescale', action='store_true',
                     help='Run large and slow tests')

    parser.addoption('--benchmark', action='store_true',
                     help='Run benchmarks')

    parser.addoption('--examples', action='store_true',
                     help='Run examples')

    parser.addoption('--doctest-doc', action='store_true',
                     help='Run doctests in the documentation')


# reusable fixtures
fn_impl_params = odl.FN_IMPLS.keys()
fn_impl_ids = [" impl='{}' ".format(p) for p in fn_impl_params]


@fixture(scope="module", ids=fn_impl_ids, params=fn_impl_params)
def fn_impl(request):
    """String with an available `FnBase` implementation name."""
    return request.param

ntuples_impl_params = odl.NTUPLES_IMPLS.keys()
ntuples_impl_ids = [" impl='{}' ".format(p) for p in ntuples_impl_params]


@fixture(scope="module", ids=ntuples_impl_ids, params=ntuples_impl_params)
def ntuples_impl(request):
    """String with an available `NtuplesBase` implementation name."""
    return request.param


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
