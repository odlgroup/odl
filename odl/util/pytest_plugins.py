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
from odl.space.entry_points import TENSOR_SET_IMPLS, TENSOR_SPACE_IMPLS
from odl.trafos.backends import PYFFTW_AVAILABLE, PYWT_AVAILABLE
from odl.util.testutils import simple_fixture
from odl.util.vectorization import OptionalArgDecorator


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

# Simple ones, use helper
tset_impl = simple_fixture(name='tset_impl',
                           params=TENSOR_SET_IMPLS.keys())
tspace_impl = simple_fixture(name='tspace_impl',
                             params=TENSOR_SPACE_IMPLS.keys())

floating_dtypes = np.sctypes['float'] + np.sctypes['complex']
floating_dtype_params = [np.dtype(dt) for dt in floating_dtypes]
floating_dtype = simple_fixture(name='dtype',
                                params=floating_dtype_params,
                                fmt=' {name} = np.{value.name} ')

scalar_dtypes = floating_dtype_params + np.sctypes['int'] + np.sctypes['uint']
scalar_dtype_params = [np.dtype(dt) for dt in floating_dtypes]
scalar_dtype = simple_fixture(name='dtype',
                              params=scalar_dtype_params,
                              fmt=' {name} = np.{value.name} ')

order = simple_fixture(name='order', params=['C', 'F', 'K'])


# More complicated ones with non-trivial documentation
ufunc_params = odl.util.ufuncs.UFUNCS
ufunc_ids = [' ufunc = {} '.format(p[0]) for p in ufunc_params]


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
reduction_ids = [' reduction = {} '.format(p[0]) for p in reduction_params]


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
arithmetic_op_ids = [" op = '{}' ".format(op)
                     for op in ['+', '/', '*', '-', '+=', '/=', '*=', '-=']]


@fixture(ids=arithmetic_op_ids, params=arithmetic_op_par)
def arithmetic_op(request):
    """An arithmetic operator, e.g. +, -, // etc."""
    return request.param
